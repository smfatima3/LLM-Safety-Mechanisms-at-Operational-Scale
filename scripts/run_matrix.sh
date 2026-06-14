#!/usr/bin/env bash
# =====================================================================
# run_matrix.sh  --  full {model-size} x {GPU} sweep in one sitting
# =====================================================================
# Assumes: one physical GPU of each type, addressable by CUDA index,
# OR you run this once per rented GPU box (recommended for clean power
# numbers -- co-tenant GPUs pollute NVML readings).
#
# CRITICAL: pre-cache all weights BEFORE the paid clock starts:
#   huggingface-cli download meta-llama/Llama-Guard-3-1B
#   huggingface-cli download meta-llama/Llama-Guard-3-8B
# Downloading inside the 20-min window wastes paid GPU time.
#
# Mechanism -> model-size -> max_tokens mapping:
#   small guard        : Llama-Guard-3-1B   max_tokens=16
#   LLM-as-judge        : Llama-Guard-3-8B   max_tokens=16
#   inference-time scale: Llama-Guard-3-8B   max_tokens=600  (long trace)
# The 8B/600 cell can be DERIVED from 8B/16 decode rate + a spot check,
# so it is optional if time is tight.
# =====================================================================
set -euo pipefail

GPU="${1:?usage: run_matrix.sh <GPU_NAME from price table>}"
PORT="${PORT:-8000}"
RUNG_SECONDS="${RUNG_SECONDS:-20}"

# (model_id, served_name, max_tokens, min_vram_gb_fp16)
MODELS=(
  "meta-llama/Llama-Guard-3-1B|guard1b|16|6"
  "meta-llama/Llama-Guard-3-8B|guard8b|16|18"
  "meta-llama/Llama-Guard-3-8B|guard8b|600|18"   # inference-time scaling proxy
)

# crude VRAM lookup mirrors sweep.py's table
declare -A VRAM=( [A10]=24 [L40S]=48 [A100_40GB]=40 [A100_80GB]=80
                  [RTX_PRO_6000]=96 [H100]=80 [H200]=141 [B200]=192 )

vram="${VRAM[$GPU]}"
mkdir -p results

# --- Sidestep FlashInfer's JIT kernel compile at startup (needs CUDA dev
# --- headers like curand.h that slim Modal/cloud images lack). Use the
# --- Torch-native sampler instead: no compile, identical results, more
# --- reproducible. Record this in the paper's methods section.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "=== matrix on $GPU (${vram}GB) | FlashInfer sampler OFF ==="

for entry in "${MODELS[@]}"; do
  IFS="|" read -r model_id served max_tok min_vram <<< "$entry"

  if (( vram < min_vram )); then
    echo "[SKIP] $served needs ${min_vram}GB > ${vram}GB on $GPU  (memory-ceiling N/A -- this is a finding)"
    echo "{\"gpu\":\"$GPU\",\"model\":\"$served\",\"max_tokens\":$max_tok,\"fits\":false,\"reason\":\"vram_ceiling\"}" \
        > "results/${GPU}_${served}_${max_tok}_NA.json"
    continue
  fi

  echo "--- serving $model_id (max_tokens=$max_tok) ---"
  # launch vLLM; --max-model-len kept small for guard tasks to free KV cache.
  # --enforce-eager skips CUDA-graph capture (another startup JIT path that
  # fails on header-less images); --gpu-memory-utilization gives KV headroom.
  python -m vllm.entrypoints.openai.api_server \
      --model "$model_id" --served-model-name "$served" \
      --port "$PORT" --max-model-len 2048 \
      --gpu-memory-utilization 0.90 --enforce-eager \
      > "vllm_${GPU}_${served}.log" 2>&1 &
  VLLM_PID=$!

  # wait for readiness (cap 120s; weights are pre-cached so this is fast)
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
    sleep 2
  done

  python harness/sweep.py \
      --gpu "$GPU" --model "$served" \
      --base-url "http://localhost:${PORT}" \
      --prompts prompts.txt --max-tokens "$max_tok" \
      --rung-seconds "$RUNG_SECONDS" --device-index 0 \
      --outdir results || echo "[warn] sweep failed for $served/$max_tok"

  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  sleep 3   # let power settle before next cell
done

echo "=== done $GPU -- results in results/ ==="
