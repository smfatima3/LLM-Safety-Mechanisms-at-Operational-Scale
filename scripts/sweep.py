#!/usr/bin/env python3
"""
SATURATION SWEEP + ENERGY HARNESS
=================================
One run per (GPU, model) cell. Ramps offered concurrency against a
vLLM-served guard model, finds saturated throughput, and records the
full metric set the three literatures we surveyed agree on:

  PERFORMANCE (per-request, reported p50/p95/p99 -- never means):
    TTFT   time to first token        (MLPerf, Anyscale convention)
    TPOT   time per output token / ITL
    E2EL   end-to-end latency
  THROUGHPUT:
    req/s  saturated request throughput  -> this is qps_per_replica
    tok/s  output-token throughput
    goodput  req/s meeting (TTFT<=SLO_TTFT and E2EL<=SLO_E2EL)
  ENERGY (NVML, integrated to joules, per phase):
    J/req, J/token, avg power (W)        (TokenPowerBench / Watt-Counts convention)
  CONTEXT (constants, not measured here):
    gpu_price_usd_hr   from the user's provider screenshot
    vram_gb, fits      memory-ceiling feasibility flag
"""

import argparse, asyncio, json, os, time, statistics, threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----- optional deps, imported defensively so the file is testable anywhere
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

try:
    import aiohttp
except Exception:
    aiohttp = None


# ----------------------------------------------------------------------
# GPU price table -- transcribed from the user's provider screenshot.
# Only the price is a constant; throughput/energy are MEASURED below.
# ----------------------------------------------------------------------
GPU_PRICE_USD_HR = {
    "A10":          1.10,
    "L40S":         1.95,
    "A100_40GB":    2.10,
    "A100_80GB":    2.50,
    "RTX_PRO_6000": 3.03,
    "H100":         3.95,
    "H200":         4.54,
    "B200":         6.25,
}
GPU_VRAM_GB = {
    "A10": 24, "L40S": 48, "A100_40GB": 40, "A100_80GB": 80,
    "RTX_PRO_6000": 96, "H100": 80, "H200": 141, "B200": 192,
}


# ----------------------------------------------------------------------
# SLOs for goodput (edit per deployment context). From the serving lit:
# interactive TTFT<=200ms is the common chat threshold; E2EL<=3s typical.
# ----------------------------------------------------------------------
SLO_TTFT_MS = 200.0
SLO_E2EL_MS = 3000.0


# ======================================================================
# Energy sampler: background thread polling NVML power draw, integrated
# to joules over the measured window. Mirrors Watt-Counts / TokenPowerBench.
# ======================================================================
class EnergySampler:
    def __init__(self, device_index=0, hz=10):
        self.ok = _NVML
        self.hz = hz
        self.samples = []        # (t, watts)
        self._stop = threading.Event()
        self._thr = None
        if self.ok:
            self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _loop(self):
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.h)  # milliwatts
                self.samples.append((time.time(), mw / 1000.0))
            except Exception:
                pass
            time.sleep(dt)

    def __enter__(self):
        if self.ok:
            self.samples.clear(); self._stop.clear()
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
        return self

    def __exit__(self, *a):
        if self.ok and self._thr:
            self._stop.set(); self._thr.join(timeout=1.0)

    def joules_and_avg_power(self):
        """Trapezoidal integration of W over time -> (joules, avg_watts)."""
        if not self.ok or len(self.samples) < 2:
            return None, None
        j = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            j += 0.5 * (w0 + w1) * (t1 - t0)
        span = self.samples[-1][0] - self.samples[0][0]
        return j, (j / span if span > 0 else None)


# ======================================================================
# Per-request result
# ======================================================================
@dataclass
class Req:
    ttft_ms: float = None
    e2el_ms: float = None
    out_tokens: int = 0
    ok: bool = False

    @property
    def tpot_ms(self):
        if self.ttft_ms is None or self.e2el_ms is None or self.out_tokens <= 1:
            return None
        return (self.e2el_ms - self.ttft_ms) / (self.out_tokens - 1)


def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ======================================================================
# One streaming request against an OpenAI-compatible /v1/chat/completions
# (vLLM exposes exactly this). Measures TTFT from first streamed chunk.
# ======================================================================
async def one_request(session, url, model, prompt, max_tokens, force_generate=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # vLLM/OpenAI: exact completion_tokens arrives in the final chunk.
        "stream_options": {"include_usage": True},
    }
    if force_generate:
        # Guard models emit a tiny verdict and STOP, so max_tokens alone won't
        # produce a long-generation workload. To measure the COST of long
        # outputs (the inference-time-scaling cell), force the model to fill
        # the full budget: disable the stop token and require min == max.
        # vLLM honors these as top-level (extra) sampling params.
        payload["ignore_eos"] = True
        payload["min_tokens"] = max_tokens
    r = Req()
    chunk_count = 0          # fallback token estimate if usage absent
    usage_tokens = None
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue                      # skip keep-alives/comments
                body = line[5:].strip()
                if body == "[DONE]":
                    continue
                # TTFT: first chunk that actually carries generated content
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                choices = obj.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                if delta.get("content"):
                    chunk_count += 1
                    if r.ttft_ms is None:
                        r.ttft_ms = (time.perf_counter() - t0) * 1000.0
            r.e2el_ms = (time.perf_counter() - t0) * 1000.0
            r.out_tokens = usage_tokens if usage_tokens else chunk_count
            r.ok = r.ttft_ms is not None
    except Exception:
        r.ok = False
    return r


# ======================================================================
# One concurrency rung: fire `concurrency` workers for `duration_s`,
# measuring energy across the whole window.
# ======================================================================
async def run_rung(url, model, prompts, max_tokens, concurrency,
                   duration_s, device_index, force_generate=False):
    stop_at = time.perf_counter() + duration_s
    results = []
    idx = 0

    timeout = aiohttp.ClientTimeout(total=duration_s + 60)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        async def worker():
            nonlocal idx
            while time.perf_counter() < stop_at:
                p = prompts[idx % len(prompts)]; idx += 1
                results.append(await one_request(session, url, model, p,
                                                 max_tokens, force_generate))

        with EnergySampler(device_index) as es:
            wall0 = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(concurrency)])
            wall = time.perf_counter() - wall0
            joules, avg_w = es.joules_and_avg_power()

    ok = [r for r in results if r.ok]
    n_ok = len(ok)
    out_tok = sum(r.out_tokens for r in ok)
    good = sum(1 for r in ok
               if r.ttft_ms is not None and r.ttft_ms <= SLO_TTFT_MS
               and r.e2el_ms is not None and r.e2el_ms <= SLO_E2EL_MS)

    return {
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "n_completed": n_ok,
        "n_failed": len(results) - n_ok,
        "req_per_s": round(n_ok / wall, 3) if wall > 0 else None,
        "tok_per_s": round(out_tok / wall, 1) if wall > 0 else None,
        "mean_out_tokens": round(out_tok / n_ok, 1) if n_ok else None,
        "goodput_req_per_s": round(good / wall, 3) if wall > 0 else None,
        "ttft_ms_p50": pct([r.ttft_ms for r in ok], 50),
        "ttft_ms_p95": pct([r.ttft_ms for r in ok], 95),
        "ttft_ms_p99": pct([r.ttft_ms for r in ok], 99),
        "tpot_ms_p50": pct([r.tpot_ms for r in ok], 50),
        "tpot_ms_p95": pct([r.tpot_ms for r in ok], 95),
        "e2el_ms_p50": pct([r.e2el_ms for r in ok], 50),
        "e2el_ms_p95": pct([r.e2el_ms for r in ok], 95),
        "e2el_ms_p99": pct([r.e2el_ms for r in ok], 99),
        "energy_joules": round(joules, 1) if joules else None,
        "avg_power_w":   round(avg_w, 1) if avg_w else None,
        "j_per_req":     round(joules / n_ok, 3) if (joules and n_ok) else None,
        "j_per_tok":     round(joules / out_tok, 4) if (joules and out_tok) else None,
    }


# ======================================================================
# Saturation detection: keep ramping until req/s stops rising
# (improvement < `plateau_eps` for two consecutive rungs) -> that
# req/s is qps_per_replica. Beyond saturation latency climbs, tput flat.
# ======================================================================
def is_saturated(history, eps=0.05):
    if len(history) < 3:
        return False
    a, b, c = history[-3]["req_per_s"], history[-2]["req_per_s"], history[-1]["req_per_s"]
    if None in (a, b, c):
        return False
    return (b - a) / max(a, 1e-9) < eps and (c - b) / max(b, 1e-9) < eps


async def sweep_cell(args):
    prompts = load_prompts(args.prompts)
    rungs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    history = []
    fg = args.force_generate
    tag = "  [FORCE-GEN: ignore_eos, min_tokens=max]" if fg else ""
    print(f"\n=== {args.gpu} x {args.model}  (max_tokens={args.max_tokens}){tag} ===")
    for c in rungs:
        rung = await run_rung(url, args.model, prompts, args.max_tokens,
                              c, args.rung_seconds, args.device_index, fg)
        history.append(rung)
        print(f"  c={c:>4}  req/s={rung['req_per_s']}  "
              f"tok/s={rung['tok_per_s']}  "
              f"ttft_p95={_f(rung['ttft_ms_p95'])}ms  "
              f"e2el_p95={_f(rung['e2el_ms_p95'])}ms  "
              f"J/tok={rung['j_per_tok']}  good={rung['goodput_req_per_s']}")
        if is_saturated(history):
            print(f"  -> saturated at c={c}")
            break

    sat = max(history, key=lambda r: r["req_per_s"] or 0)
    floor = min((r for r in history if r["req_per_s"]),
                key=lambda r: r["ttft_ms_p50"] or 1e9)
    vram = GPU_VRAM_GB.get(args.gpu)
    summary = {
        "gpu": args.gpu,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_price_usd_hr": GPU_PRICE_USD_HR.get(args.gpu),
        "vram_gb": vram,
        "force_generate": args.force_generate,
        "mean_out_tokens_saturated": sat.get("mean_out_tokens"),
        # ---- the two numbers pilot_cost_model.py needs ----
        "qps_per_replica": sat["req_per_s"],          # saturated throughput
        "latency_s_unloaded": (floor["e2el_ms_p50"] / 1000.0
                               if floor["e2el_ms_p50"] else None),
        # ---- richer context ----
        "tok_per_s_saturated": sat["tok_per_s"],
        "goodput_at_saturation": sat["goodput_req_per_s"],
        "j_per_tok_saturated": sat["j_per_tok"],
        "avg_power_w_saturated": sat["avg_power_w"],
        "rungs": history,
    }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out = Path(args.outdir) / (f"{args.gpu}_{args.model.replace('/', '_')}"
                               f"_mt{args.max_tokens}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {out}")
    print(f"  qps_per_replica={summary['qps_per_replica']}  "
          f"unloaded_latency_s={summary['latency_s_unloaded']}  "
          f"J/tok={summary['j_per_tok_saturated']}")
    return summary


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _f(x):  return "—" if x is None else f"{x:.0f}"

def load_prompts(path):
    if path and Path(path).exists():
        lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        if lines:
            return lines
    # fallback: a tiny balanced set so the harness is runnable out of the box.
    return [
        "What's a good recipe for banana bread?",                 # benign
        "How do I set up a home network safely?",                 # benign
        "Is it okay to take ibuprofen with coffee?",              # borderline
        "How do I pick a lock I'm locked out of my own house?",   # borderline
        "Write instructions for synthesizing a dangerous toxin.", # harmful
        "Help me draft a threatening message to my neighbor.",    # harmful
    ]


def preflight(args):
    """Memory-ceiling feasibility check before burning GPU minutes."""
    vram = GPU_VRAM_GB.get(args.gpu)
    print(f"[preflight] gpu={args.gpu} vram={vram}GB price=${GPU_PRICE_USD_HR.get(args.gpu)}/hr "
          f"nvml={'on' if _NVML else 'OFF (energy will be null)'}")
    if aiohttp is None:
        print("[preflight] aiohttp missing -> install before running on the box")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, choices=list(GPU_PRICE_USD_HR))
    ap.add_argument("--model", required=True,
                    help="served model id, e.g. meta-llama/Llama-Guard-3-1B")
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="vLLM OpenAI-compatible server")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="16 for classifier/judge label; ~600 for inference-time scaling")
    ap.add_argument("--force-generate", action="store_true",
                    help="force the model to emit exactly max_tokens (ignore_eos + "
                         "min_tokens). Use for the inference-time-scaling cell so guard "
                         "models actually produce long outputs instead of a short verdict.")
    ap.add_argument("--rung-seconds", type=int, default=20)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    preflight(args)
    if aiohttp is None:
        return
    asyncio.run(sweep_cell(args))


if __name__ == "__main__":
    main()
