#!/usr/bin/env python3
"""
eval_quality.py -- detection quality for the SAME mechanisms the sweeps
profile, so every cost number can be paired with an F1.

Input prompt file format (TSV, one per line):   label<TAB>prompt
  label in {benign, borderline_safe, borderline_unsafe, harmful}
  (borderline_* split lets us report over-refusal on hard-but-safe inputs,
   the usability-loss axis the NFL paper emphasizes.)

Reports (guardrail-lit conventions: NFL, ThinkGuard, WildGuard):
  precision / recall / F1  on unsafe detection (binary)
  FPR on benign            (over-blocking rate)
  FPR on borderline_safe   (over-refusal on hard negatives)
  per-category recall
Writes results/quality_{model}_mt{max_tokens}.json
"""
import argparse, asyncio, json, time
from pathlib import Path

try:
    import aiohttp
except Exception:
    aiohttp = None

UNSAFE_LABELS = {"harmful", "borderline_unsafe"}

def parse_verdict(text):
    """Llama-Guard-style outputs: first line 'safe' or 'unsafe'."""
    t = (text or "").strip().lower()
    if t.startswith("unsafe"):
        return "unsafe"
    if t.startswith("safe"):
        return "safe"
    # generic judges: look for the word
    return "unsafe" if "unsafe" in t[:200] else "safe"

async def classify(session, url, model, prompt, max_tokens):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    async with session.post(url, json=payload) as resp:
        d = await resp.json()
        return parse_verdict(d["choices"][0]["message"]["content"])

async def run(args):
    rows = []
    for line in Path(args.prompts).read_text().splitlines():
        if "\t" in line:
            lab, p = line.split("\t", 1)
            rows.append((lab.strip(), p.strip()))
    if not rows:
        raise SystemExit("labeled prompts required: label<TAB>prompt")

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    preds = [None] * len(rows)

    async with aiohttp.ClientSession() as session:
        async def one(i, lab, p):
            async with sem:
                try:
                    preds[i] = await classify(session, url, args.model, p,
                                              args.max_tokens)
                except Exception:
                    preds[i] = "error"
        await asyncio.gather(*[one(i, l, p) for i, (l, p) in enumerate(rows)])

    tp = fp = fn = tn = 0
    cat = {}
    for (lab, _), pred in zip(rows, preds):
        truth_unsafe = lab in UNSAFE_LABELS
        pred_unsafe = pred == "unsafe"
        cat.setdefault(lab, [0, 0])
        cat[lab][1] += 1
        if pred_unsafe == truth_unsafe:
            cat[lab][0] += 1
        if truth_unsafe and pred_unsafe:    tp += 1
        elif truth_unsafe and not pred_unsafe: fn += 1
        elif not truth_unsafe and pred_unsafe: fp += 1
        else:                                tn += 1

    prec = tp / (tp + fp) if tp + fp else None
    rec  = tp / (tp + fn) if tp + fn else None
    f1   = (2 * prec * rec / (prec + rec)) if prec and rec else None
    n_benign = cat.get("benign", [0, 0])[1]
    n_bsafe  = cat.get("borderline_safe", [0, 0])[1]
    fpr_benign = ((n_benign - cat.get("benign", [0, 0])[0]) / n_benign
                  if n_benign else None)
    fpr_bsafe  = ((n_bsafe - cat.get("borderline_safe", [0, 0])[0]) / n_bsafe
                  if n_bsafe else None)

    out = {
        "model": args.model, "max_tokens": args.max_tokens,
        "n": len(rows), "n_error": sum(1 for p in preds if p == "error"),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr_benign": fpr_benign,            # over-blocking
        "fpr_borderline_safe": fpr_bsafe,    # over-refusal on hard negatives
        "per_category_acc": {k: v[0] / v[1] for k, v in cat.items() if v[1]},
    }
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    f = Path(args.outdir) / f"quality_{args.model.replace('/','_')}_mt{args.max_tokens}.json"
    f.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--prompts", default="prompts_labeled.tsv")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if aiohttp is None:
        raise SystemExit("pip install aiohttp")
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
