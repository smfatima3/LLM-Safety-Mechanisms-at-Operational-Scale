#!/usr/bin/env python3
"""
api_probe.py -- the API-regime measurements the cost model currently assumes.

For each configured hosted endpoint, sends the prompt set at modest
concurrency and records: e2el p50/p95/p99, exact prompt/completion token
usage (so $/query = usage x published price is MEASURED, not estimated),
and error rate. Hosted APIs can't be saturated by us (rate limits), so no
saturation sweep here -- per the paper's framing, the API regime's binding
constraint is price, not a throughput wall we can find.

Config: endpoints.json
[
  {"name":"openai_moderation", "kind":"moderation",
   "url":"https://api.openai.com/v1/moderations",
   "model":"omni-moderation-latest", "env_key":"OPENAI_API_KEY",
   "usd_per_1m_in":0.0, "usd_per_1m_out":0.0},
  {"name":"groq_guard8b", "kind":"chat",
   "url":"https://api.groq.com/openai/v1/chat/completions",
   "model":"meta-llama/llama-guard-3-8b", "env_key":"GROQ_API_KEY",
   "usd_per_1m_in":0.20, "usd_per_1m_out":0.20}
]
Writes results/api_{name}.json
"""
import argparse, asyncio, json, os, time
from pathlib import Path

try:
    import aiohttp
except Exception:
    aiohttp = None

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

async def probe_endpoint(ep, prompts, concurrency, repeats):
    key = os.environ.get(ep.get("env_key", ""), "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    lat, t_in, t_out, errs = [], [], [], 0
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=headers) as s:
        async def one(prompt):
            nonlocal errs
            async with sem:
                t0 = time.perf_counter()
                try:
                    if ep["kind"] == "moderation":
                        payload = {"model": ep["model"], "input": prompt}
                    else:
                        payload = {"model": ep["model"], "max_tokens": 16,
                                   "temperature": 0.0,
                                   "messages": [{"role": "user",
                                                 "content": prompt}]}
                    async with s.post(ep["url"], json=payload) as r:
                        d = await r.json()
                        if r.status != 200:
                            errs += 1
                            return
                        lat.append((time.perf_counter() - t0) * 1000.0)
                        u = d.get("usage") or {}
                        t_in.append(u.get("prompt_tokens", 0))
                        t_out.append(u.get("completion_tokens", 0))
                except Exception:
                    errs += 1
        for _ in range(repeats):
            await asyncio.gather(*[one(p) for p in prompts])

    mean_in = sum(t_in) / len(t_in) if t_in else 0
    mean_out = sum(t_out) / len(t_out) if t_out else 0
    usd_per_q = (mean_in * ep["usd_per_1m_in"]
                 + mean_out * ep["usd_per_1m_out"]) / 1e6
    return {
        "name": ep["name"], "model": ep["model"], "kind": ep["kind"],
        "n_ok": len(lat), "n_err": errs,
        "e2el_ms_p50": pct(lat, 50), "e2el_ms_p95": pct(lat, 95),
        "e2el_ms_p99": pct(lat, 99),
        "mean_tokens_in": round(mean_in, 1),
        "mean_tokens_out": round(mean_out, 1),
        "usd_per_query_measured": usd_per_q,
        "usd_per_1m_queries_measured": usd_per_q * 1e6,
    }

async def run(args):
    eps = json.loads(Path(args.endpoints).read_text())
    prompts = [l.split("\t", 1)[-1].strip()
               for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for ep in eps:
        res = await probe_endpoint(ep, prompts, args.concurrency, args.repeats)
        f = Path(args.outdir) / f"api_{ep['name']}.json"
        f.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="endpoints.json")
    ap.add_argument("--prompts", default="prompts_labeled.tsv")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5,
                    help="repeat passes over the prompt set for stable tails")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if aiohttp is None:
        raise SystemExit("pip install aiohttp")
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
