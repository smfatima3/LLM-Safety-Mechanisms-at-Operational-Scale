"""
PILOT / GO-NO-GO EXPERIMENT
===========================
Question this answers: do the four LLM-safety mechanism CLASSES separate
enough in $/1M-safe-queries to be worth a full study, and does a real
self-host-vs-API crossover exist at a realistic traffic scale?

This is NOT measurement. Every number below is seeded from the literature
we verified (sources noted inline). The real load-test harness replaces
these constants with measured values; the MODEL stays identical.

Two cost regimes:
  API        : cost is per-token, flat in volume.   cost/query = tok*price
  SELF-HOST  : cost is $/GPU-hour. cost/query falls as volume rises
               (idle GPU at low traffic; near a flat floor once saturated).

"Infeasible" is defined against explicit constraints, not raw QPS:
  - interactive SLO : single-request latency must be < SLO_MS
  - cost ceiling    : $/1M-safe-queries must be < COST_CEILING
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONSTRAINTS (edit to your deployment context)
# ----------------------------------------------------------------------
SLO_MS        = 200.0     # interactive latency budget (chat UX threshold)
COST_CEILING  = 100.0     # $ per 1M safe queries you're willing to pay

# ----------------------------------------------------------------------
# MECHANISM CLASSES  (numbers seeded from verified literature)
#   latency_s        : single-request service time (for SLO check)
#   tok_in / tok_out : tokens processed per safety check
#   api_in/api_out   : per-token price $/1M if called via hosted API
#   qps_per_replica  : SATURATED throughput of one GPU replica (batched)
#   gpu_hr           : $/hour for that replica's GPU
# Sources (from our searches):
#   classifier ~29-50ms            (generalanalysis; truefoundry Azure 52ms)
#   LLM-judge 5-11s                (generalanalysis GPT-5/mini; budecosystem 7-8.6s)
#   Groq 8B $0.05/1M, 70B $0.59/$0.79 (cloudzero/aipricing.guru)
#   GPT-4o-mini ~$0.15/1M          (project brief)
#   GPU: L40S ~$1/hr, A100 $2.90/hr (morphllm / runpod-class rates)
# ----------------------------------------------------------------------
MECHS = {
    "Provider API (moderation)": dict(
        regime="API", latency_s=0.10, tok_in=160, tok_out=10,
        api_in=0.15, api_out=0.60,            # GPT-4o-mini-class
        qps_per_replica=None, gpu_hr=None),

    "Small guard 1B (self-host or API)": dict(
        regime="BOTH", latency_s=0.04, tok_in=160, tok_out=10,
        api_in=0.05, api_out=0.08,            # Groq 8B-class floor
        qps_per_replica=80.0, gpu_hr=1.0),    # tiny model, cheap GPU

    "LLM-as-judge 8-70B": dict(
        regime="BOTH", latency_s=5.0, tok_in=160, tok_out=200,
        api_in=0.59, api_out=0.79,            # Groq 70B
        qps_per_replica=8.0, gpu_hr=2.9),     # batched on A100

    "Inference-time scaling (deliberative)": dict(
        regime="BOTH", latency_s=9.0, tok_in=160, tok_out=600,
        api_in=0.59, api_out=0.79,
        qps_per_replica=3.0, gpu_hr=2.9),     # long traces / multi-pass
}

# ----------------------------------------------------------------------
# COST MODEL
# ----------------------------------------------------------------------
def api_cost_per_million(m):
    """Flat: independent of volume."""
    return m["tok_in"] * m["api_in"] + m["tok_out"] * m["api_out"]

def selfhost_cost_per_million(m, qps):
    """Falls with volume; you add whole replicas as load grows."""
    replicas = np.ceil(qps / m["qps_per_replica"])
    cost_per_hr = replicas * m["gpu_hr"]
    queries_per_hr = qps * 3600.0
    return cost_per_hr / queries_per_hr * 1e6

def selfhost_floor(m):
    """Asymptotic $/1M at full GPU saturation."""
    return m["gpu_hr"] / (m["qps_per_replica"] * 3600.0) * 1e6

def crossover_qps(m):
    """Traffic level above which self-hosting (1 replica ramp) beats the API."""
    api = api_cost_per_million(m)
    # 1-replica self-host cost = gpu_hr/(qps*3600)*1e6 ; set == api
    return 1e6 * m["gpu_hr"] / (api * 3600.0)

# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
qps_grid = np.logspace(-1, 4, 400)   # 0.1 .. 10,000 qps

print("=" * 78)
print(f"{'MECHANISM':38} {'lat':>6} {'SLO':>5} {'API$/1M':>9} {'SH floor':>9} {'cross qps':>10}")
print("-" * 78)

summary = {}
for name, m in MECHS.items():
    api = api_cost_per_million(m)
    slo_ok = "PASS" if m["latency_s"] * 1000 < SLO_MS else "FAIL"
    if m["regime"] in ("BOTH",):
        floor = selfhost_floor(m)
        cross = crossover_qps(m)
        print(f"{name:38} {m['latency_s']:5.2f}s {slo_ok:>5} "
              f"{api:8.1f} {floor:8.1f} {cross:9.0f}")
    else:
        floor, cross = None, None
        print(f"{name:38} {m['latency_s']:5.2f}s {slo_ok:>5} "
              f"{api:8.1f} {'--':>8} {'--':>9}")
    summary[name] = dict(api=api, floor=floor, cross=cross, slo=slo_ok)
print("=" * 78)

# spread check
apis = [s["api"] for s in summary.values()]
floors = [s["floor"] for s in summary.values() if s["floor"]]
spread = max(apis + floors) / min(apis + floors)
print(f"\n$/1M spread across classes (best floor -> worst API): {spread:,.0f}x")
print(f"Interactive-SLO ({SLO_MS:.0f}ms) survivors: "
      f"{[n for n,s in summary.items() if s['slo']=='PASS']}")

# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.2))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(MECHS)))

for (name, m), c in zip(MECHS.items(), colors):
    api = api_cost_per_million(m)
    ax.axhline(api, color=c, ls="--", lw=1.4, alpha=0.9)
    ax.text(qps_grid[-1], api, f"  {name.split('(')[0].strip()} (API)",
            color=c, va="center", fontsize=7.5)
    if m["regime"] == "BOTH":
        sh = np.array([selfhost_cost_per_million(m, q) for q in qps_grid])
        ax.plot(qps_grid, sh, color=c, lw=2.2,
                label=f"{name.split('(')[0].strip()} (self-host)")
        cx = crossover_qps(m)
        ax.scatter([cx], [api], color=c, s=55, zorder=5, edgecolor="k", lw=0.6)

ax.axhline(COST_CEILING, color="crimson", ls=":", lw=1.6)
ax.text(0.12, COST_CEILING * 1.12, f"cost ceiling ${COST_CEILING:.0f}/1M",
        color="crimson", fontsize=8)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Arrival rate (safe queries / second)")
ax.set_ylabel("Cost per 1,000,000 safe queries (USD)")
ax.set_title("Safety-mechanism cost vs. operational scale\n"
             "(dashed = per-token API floor; solid = self-host; dots = crossover)",
             fontsize=11)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_xlim(qps_grid[0], qps_grid[-1] * 3.5)
fig.tight_layout()
fig.savefig("pilot_cost_curves.png", dpi=150)
print("\nSaved chart -> pilot_cost_curves.png")
