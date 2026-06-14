#!/usr/bin/env python3
"""
make_figures.py -- builds the paper's full visual result set from results/.

  fig1_cost_vs_scale.pdf   $/1M safe queries vs traffic; API floors,
                           self-host curves per GPU, crossover markers
  fig2_saturation.pdf      req/s and p95 E2EL vs concurrency, per cell
                           (the evidence behind every qps_per_replica)
  fig3_energy.pdf          J/token at saturation per GPU x model;
                           secondary axis: safe queries per kWh
  fig4_pareto.pdf          detection F1 vs $/1M safe queries (log x);
                           the quality-cost frontier, one point per
                           mechanism x cheapest-feasible-GPU
  table_feasibility.csv    mechanism x GPU: qps/replica, SLO pass,
                           crossover qps, fits-in-VRAM

PDF output (vector) per *ACL camera-ready norms; PNG fallback alongside.
Single-column friendly: each figure legible at 3.3in width.
"""
import json, glob, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
SLO_E2EL_MS = 3000.0
SLO_TTFT_MS = 200.0

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

def load(pattern):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RES / pattern)))]

def cells():
    return [d for d in load("*.json")
            if "qps_per_replica" in d and d.get("fits") is not False]

def na_cells():
    return [d for d in load("*.json") if d.get("fits") is False]

def apis():
    return load("api_*.json")

def quality():
    return {q["model"]: q for q in load("quality_*.json")}

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png")
    plt.close(fig); print(f"  wrote figures/{name}.pdf/.png")

# ---------------------------------------------------------------- fig 2
def fig_saturation(cs):
    n = len(cs)
    if not n: return
    ncol = min(4, n); nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, d in zip(axes.flat, cs):
        rung = d["rungs"]; c = [r["concurrency"] for r in rung]
        ax.plot(c, [r["req_per_s"] for r in rung], "o-", lw=1.4,
                color="tab:blue", label="req/s")
        ax2 = ax.twinx()
        ax2.plot(c, [r["e2el_ms_p95"] for r in rung], "s--", lw=1.2,
                 color="tab:red", label="p95 E2EL")
        ax2.axhline(SLO_E2EL_MS, color="tab:red", ls=":", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xlabel("concurrency")
        ax.set_ylabel("req/s", color="tab:blue")
        ax2.set_ylabel("p95 ms", color="tab:red")
        ax.set_title(f'{d["gpu"]} · {d["model"]} · mt{d["max_tokens"]}')
    for ax in axes.flat[n:]: ax.axis("off")
    fig.suptitle("Throughput saturation and tail-latency blowup", y=1.02)
    save(fig, "fig2_saturation")

# ---------------------------------------------------------------- fig 3
def fig_energy(cs):
    cs = [d for d in cs if d.get("j_per_tok_saturated")]
    if not cs: print("  fig3 skipped (no energy data)"); return
    labels = [f'{d["gpu"]}\n{d["model"]} mt{d["max_tokens"]}' for d in cs]
    jt = [d["j_per_tok_saturated"] for d in cs]
    # queries/kWh at saturation: 3.6e6 J/kWh / (J/tok * tokens-per-query)
    qkwh = [3.6e6 / (d["j_per_tok_saturated"] *
            max(d.get("rungs", [{}])[-1].get("tok_per_s", 1) /
                max(d.get("rungs", [{}])[-1].get("req_per_s", 1), 1e-9), 1))
            for d in cs]
    x = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(0.65 * len(cs) + 2, 2.6))
    ax.bar(x, jt, color="tab:green", width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("J / output token (saturated)")
    ax2 = ax.twinx()
    ax2.plot(x, qkwh, "k^--", lw=1)
    ax2.set_ylabel("safe queries / kWh")
    ax.set_title("Energy cost of safety checks by hardware and mechanism")
    save(fig, "fig3_energy")

# ---------------------------------------------------------------- fig 1
def fig_cost(cs, api_list):
    qps_grid = np.logspace(-1, 4, 300)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(cs), 1)))
    for d, c in zip(cs, cmap):
        if not d.get("qps_per_replica"): continue
        price, qcap = d["gpu_price_usd_hr"], d["qps_per_replica"]
        y = [np.ceil(q / qcap) * price / (q * 3600) * 1e6 for q in qps_grid]
        ax.plot(qps_grid, y, lw=1.5, color=c,
                label=f'{d["model"]} mt{d["max_tokens"]} @ {d["gpu"]}')
    for a in api_list:
        v = a.get("usd_per_1m_queries_measured")
        if v:
            ax.axhline(v, ls="--", lw=1.2, color="gray")
            ax.text(qps_grid[-1], v, f'  {a["name"]} (API)', fontsize=6,
                    va="center", color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("traffic (safe queries / s)")
    ax.set_ylabel("USD / 1M safe queries")
    ax.set_title("Unit-economic cost vs. operational scale (measured)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=5.5, loc="upper right")
    save(fig, "fig1_cost_vs_scale")

# ---------------------------------------------------------------- fig 4
def fig_pareto(cs, api_list, qual):
    pts = []
    for d in cs:
        q = qual.get(d["model"])
        if not (q and q.get("f1") and d.get("qps_per_replica")): continue
        # cheapest cost at saturation for this cell
        cost = d["gpu_price_usd_hr"] / (d["qps_per_replica"] * 3600) * 1e6
        pts.append((cost, q["f1"], f'{d["model"]} @ {d["gpu"]}', "self-host"))
    for a in api_list:
        q = qual.get(a["model"])
        if q and q.get("f1") and a.get("usd_per_1m_queries_measured"):
            pts.append((a["usd_per_1m_queries_measured"], q["f1"],
                        a["name"], "API"))
    if not pts: print("  fig4 skipped (need quality + cost)"); return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for cost, f1, lab, kind in pts:
        m = "o" if kind == "self-host" else "D"
        ax.scatter(cost, f1, marker=m, s=40)
        ax.annotate(lab, (cost, f1), fontsize=5.5,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xscale("log")
    ax.set_xlabel("USD / 1M safe queries (saturated / measured)")
    ax.set_ylabel("unsafe-detection F1")
    ax.set_title("Quality–cost Pareto frontier of safety mechanisms")
    ax.grid(alpha=0.25)
    save(fig, "fig4_pareto")

# ---------------------------------------------------------------- table
def feasibility_table(cs, nas, api_list):
    rows = ["mechanism_cell,gpu,fits,qps_per_replica,ttft_slo_pass,"
            "cost_floor_usd_per_1m,crossover_vs_cheapest_api_qps"]
    cheapest_api = min((a["usd_per_1m_queries_measured"]
                        for a in api_list
                        if a.get("usd_per_1m_queries_measured")), default=None)
    for d in cs:
        qcap = d.get("qps_per_replica")
        floor = (d["gpu_price_usd_hr"] / (qcap * 3600) * 1e6) if qcap else None
        lat_ok = (d.get("latency_s_unloaded") or 9e9) * 1000 <= SLO_TTFT_MS * 10
        cross = (1e6 * d["gpu_price_usd_hr"] / (cheapest_api * 3600)
                 if cheapest_api else None)
        rows.append(f'{d["model"]}_mt{d["max_tokens"]},{d["gpu"]},yes,'
                    f'{qcap},{lat_ok},'
                    f'{floor:.2f},' + (f"{cross:.1f}" if cross else ""))
    for d in nas:
        rows.append(f'{d["model"]}_mt{d.get("max_tokens","")},{d["gpu"]},'
                    f'NO (vram),,,,')
    (OUT / "table_feasibility.csv").write_text("\n".join(rows))
    print("  wrote figures/table_feasibility.csv")

if __name__ == "__main__":
    cs, nas, ap, qu = cells(), na_cells(), apis(), quality()
    print(f"loaded {len(cs)} cells, {len(nas)} N/A, {len(ap)} APIs, "
          f"{len(qu)} quality files")
    fig_cost(cs, ap)
    fig_saturation(cs)
    fig_energy(cs)
    fig_pareto(cs, ap, qu)
    feasibility_table(cs, nas, ap)
