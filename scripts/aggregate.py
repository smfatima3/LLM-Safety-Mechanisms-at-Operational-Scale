#!/usr/bin/env python3
"""
aggregate.py  --  turn measured sweep results into the cost model's inputs.

Reads results/*.json (one per GPU x model cell), builds a tidy table, and
emits measured_constants.py -- a drop-in replacement for the seeded MECHS
dict in pilot_cost_model.py. Also prints the headline crossover/feasibility
table so you can eyeball whether the measured numbers preserve the 164x
separation the pilot predicted.
"""
import json, glob, os
from pathlib import Path

RESULTS = "results"

def load_cells():
    cells = []
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        name = os.path.basename(f)
        # only sweep cells belong here; quality_*.json and api_*.json have
        # a different schema (no "gpu") and are consumed by make_figures.py
        if name.startswith("quality_") or name.startswith("api_"):
            continue
        d = json.loads(Path(f).read_text())
        if "gpu" not in d:                 # belt-and-suspenders: skip anything
            continue                       # without the sweep-cell schema
        cells.append(d)
    return cells

def main():
    cells = load_cells()
    if not cells:
        print("No results yet. Run the matrix on the GPU box first.")
        return

    # tidy print
    hdr = ("GPU", "model", "max_tok", "$/hr", "vram", "qps/rep",
           "lat_s", "tok/s", "J/tok", "goodput")
    print("{:<13}{:<9}{:>8}{:>7}{:>6}{:>9}{:>8}{:>9}{:>9}{:>9}".format(*hdr))
    print("-" * 96)
    fitted = []
    for d in sorted(cells, key=lambda x: (x.get("gpu", ""), x.get("model", ""))):
        if d.get("fits") is False:
            print("{:<13}{:<9}{:>8}{:>7}{:>6}{:>9}".format(
                d["gpu"], d["model"], d.get("max_tokens", "-"),
                "-", "-", "N/A (vram)"))
            continue
        fitted.append(d)
        print("{:<13}{:<9}{:>8}{:>7.2f}{:>6}{:>9}{:>8}{:>9}{:>9}{:>9}".format(
            d["gpu"], d["model"], d.get("max_tokens", "-"),
            d.get("gpu_price_usd_hr") or 0, d.get("vram_gb") or 0,
            _s(d.get("qps_per_replica")), _s(d.get("latency_s_unloaded")),
            _s(d.get("tok_per_s_saturated")), _s(d.get("j_per_tok_saturated")),
            _s(d.get("goodput_at_saturation"))))

    # emit measured_constants.py for the cost model
    lines = ["# AUTO-GENERATED from measured sweeps. Feeds pilot_cost_model.py.",
             "MEASURED = {"]
    for d in fitted:
        key = f'{d["gpu"]}::{d["model"]}::{d.get("max_tokens")}'
        lines.append(f'    "{key}": dict('
                     f'gpu_hr={d.get("gpu_price_usd_hr")}, '
                     f'qps_per_replica={d.get("qps_per_replica")}, '
                     f'latency_s={d.get("latency_s_unloaded")}, '
                     f'j_per_tok={d.get("j_per_tok_saturated")}),')
    lines.append("}")
    Path("measured_constants.py").write_text("\n".join(lines) + "\n")
    print("\nWrote measured_constants.py "
          f"({len(fitted)} fitted cells, {len(cells)-len(fitted)} N/A).")

    # quick separation check
    qps = [d["qps_per_replica"] for d in fitted if d.get("qps_per_replica")]
    if qps:
        print(f"Saturated-throughput spread across cells: "
              f"{max(qps)/min(qps):.1f}x  (min={min(qps)}, max={max(qps)})")

def _s(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

if __name__ == "__main__":
    main()
