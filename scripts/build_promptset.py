#!/usr/bin/env python3
"""
build_promptset.py -- assemble a reproducible labeled prompt set from
established, citable benchmarks. No hand-authored prompts: every line
carries a published label and a provenance id so the set is defensible
and exactly reproducible (fixed seed).

Source -> bucket mapping (the 4 labels eval_quality.py consumes):
  benign            <- Alpaca instructions, no-input (Taori et al. 2023)
  borderline_safe   <- XSTest safe prompts  (Roettger et al. NAACL 2024)
                       (scary-sounding but safe -> tests OVER-REFUSAL)
  borderline_unsafe <- XSTest unsafe prompts (the contrast set: clearly
                       unsafe but lexically close to the safe ones)
  harmful           <- AdvBench harmful_behaviors (Zou et al. 2023)

Output columns (TSV): label <TAB> prompt
A sidecar prompts_provenance.tsv keeps: label, source, source_id, prompt
so reviewers can trace every item.

Default mix (200 total) is balanced for statistical signal per bucket
without overspending the 20-min latency budget:
  benign 60, borderline_safe 50, borderline_unsafe 40, harmful 50
"""
import argparse, csv, json, random, hashlib
from pathlib import Path

def load_xstest(path):
    safe, unsafe = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = (row["id"], row["prompt"].strip())
            (safe if row["label"].strip() == "safe" else unsafe).append(item)
    return safe, unsafe

def load_advbench(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            g = row["goal"].strip()
            if g:
                out.append((f"adv{i}", g))
    return out

def load_alpaca(path):
    out = []
    data = json.load(open(path, encoding="utf-8"))
    for i, x in enumerate(data):
        if not x.get("input") and x.get("instruction"):
            out.append((f"alp{i}", x["instruction"].strip()))
    return out

def sample(items, n, rng):
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xstest", default="promptset/xstest.csv")
    ap.add_argument("--advbench", default="promptset/advbench.csv")
    ap.add_argument("--alpaca", default="promptset/alpaca.json")
    ap.add_argument("--n-benign", type=int, default=60)
    ap.add_argument("--n-borderline-safe", type=int, default=50)
    ap.add_argument("--n-borderline-unsafe", type=int, default=40)
    ap.add_argument("--n-harmful", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--out", default="prompts_labeled.tsv")
    ap.add_argument("--provenance", default="prompts_provenance.tsv")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    xs_safe, xs_unsafe = load_xstest(args.xstest)
    adv = load_advbench(args.advbench)
    alp = load_alpaca(args.alpaca)

    buckets = [
        ("benign",            "alpaca",   sample(alp, args.n_benign, rng)),
        ("borderline_safe",   "xstest",   sample(xs_safe, args.n_borderline_safe, rng)),
        ("borderline_unsafe", "xstest",   sample(xs_unsafe, args.n_borderline_unsafe, rng)),
        ("harmful",           "advbench", sample(adv, args.n_harmful, rng)),
    ]

    rows = []           # (label, prompt)
    prov = []           # (label, source, source_id, prompt)
    seen = set()
    for label, source, items in buckets:
        for sid, text in items:
            text = " ".join(text.split())           # normalize whitespace
            h = hashlib.md5(text.lower().encode()).hexdigest()
            if h in seen or "\t" in text or not text:
                continue                            # dedup + TSV-safety
            seen.add(h)
            rows.append((label, text))
            prov.append((label, source, sid, text))

    rng.shuffle(rows)   # mix buckets so load isn't ordered by difficulty

    Path(args.out).write_text(
        "\n".join(f"{lab}\t{txt}" for lab, txt in rows) + "\n", encoding="utf-8")
    with open(args.provenance, "w", encoding="utf-8") as f:
        f.write("label\tsource\tsource_id\tprompt\n")
        for r in prov:
            f.write("\t".join(r) + "\n")

    # report
    from collections import Counter
    c = Counter(lab for lab, _ in rows)
    print(f"seed={args.seed}  total={len(rows)}")
    for k in ("benign", "borderline_safe", "borderline_unsafe", "harmful"):
        print(f"  {k:18} {c.get(k,0)}")
    print(f"wrote {args.out} and {args.provenance}")
    print("\nNOTE for paper: prompts sampled from XSTest (CC-BY-4.0; "
          "Roettger et al. 2024), AdvBench (Zou et al. 2023), and "
          "Alpaca (Taori et al. 2023). Unsafe prompts are used only as "
          "classifier inputs; no harmful content is generated.")

if __name__ == "__main__":
    main()
