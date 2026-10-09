"""
Generate certified perturb-and-invert datapoints (Perturb & Invert classes, bins, and tools).

Usage (font clusters come from scripts/build_index.py --build_clusters):
  # evaluation set (Crello test split)
  python -m scripts.generate_data --split test --n 1000 --sampler coverage \
      --clusters data/font_clusters.json --out data/eval.json
  # ranker-training set (validation designs 300-499, disjoint from the 300 index designs)
  python -m scripts.generate_data --split validation --skip 300 --n 200 --sampler coverage \
      --clusters data/font_clusters.json --out data/rank_train.json

Prints certification pass rates per class and severity, the share of defects the reference-free
objective can see (phi_visible), and class-pair coverage.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.domains.cli import add_domain_args, load_designs
from verifix.perturb.coverage import coverage_rate
from verifix.perturb.fonts import FontClusters
from verifix.perturb.generate import generate
from verifix.perturb.operators import ALL_CLASSES



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k_lo", type=int, default=1)
    ap.add_argument("--k_hi", type=int, default=3)
    ap.add_argument("--mix", default="both", choices=["metadata", "tools", "both"])
    ap.add_argument("--sampler", default="coverage", choices=["random", "coverage"])
    ap.add_argument("--pool_per_design", type=int, default=4)
    ap.add_argument("--query_mode", default="inverse", choices=["inverse", "inverse_exact", "generic"],
                    help="user request: the inverse of the damage in words (default), with exact "
                         "values, or the generic 'fix the design problems' request")
    ap.add_argument("--clusters", default="data/font_clusters.json")
    ap.add_argument("--asset_dir", default="data/assets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_uncertified", action="store_true")
    ap.add_argument("--out", default="data/eval.json")
    add_domain_args(ap)
    args = ap.parse_args()

    designs, dom = load_designs(args, args.n, args.split)
    if os.path.exists(args.clusters):
        clusters = FontClusters.load(args.clusters)
    elif args.allow_synthetic:
        print("[fonts] WARNING: no cluster file; using the offline fallback clusters")
        clusters = FontClusters.fallback()
    else:
        raise SystemExit(f"{args.clusters} not found: run scripts.build_index --build_clusters first")

    dps = generate(designs, k_range=(args.k_lo, args.k_hi), seed=args.seed, mix=args.mix,
                   sampler=args.sampler, pool_per_design=args.pool_per_design, clusters=clusters,
                   query_mode=args.query_mode, keep_only_certified=not args.keep_uncertified,
                   fonts_dir=args.fonts_dir)

    stats = defaultdict(lambda: defaultdict(int))
    for dp in dps:
        for c, s in zip(dp.pclasses, dp.severities):
            key = f"{c}:{s}"
            stats[key]["n"] += 1
            for k in ("invertible", "violates", "visible", "phi_visible"):
                stats[key][k] += int(bool(dp.certificate.get(k)))
    print(f"\n{'class:severity':34}{'n':>6}{'invert':>8}{'violate':>9}{'visible':>9}{'phi_sees':>9}")
    for key in sorted(stats):
        s = stats[key]
        print(f"{key:34}{s['n']:>6}{s['invertible']:>8}{s['violates']:>9}{s['visible']:>9}{s['phi_visible']:>9}")
    print(f"class-pair coverage: {coverage_rate([d.pclasses for d in dps], ALL_CLASSES):.1%}")
    from collections import Counter as _C
    print("defects per case:", dict(sorted(_C(len(d.pclasses) for d in dps).items())))
    for d in dps[:3]:
        print("  example request:", d.query)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([dp.to_dict(args.asset_dir) for dp in dps], f)
    print(f"wrote {len(dps)} datapoints -> {args.out}")
    json.dump(dom.last_stats, open(str(Path(args.out).with_suffix("")) + "_load_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
