"""
Generate certified perturb-and-invert datapoints (Perturb & Invert classes, bins, and tools).

Usage:
  # 1. font clusters from the INDEX split (built once, reused for evaluation)
  python -m scripts.generate_data --split validation --n 300 --out data/index_designs.json \
      --clusters data/font_clusters.json --build_clusters
  python -m scripts.fetch_fonts --clusters data/font_clusters.json --out fonts
  # 2. evaluation set
  python -m scripts.generate_data --split test --n 1000 --k_lo 1 --k_hi 3 --mix both \
      --sampler coverage --clusters data/font_clusters.json --out data/eval.json

Prints certification pass rates per class and severity, the share of defects the reference-free
objective can see (phi_visible), and class-pair coverage.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from verifix.domains.graphic_design import GraphicDesignDomain
from verifix.perturb.coverage import coverage_rate
from verifix.perturb.fonts import FontClusters
from verifix.perturb.generate import generate
from verifix.perturb.operators import ALL_CLASSES

DOMAINS = {"graphic_design": GraphicDesignDomain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="graphic_design", choices=list(DOMAINS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k_lo", type=int, default=1)
    ap.add_argument("--k_hi", type=int, default=3)
    ap.add_argument("--mix", default="both", choices=["metadata", "tools", "both"])
    ap.add_argument("--sampler", default="random", choices=["random", "coverage"])
    ap.add_argument("--pool_per_design", type=int, default=4)
    ap.add_argument("--query_mode", default="generic", choices=["generic", "specific"])
    ap.add_argument("--clusters", default="data/font_clusters.json")
    ap.add_argument("--build_clusters", action="store_true",
                    help="build font clusters from these designs and save them to --clusters")
    ap.add_argument("--fonts_dir", default="fonts")
    ap.add_argument("--asset_dir", default="data/assets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_uncertified", action="store_true")
    ap.add_argument("--out", default="data/eval.json")
    args = ap.parse_args()

    designs = DOMAINS[args.domain]().load(args.n, split=args.split)
    if args.build_clusters:
        clusters = FontClusters.from_designs(designs)
        Path(args.clusters).parent.mkdir(parents=True, exist_ok=True)
        clusters.save(args.clusters)
        print(f"[fonts] {len(clusters.fonts)} fonts in {clusters.centroids.shape[0]} clusters -> {args.clusters}")
    elif os.path.exists(args.clusters):
        clusters = FontClusters.load(args.clusters)
    else:
        print("[fonts] WARNING: no cluster file; using the offline fallback clusters")
        clusters = FontClusters.fallback()

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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([dp.to_dict(args.asset_dir) for dp in dps], f)
    print(f"wrote {len(dps)} datapoints -> {args.out}")


if __name__ == "__main__":
    main()
