"""
Build the verified-fix index from the INDEX split (disjoint from evaluation).

Usage:
  python -m scripts.build_index --split validation --n 300 --clusters data/font_clusters.json --out data/index.json
  python -m scripts.build_index --split validation --n 300 --clusters data/font_clusters.json --synthetic --out data/index_syn.json
"""
import argparse
import os
from pathlib import Path

from verifix.domains.graphic_design import GraphicDesignDomain
from verifix.perturb.fonts import FontClusters
from verifix.retrieval.build import build_index

DOMAINS = {"graphic_design": GraphicDesignDomain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="graphic_design", choices=list(DOMAINS))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--clusters", default="data/font_clusters.json")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="data/index.json")
    args = ap.parse_args()

    designs = DOMAINS[args.domain]().load(args.n, split=args.split)
    clusters = FontClusters.load(args.clusters) if os.path.exists(args.clusters) else None
    idx = build_index(designs, synthetic=args.synthetic, clusters=clusters)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.out)
    print("index entries per class:", idx.stats())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
