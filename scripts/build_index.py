"""
Build the verified-fix index from the INDEX split (Crello validation, disjoint from evaluation),
and, with --build_clusters, the font clusters used by the style perturbation.

Usage:
  python -m scripts.build_index --split validation --n 300 --build_clusters \
      --clusters data/font_clusters.json --out data/index.json
  python -m scripts.build_index --split validation --n 300 --synthetic \
      --clusters data/font_clusters.json --out data/index_syn.json

Each entry stores its degraded design in <out stem>_Y.jsonl (assets under --asset_dir) so that
scripts/rekey_index.py can re-key the index with the deployed judge's own critiques.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.domains.cli import add_domain_args, load_designs  # noqa: E402
from verifix.perturb.fonts import FontClusters  # noqa: E402
from verifix.retrieval.build import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--clusters", default="data/font_clusters.json")
    ap.add_argument("--build_clusters", action="store_true",
                    help="build font clusters from these designs and save them to --clusters")
    ap.add_argument("--synthetic", action="store_true", help="add 2- and 3-class compositions (M4)")
    ap.add_argument("--asset_dir", default="data/assets")
    ap.add_argument("--out", default="data/index.json")
    add_domain_args(ap)
    args = ap.parse_args()

    designs, dom = load_designs(args, args.n, args.split)
    if args.build_clusters:
        clusters = FontClusters.from_designs(designs)
        Path(args.clusters).parent.mkdir(parents=True, exist_ok=True)
        clusters.save(args.clusters)
        print(f"[fonts] {len(clusters.fonts)} fonts in {clusters.centroids.shape[0]} clusters -> {args.clusters}")
    elif os.path.exists(args.clusters):
        clusters = FontClusters.load(args.clusters)
    else:
        raise SystemExit(f"{args.clusters} not found: run once with --build_clusters")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    y_path = str(Path(args.out).with_name(Path(args.out).stem + "_Y.jsonl"))
    yf = open(y_path, "w")
    count = [0]

    def y_store(Y):
        yf.write(json.dumps(Y.to_dict(args.asset_dir)) + "\n")
        count[0] += 1
        return count[0] - 1                       # line number in the jsonl

    idx = build_index(designs, synthetic=args.synthetic, clusters=clusters, y_store=y_store)
    yf.close()
    idx.save(args.out)
    print("index entries per class:", idx.stats())
    print(f"wrote {args.out} ({sum(idx.stats().values())} entries) and {y_path}")
    json.dump(dom.last_stats, open(str(Path(args.out).with_suffix("")) + "_load_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
