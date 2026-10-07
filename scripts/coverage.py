"""Measure index coverage of an eval set's critiques (item 3 diagnostic).
Usage: python -m scripts.coverage --data data/eval.json --index data/index.json --key_mode critique
Without a model judge offline, each case's templated defect description stands in for the
critique. For the real diagnostic, use the deployed judge's turn-1 critiques (results/turn1_cache.json)."""
import argparse, json
from verifix.perturb.operators import CRITIQUE
from verifix.retrieval.index import FixIndex
from verifix.retrieval.coverage import coverage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.json")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--key_mode", default="critique", choices=["critique","actions","both"])
    ap.add_argument("--threshold", type=float, default=0.08)
    args = ap.parse_args()
    idx = FixIndex.load(args.index)
    data = json.load(open(args.data))
    items = [(d["pclasses"], d.get("judge_critique") or "; ".join(CRITIQUE.get(c, c) for c in d["pclasses"]))
             for d in data]
    cov = coverage(idx, items, key_mode=args.key_mode, threshold=args.threshold)
    print("coverage (key_mode=%s, thr=%.2f):" % (args.key_mode, args.threshold))
    for k, v in sorted(cov.items()):
        print(f"  {k:<12} {v*100:5.1f}%")

if __name__ == "__main__":
    main()
