"""
Train the A6 learned outcome-weighted ranker (LambdaMART via LightGBM) on logged outcomes.

Input: a jsonl where each row is {group_id, dim, features:[4], delta OR grade}. Produce these
by running M3/M4 with --log_outcomes (see scripts/run_experiments.py). delta is converted to a
relevance grade (>=.4 ->3, >=.2 ->2, >0 ->1, else 0); rows are grouped by group_id for ranking.

Usage:
  python -m scripts.run_experiments --conditions M3 --retriever outcome \
      --log_outcomes data/outcomes.jsonl        # collect training data
  python -m scripts.train_ranker --log data/outcomes.jsonl --out data/a6_model.json
  python -m scripts.run_experiments --conditions M3 --retriever learned \
      --learned_model data/a6_model.json         # use it
"""
import argparse
import json
from collections import defaultdict

from verifix.retrieval.learned import LearnedRanker, delta_to_grade, train_from_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="jsonl of {group_id, features, delta|grade}")
    ap.add_argument("--out", default="data/a6_model.json")
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    groups = defaultdict(list)
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            grade = r["grade"] if "grade" in r else delta_to_grade(float(r.get("delta", 0.0)))
            groups[r["group_id"]].append((r["features"], int(grade)))

    X, y, gsizes = [], [], []
    for gid, items in groups.items():
        # a ranking group needs >= 2 items with some grade variation; keep all, LightGBM copes
        for feat, grade in items:
            X.append(feat); y.append(grade)
        gsizes.append(len(items))

    print(f"groups={len(gsizes)} rows={len(X)} positive={sum(1 for g in y if g>0)}")
    booster = train_from_groups(X, y, gsizes, num_boost_round=args.rounds)
    LearnedRanker(booster=booster).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
