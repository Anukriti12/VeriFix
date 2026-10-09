"""
Measure how much each retrieved fix actually helps, on the RANKER-TRAINING split (needs the model
endpoint). This produces (1) training data for the learned ranker and (2) measured utilities for
the index. Never run it on the evaluation split: that would leak evaluation outcomes into the
ranker.

For each case (a certified datapoint from Crello validation designs disjoint from the index):
  1. turn 1: the planner edits Y once -> D1; the judge critiques D1
  2. the critique is compiled and verified (the M2 block)
  3. candidates: index entries of the detected defect classes, applicability-filtered,
     shortlisted to --pool by BM25
  4. for EACH candidate alone: the planner gets the M2 block plus that one fix and edits D1;
     the realized gain dQ = Q(D2) - Q(D1) is measured against the answer key
  5. log one row per candidate (features + dQ, grouped by case), and update the entry's utility
     with the running mean of its realized gains

Usage:
  python -m scripts.collect_ranker_logs --data data/rank_train.json --index data/index_rekeyed.json \
      --log data/ranker_logs.jsonl --out_index data/index_measured.json --pool 8
  python -m scripts.train_ranker --log data/ranker_logs.jsonl --out data/ranker.json
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.agents.compiler import Compiler  # noqa: E402
from verifix.agents.judge import Judge, critique_text  # noqa: E402
from verifix.agents.llm_client import make_client  # noqa: E402
from verifix.agents.planner import Planner  # noqa: E402
from verifix.agents.verifier import format_for_planner, verify  # noqa: E402
from verifix.core.metrics import quality  # noqa: E402
from verifix.core.tools import execute_actions  # noqa: E402
from verifix.loop.pipeline import Pipeline, fix_block  # noqa: E402
from verifix.perturb.generate import DataPoint  # noqa: E402
from verifix.retrieval import sparse  # noqa: E402
from verifix.retrieval.index import FixIndex  # noqa: E402
from verifix.retrieval.learned import features  # noqa: E402
from verifix.retrieval.outcome import UtilityUpdater  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rank_train.json")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--log", default="data/ranker_logs.jsonl")
    ap.add_argument("--out_index", default="data/index_measured.json")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--key_mode", default="both", choices=["critique", "actions", "both"])
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if "eval" in Path(args.data).name:
        raise SystemExit("refusing to collect ranker logs on an evaluation file (leakage)")
    cases = [DataPoint.from_dict(d) for d in json.load(open(args.data))[: args.max_cases]]
    index = FixIndex.load(args.index)
    client = make_client()
    pl, jd = Planner(client=client, model=args.model), Judge(client=client, model=args.model)
    cp = Compiler(client=client, model=args.model)
    pipe = Pipeline(pl, jd, cp, index, key_mode=args.key_mode)
    rows = 0
    with open(args.log, "a") as log:
        for ci, dp in enumerate(tqdm(cases, desc="cases")):
            t1 = pipe.run_turn1(dp.Y, dp.query)
            D1, jr = t1["doc"], t1["judge"]
            q1 = quality(D1, dp.targets)
            kept, _ = verify(D1, cp.compile(D1, jr))
            base = format_for_planner(kept, D1)
            cands = pipe.candidates(jr, D1)
            if not cands:
                continue
            crit = critique_text(jr)
            short = sparse.rank(crit, cands, top_k=args.pool, key_mode=args.key_mode)
            for _, ex in short:
                acts = pl.plan(D1, dp.query, extra_block=base + "\n\n" + fix_block([ex]))
                dq = quality(execute_actions(D1, acts), dp.targets) - q1
                log.write(json.dumps({"group_id": f"{Path(args.data).stem}:{ci}",
                                      "entry": ex.meta.get("uid"), "class": ex.defect_class,
                                      "features": features(crit, ex, key_mode=args.key_mode),
                                      "delta": round(dq, 4)}) + "\n")
                UtilityUpdater.update(ex, dq)
                rows += 1
            log.flush()
    index.save(args.out_index)
    print(f"logged {rows} candidate outcomes -> {args.log}; measured utilities -> {args.out_index}")


if __name__ == "__main__":
    main()
