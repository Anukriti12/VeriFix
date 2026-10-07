"""
Run the condition ladder with the turn 1 -> turn 2 protocol. NEEDS the model endpoint.

Turn 1 is computed once per case and cached in <out>/turn1_cache.json, so every condition, even
across separate invocations, starts turn 2 from the identical design. Writes
<out>/results_<cond>.json for eval/aggregate.py.

Usage (set VERIFIX_BASE_URL / VERIFIX_MODEL / VERIFIX_API_KEY first):
  # go/no-go on a small subset first
  python -m scripts.run_experiments --data data/eval.json --conditions B0 B1 B2 M1 M2 ceiling --max_cases 30
  python -m scripts.aggregate_results --results results/
  # full ladder
  python -m scripts.run_experiments --data data/eval.json --index data/index.json \
      --index_syn data/index_syn.json --conditions B0 B1 B2 B3 B4 M1 M2 M3 M4 ceiling oracle
  # quality vs budget (multi-pass)
  python -m scripts.run_experiments --data data/eval.json --protocol multi_turn --max_passes 4 --out results_multi/
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from verifix.agents.compiler import Compiler
from verifix.agents.judge import Judge, finalize
from verifix.agents.llm_client import make_client
from verifix.agents.planner import Planner
from verifix.core.metrics import CostCounter
from verifix.core.tools import execute_actions
from verifix.loop.conditions import ALL
from verifix.loop.pipeline import COMPILE, Pipeline
from verifix.perturb.generate import DataPoint
from verifix.retrieval.index import FixIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.json")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--index_syn", default="data/index_syn.json")
    ap.add_argument("--conditions", nargs="+", default=ALL)
    ap.add_argument("--protocol", default="two_turn", choices=["two_turn", "multi_turn"])
    ap.add_argument("--max_passes", type=int, default=3, help="multi_turn only")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--retriever", default="learned", choices=["sparse", "dense", "outcome", "learned"])
    ap.add_argument("--key_mode", default="both", choices=["critique", "actions", "both"])
    ap.add_argument("--learned_model", default=None)
    ap.add_argument("--log_outcomes", default=None, help="jsonl for training the learned ranker")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fonts_dir", default="fonts")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(args.data))[: args.max_cases]
    cases = [DataPoint.from_dict(d) for d in raw]
    print(f"{len(cases)} eval cases, protocol={args.protocol}")

    index = FixIndex.load(args.index) if Path(args.index).exists() else None
    index_syn = FixIndex.load(args.index_syn) if Path(args.index_syn).exists() else None
    client = make_client()
    learned = None
    if args.retriever == "learned":
        from verifix.retrieval.learned import LearnedRanker
        learned = LearnedRanker.load(args.learned_model)

    # ---------------- shared turn 1 ----------------
    cache_path = out / "turn1_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    for ci, dp in enumerate(tqdm(cases, desc="turn1")):
        if str(ci) in cache:
            continue
        counter = CostCounter()
        pl = Planner(client=client, model=args.model, counter=counter, fonts_dir=args.fonts_dir)
        jd = Judge(client=client, model=args.model, counter=counter, fonts_dir=args.fonts_dir)
        t1 = Pipeline(pl, jd).run_turn1(dp.Y, dp.query)
        jr = {k: v for k, v in t1["judge"].items() if k not in ("overall", "feedback_text")}
        cache[str(ci)] = {"actions": t1["actions"], "judge": jr, "cost": counter.as_dict()}
        json.dump(cache, open(cache_path, "w"))

    log_f = open(args.log_outcomes, "a") if args.log_outcomes else None
    for cond in args.conditions:
        rows = []
        for ci, dp in enumerate(tqdm(cases, desc=cond)):
            c1 = cache[str(ci)]
            turn1 = {"doc": execute_actions(dp.Y, c1["actions"]), "actions": c1["actions"],
                     "judge": finalize(dict(c1["judge"]))}
            counter = CostCounter()
            pl = Planner(client=client, model=args.model, counter=counter, fonts_dir=args.fonts_dir)
            jd = Judge(client=client, model=args.model, counter=counter, fonts_dir=args.fonts_dir)
            cp = Compiler(client=client, model=args.model, counter=counter,
                          fonts_dir=args.fonts_dir) if cond in COMPILE else None
            pipe = Pipeline(pl, jd, cp, index_syn if cond == "M4" else index,
                            max_passes=args.max_passes, retriever=args.retriever, lam=args.lam,
                            embed_client=client, key_mode=args.key_mode, learned_ranker=learned)
            run = pipe.run if args.protocol == "two_turn" else pipe.run_multi
            r = run(dp.Y, dp.query, cond, targets=dp.targets, X=dp.X, turn1=turn1)
            row = {k: v for k, v in r.items() if k not in ("final_doc", "_injected")}
            row.update({"case_id": ci, "pclasses": dp.pclasses, "severities": dp.severities,
                        "phi_visible": dp.certificate.get("phi_visible"), "cost": counter.as_dict(),
                        "cost_turn1": c1["cost"]})
            rows.append(row)
            if log_f and r.get("_injected"):
                from verifix.retrieval.learned import features
                for crit, ex in r["_injected"]:
                    log_f.write(json.dumps({"group_id": f"{cond}:{ci}", "dim": ex.defect_class,
                                            "features": features(crit, ex, key_mode=args.key_mode),
                                            "delta": r.get("_realized_delta", 0.0)}) + "\n")
                log_f.flush()
        json.dump(rows, open(out / f"results_{cond}.json", "w"), indent=2)
        n = max(1, len(rows))
        print(f"[{cond}] Q2={sum(x.get('q_turn2', 0) for x in rows) / n:.3f} "
              f"dQ={sum(x.get('delta_q', 0) for x in rows) / n:+.3f} "
              f"calls={sum(x['cost']['calls'] for x in rows) / n:.1f}")


if __name__ == "__main__":
    main()
