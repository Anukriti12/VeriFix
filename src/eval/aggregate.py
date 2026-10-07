"""
eval/aggregate.py - Turn per-case rows (results/results_<cond>.json) into the paper's tables.
Every number is computed from the rows present; nothing is projected or filled in.

Rows come from the two-turn protocol (loop/pipeline.py): q_turn1, q_turn2, delta_q, judge, Phi,
collateral, verifier log, cost. Printed:
  - HEADLINE   : Q at turn 2, paired gain dQ = Q2 - Q1 with a 95% bootstrap CI, exact-match rate,
                 judge at turn 2, collateral, calls
  - GO/NO-GO   : M2 vs B2 on dQ (paired over the same cases)
  - OracleHeadroom : mean dQ of a method / mean dQ of the oracle, over cases the oracle improves
  - VERIFIER   : proposals, acceptance rate, rejection reasons (M2-M4, ceiling)
  - per class / severity / k / Phi-visible: dQ broken down
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ORDER = ["B0", "B1", "B2", "B3", "B4", "M1", "M2", "M3", "M4", "ceiling", "oracle"]


def load(results_dir: str) -> Dict[str, List[Dict]]:
    return {f.stem.replace("results_", ""): json.load(open(f))
            for f in sorted(Path(results_dir).glob("results_*.json"))}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: List[float], n: int = 2000, seed: int = 0):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    means = sorted(_mean([rnd.choice(xs) for _ in xs]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _by_case(rows):
    return {r.get("case_id", i): r for i, r in enumerate(rows)}


def report(results_dir: str) -> None:
    data = load(results_dir)
    if not data:
        print("no results found in", results_dir)
        return
    conds = [c for c in ORDER if c in data] + [c for c in data if c not in ORDER]

    print("\n=== HEADLINE (two-turn protocol; Q = recovery fraction) ===")
    print(f"{'cond':<9}{'n':>5}{'Q1':>7}{'Q2':>7}{'dQ':>8}{'95% CI':>18}{'exact2':>8}"
          f"{'judge2':>8}{'collat':>8}{'calls':>7}")
    for c in conds:
        rows = data[c]
        dq = [r.get("delta_q") for r in rows]
        lo, hi = bootstrap_ci(dq)
        tag = "  (uses X)" if c in ("ceiling", "oracle") else ""
        print(f"{c:<9}{len(rows):>5}{_mean([r.get('q_turn1') for r in rows]):>7.3f}"
              f"{_mean([r.get('q_turn2') for r in rows]):>7.3f}{_mean(dq):>+8.3f}"
              f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>18}"
              f"{_mean([r.get('exact_turn2') for r in rows]):>8.3f}"
              f"{_mean([r.get('judge_turn2') for r in rows]):>8.3f}"
              f"{_mean([r.get('collateral') for r in rows]):>8.2f}"
              f"{_mean([r.get('cost', {}).get('calls') for r in rows]):>7.1f}{tag}")

    if "B2" in data and "M2" in data:
        b2, m2 = _by_case(data["B2"]), _by_case(data["M2"])
        diffs = [m2[k]["delta_q"] - b2[k]["delta_q"] for k in m2 if k in b2]
        lo, hi = bootstrap_ci(diffs)
        verdict = "PASS" if lo > 0 else ("FAIL" if hi < 0 else "INCONCLUSIVE (CI spans 0)")
        print(f"\n=== GO/NO-GO: dQ(M2) - dQ(B2) = {_mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"  n={len(diffs)}  -> {verdict} ===")

    if "oracle" in data:
        orc = {k: r["delta_q"] for k, r in _by_case(data["oracle"]).items()
               if (r.get("delta_q") or 0) > 0}
        if orc:
            denom = _mean(list(orc.values()))
            print("\n=== OracleHeadroom (share of the oracle's turn-2 gain recovered) ===")
            for c in conds:
                if c == "oracle":
                    continue
                rows = _by_case(data[c])
                vals = [rows[k]["delta_q"] for k in orc if k in rows]
                if vals:
                    print(f"{c:<9} {(_mean(vals) / denom) * 100:6.1f}%   (n={len(vals)})")

    vrows = [(c, r["verifier"]) for c in conds for r in data[c] if r.get("verifier")]
    if vrows:
        print("\n=== VERIFIER (compiled edits) ===")
        agg = defaultdict(lambda: defaultdict(int))
        for c, v in vrows:
            for k, n in v.items():
                agg[c][k] += n
        for c, v in agg.items():
            prop_ = v.get("proposed", 0)
            rate = v.get("accepted", 0) / prop_ if prop_ else float("nan")
            reasons = ", ".join(f"{k}={n}" for k, n in v.items() if k not in ("proposed", "accepted"))
            print(f"{c:<9} proposed={prop_} accepted={rate:.2%}  rejected: {reasons or '-'}")

    def breakdown(title, keyfn):
        keys = sorted({k for c in conds for r in data[c] for k in keyfn(r)})
        if not keys:
            return
        print(f"\n=== dQ by {title} ===")
        print(f"{'cond':<9}" + "".join(f"{k[:12]:>13}" for k in keys))
        for c in conds:
            b = defaultdict(list)
            for r in data[c]:
                for k in keyfn(r):
                    b[k].append(r.get("delta_q"))
            print(f"{c:<9}" + "".join(f"{_mean(b.get(k, [])):>+13.3f}" for k in keys))

    breakdown("perturbation class", lambda r: r.get("pclasses", []))
    breakdown("severity", lambda r: [s for s in r.get("severities", []) if s not in ("-",)])
    breakdown("number of defects k", lambda r: [f"k={len(r.get('pclasses', []))}"])
    breakdown("Phi can see the defect", lambda r: [f"phi_visible={r.get('phi_visible')}"])
