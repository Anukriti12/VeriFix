"""
retrieval/coverage.py - Coverage diagnostic + real-judge re-keying. Item 3.

The internal finding: P&I retrieval looked flat because the index was keyed on short defect
statements while the deployed judge writes descriptive critiques, so the eval critique rarely
reached an index key (defect-key coverage ~5% typo / ~32% layout vs ~90% real-judge
self-coverage). coverage() measures that gap on the public set; rekey_with_judge() is the fix
(key the index by the judge's own critique over the index designs), which the internal run
projected to lift coverage to ~90%.

coverage() is deterministic and offline-testable. rekey_with_judge() takes an injectable
judge_fn so it is testable with a stub; production passes a real judge wrapper.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from .index import Exemplar, FixIndex
from .sparse import weighted_jaccard


def coverage(index: FixIndex, eval_items: List[Tuple[List[str], str]],
             key_mode: str = "critique", threshold: float = 0.08) -> Dict[str, float]:
    """eval_items: list of (defect_classes, query_critique). An item is 'covered' if the index
    has at least one candidate whose key similarity to the query >= threshold. Returns overall
    coverage plus per-defect-class coverage."""
    hits = 0
    by_cls_hit = defaultdict(int)
    by_cls_tot = defaultdict(int)
    for defects, critique in eval_items:
        cands = index.candidates(defects)
        best = max((weighted_jaccard(critique, e, key_mode) for e in cands), default=0.0)
        covered = best >= threshold
        hits += int(covered)
        for c in defects:
            by_cls_tot[c] += 1
            by_cls_hit[c] += int(covered)
    out = {"overall": hits / max(1, len(eval_items))}
    for c in by_cls_tot:
        out[c] = by_cls_hit[c] / by_cls_tot[c]
    return out


def rekey_with_judge(index: FixIndex, judge_fn: Callable[[Exemplar], str]) -> FixIndex:
    """Return a new index whose exemplar critiques are replaced by judge-style critiques.
    judge_fn(exemplar) -> critique text. In production, judge_fn renders the exemplar's source
    design and asks the deployed judge for its verbatim critique (needs the endpoint)."""
    new = FixIndex()
    for key, exs in index.by_class.items():
        for e in exs:
            new.add(Exemplar(defect_class=e.defect_class, critique=judge_fn(e),
                             actions=e.actions, utility=e.utility, n_uses=e.n_uses,
                             source=e.source + "+rekeyed", meta=dict(e.meta)))
    return new
