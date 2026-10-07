"""
retrieval/outcome.py - Outcome-weighted retrieval of verified fixes (THE contribution).

Similarity answers "does this stored critique look like the live one." It does NOT answer
"will this fix actually improve THIS design." Outcome-weighted retrieval ranks by a blend of
similarity and the exemplar's MEASURED utility, where utility is updated from logged outcomes
(Delta objective at inference, or Delta recovery when a ground-truth target exists in training).

Lineage: utility/uplift-aware agent memory (MemRL arXiv:2601.03192, UpliftMem arXiv:2609.36805,
ExpGraph arXiv:2605.30712) in NLP agents, and RALF (CVPR'24) for similarity retrieval in design.
The gap we fill: (defect -> VERIFIED fix) pairs retrieved by measured improvement, in the visual
design domain, with an ORACLE-UTILITY upper bound that is only computable because P&I gives us
ground truth. Report similarity-retrieval vs outcome-weighted vs oracle-utility as the headline
retrieval ablation.

score(ex) = (1 - lambda) * sim_norm(ex) + lambda * utility_norm(ex)
Both terms are min-max normalized within the candidate set so lambda is interpretable in [0,1].
lambda=0 recovers pure similarity (A1/A2); lambda=1 is pure outcome ranking.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .index import Exemplar
from .sparse import rank as sparse_rank


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def rank(query_critique: str, candidates: List[Exemplar], top_k: int = 3,
         lam: float = 0.5, sim_pool: int = 20, key_mode: str = "critique"
         ) -> List[Tuple[float, Exemplar]]:
    """Outcome-weighted ranking. First take a similarity shortlist (sim_pool), then blend
    similarity with stored utility."""
    if not candidates:
        return []
    shortlist = sparse_rank(query_critique, candidates,
                            top_k=min(sim_pool, len(candidates)), key_mode=key_mode)
    if not shortlist:
        shortlist = [(0.0, e) for e in candidates[:sim_pool]]
    sims = [s for s, _ in shortlist]
    utils = [e.utility for _, e in shortlist]
    sim_n = _minmax(sims)
    util_n = _minmax(utils)
    blended = [
        ((1 - lam) * sn + lam * un, e)
        for sn, un, (_, e) in zip(sim_n, util_n, shortlist)
    ]
    blended.sort(key=lambda x: -x[0])
    return blended[:top_k]


def oracle_rank(query_critique: str, candidates: List[Exemplar], top_k: int = 3
                ) -> List[Tuple[float, Exemplar]]:
    """Upper bound: rank purely by stored utility (assumes utility == true usefulness).
    Only meaningful when utilities were filled from ground-truth outcomes."""
    ranked = sorted(candidates, key=lambda e: -e.utility)
    return [(e.utility, e) for e in ranked[:top_k]]


class UtilityUpdater:
    """Online running-mean update of exemplar utility from logged outcomes.

    After an exemplar's fix is injected and the resulting design measured, call update() with
    the realized delta (improvement). Utility becomes the running mean of realized deltas.
    Persist the index afterwards to keep the learned utilities.
    """

    @staticmethod
    def update(ex: Exemplar, realized_delta: float) -> None:
        n = ex.n_uses
        ex.utility = (ex.utility * n + realized_delta) / (n + 1)
        ex.n_uses = n + 1
