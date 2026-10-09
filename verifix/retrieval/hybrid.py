"""
retrieval/hybrid.py - Hybrid sparse + dense ranking fused by reciprocal rank fusion (RRF).

    RRF(e) = sum over rankers m in {bm25, dense} of 1 / (rho + rank_m(e)),   rho = 60

(Cormack et al. 2009). Like retrieval/dense.py, this RAISES when no embedding backend is
available instead of silently degrading to BM25, so a "hybrid" number is always hybrid.
"""
from __future__ import annotations

from typing import List, Tuple

from .index import Exemplar
from . import dense, sparse


def rank(query_critique: str, candidates: List[Exemplar], top_k: int = 3, rho: int = 60,
         client=None, key_mode: str = "both") -> List[Tuple[float, Exemplar]]:
    if not candidates:
        return []
    n = len(candidates)
    by_sparse = sparse.rank(query_critique, candidates, top_k=n, key_mode=key_mode)
    by_dense = dense.rank(query_critique, candidates, top_k=n, client=client, key_mode=key_mode)
    score = {id(e): 0.0 for e in candidates}
    for ranking in (by_sparse, by_dense):
        for r, (_, e) in enumerate(ranking, start=1):
            score[id(e)] += 1.0 / (rho + r)
    ranked = sorted(candidates, key=lambda e: -score[id(e)])
    return [(score[id(e)], e) for e in ranked[:top_k]]
