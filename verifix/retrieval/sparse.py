"""
retrieval/sparse.py - BM25 ranking over exemplar critiques (Approach A1 baseline).

BM25 is the honest floor every retrieval paper needs: zero-training, robust, and it exploits
the rare discriminative tokens in judge critiques ("low contrast", "cropped"). If a fancier
ranker cannot beat BM25 on downstream Delta-quality, it is not earning its complexity.

Falls back to a weighted-Jaccard token overlap if rank_bm25 is not installed, so the module
always imports. (Weighted Jaccard = |shared important tokens| / |all important tokens|; it is
the cruder cousin of BM25 and matches the earlier internal "jaccard" scorer.)
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

from .index import Exemplar

_DESIGN_TERMS = {
    "font", "color", "size", "spacing", "alignment", "hierarchy", "contrast", "overlap",
    "margin", "padding", "bold", "readable", "legible", "palette", "heading", "title",
    "layout", "position", "rotate", "opacity", "weight", "tilt", "cramped", "crooked",
}


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z]+", s.lower())


try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAVE_BM25 = True
except Exception:
    _HAVE_BM25 = False


def rank(query_critique: str, candidates: List[Exemplar], top_k: int = 3,
         key_mode: str = "critique") -> List[Tuple[float, Exemplar]]:
    """Return [(score, exemplar), ...] sorted desc, length <= top_k. key_mode selects
    which exemplar text is the retrieval key (critique / actions / both)."""
    from .index import exemplar_key_text
    if not candidates:
        return []
    qtok = _tok(query_critique)
    keys = [exemplar_key_text(e, key_mode) for e in candidates]
    if _HAVE_BM25 and qtok:
        corpus = [_tok(k) for k in keys]
        bm = BM25Okapi(corpus)
        scores = bm.get_scores(qtok)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(float(s), e) for s, e in ranked[:top_k]]
    # fallback: weighted Jaccard
    qs = set(qtok)

    def w(t: str) -> float:
        return 2.0 if t in _DESIGN_TERMS else 1.0

    scored = []
    for e, k in zip(candidates, keys):
        es = set(_tok(k))
        if not qs or not es:
            scored.append((0.0, e)); continue
        inter = sum(w(t) for t in qs & es)
        union = sum(w(t) for t in qs | es)
        scored.append((inter / union if union else 0.0, e))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def weighted_jaccard(query_critique: str, ex: Exemplar, key_mode: str = "both") -> float:
    """Weighted-Jaccard similarity between query and exemplar key (design terms weighted
    2x). A stable A6 feature independent of the BM25 corpus."""
    from .index import exemplar_key_text
    qs, es = set(_tok(query_critique)), set(_tok(exemplar_key_text(ex, key_mode)))
    if not qs or not es:
        return 0.0
    w = lambda t: 2.0 if t in _DESIGN_TERMS else 1.0
    inter = sum(w(t) for t in qs & es)
    union = sum(w(t) for t in qs | es)
    return inter / union if union else 0.0


def lex_precision(query_critique: str, ex: Exemplar, key_mode: str = "both") -> float:
    """Fraction of query tokens covered by the exemplar key (query-normalized).
    'does this address what the judge complained about?' Used as an A6 feature."""
    from .index import exemplar_key_text
    q = set(_tok(query_critique))
    if not q:
        return 0.0
    k = set(_tok(exemplar_key_text(ex, key_mode)))
    return len(q & k) / len(q)
