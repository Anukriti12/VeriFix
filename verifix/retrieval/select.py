"""
retrieval/select.py - Candidate filtering and diverse top-k selection.

applicable(ex, doc): an index entry is kept only if every call in its recipe could apply to the
current design: element tools need at least one element of a type the tool accepts (a font change
needs a text element; a crop needs a photo). Canvas tools always apply.

mmr(ranked, k): maximal marginal relevance (Carbonell & Goldstein 1998). Picks the top-k one at a
time, trading relevance (the ranker's score, min-max normalized) against similarity to what is
already picked, so the planner does not get three near-identical fixes.
    mmr(e) = lam * rel(e) - (1 - lam) * max_{s in picked} sim(e, s)
sim = token Jaccard of the entries' recipes (tool names + parameter names) and defect classes.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..core.design import Document
from ..core.tools import TOOLS
from .index import Exemplar


def _photo_or_type(e) -> str:
    if e.get("type") == "image" and not e.get("tint"):
        return "photo"
    return e.get("type", "text")


def applicable(ex: Exemplar, doc: Document) -> bool:
    types = {e.get("type", "text") for e in doc.elements}
    photos = any(_photo_or_type(e) == "photo" for e in doc.elements)
    for a in ex.actions or []:
        spec = TOOLS.get(a.get("action", ""))
        if spec is None:
            continue
        if spec.applies_to == ("canvas",):
            continue
        if a.get("action") in ("replace_image", "crop_image", "apply_filter"):
            if not photos:
                return False
            continue
        if not (set(spec.applies_to) & types):
            return False
    return True


def filter_applicable(cands: List[Exemplar], doc: Document) -> List[Exemplar]:
    return [c for c in cands if applicable(c, doc)]


def _sig(ex: Exemplar) -> set:
    toks = set(ex.defect_class.split("+"))
    for a in ex.actions or []:
        toks.add(a.get("action", ""))
        toks.update(f"{a.get('action','')}.{p}" for p in (a.get("params") or {}))
    return toks


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def mmr(ranked: List[Tuple[float, Exemplar]], k: int, lam: float = 0.7
        ) -> List[Tuple[float, Exemplar]]:
    if len(ranked) <= 1:
        return ranked[:k]
    scores = [s for s, _ in ranked]
    lo, hi = min(scores), max(scores)
    rel = [(s - lo) / (hi - lo) if hi - lo > 1e-12 else 1.0 for s in scores]
    sigs = [_sig(e) for _, e in ranked]
    picked: List[int] = []
    left = list(range(len(ranked)))
    while left and len(picked) < k:
        best = max(left, key=lambda i: lam * rel[i] - (1 - lam) *
                   max((_jac(sigs[i], sigs[j]) for j in picked), default=0.0))
        picked.append(best)
        left.remove(best)
    return [ranked[i] for i in picked]
