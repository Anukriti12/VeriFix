"""
retrieval/learned.py - Learned outcome-weighted ranker (A6). Item 2.

Ranks candidates by a model trained on REAL logged outcomes, not hand weights. This is the
headline retrieval method and the "trained component" CVPR expects. Lineage: LambdaMART
(Burges 2010) via LightGBM; the internal version used the same 4 features and a ranking
objective. We reimplement here against verifix's OWN logged outcomes (never the internal
labels or code).

Feature vector per candidate (identical in training and serving, or the model breaks):
    [ weighted_jaccard , dense_cosine , lex_precision , utility(score_delta) ]
dense_cosine is 0.0 when no embedding backend is configured (the model then leans on the
other three; document this in results).

Serving degrades gracefully: if LightGBM or a trained model is unavailable, a linear fallback
with the A5 hand weights (0.6 sim, 0.0 dense, 0.3 lexPrecision, 0.1 utility) is used, so the
"learned" path always returns a ranking.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from .index import Exemplar
from .sparse import lex_precision, weighted_jaccard

try:
    import lightgbm as lgb  # type: ignore
    _HAVE_LGB = True
except Exception:
    _HAVE_LGB = False

FEATURES = ["weighted_jaccard", "dense_cosine", "lex_precision", "utility"]
_LINEAR_FALLBACK = [0.6, 0.0, 0.3, 0.1]  # A5 hand weights


def features(query_critique: str, ex: Exemplar, key_mode: str = "both",
             dense_cosine: float = 0.0) -> List[float]:
    return [
        weighted_jaccard(query_critique, ex, key_mode),
        float(dense_cosine),
        lex_precision(query_critique, ex, key_mode),
        float(ex.utility),
    ]


def delta_to_grade(delta: float) -> int:
    """Convert a realized quality gain into a LambdaMART relevance grade (internal defaults)."""
    if delta >= 0.4:
        return 3
    if delta >= 0.2:
        return 2
    if delta > 0:
        return 1
    return 0


def train_from_groups(X, y, group_sizes, num_boost_round: int = 300):
    """Train a LightGBM LambdaMART ranker. X: list of feature rows, y: relevance grades,
    group_sizes: #rows per query-group (sums to len(X)). Returns a Booster."""
    if not _HAVE_LGB:
        raise RuntimeError("lightgbm not installed; cannot train the learned ranker.")
    import numpy as np
    dtrain = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y, dtype=float),
                         group=np.asarray(group_sizes, dtype=int))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "num_leaves": 15,
        "learning_rate": 0.05, "min_child_samples": 20, "verbose": -1,
    }
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


class LearnedRanker:
    def __init__(self, booster=None, linear: Optional[List[float]] = None):
        self.booster = booster
        self.linear = linear or _LINEAR_FALLBACK

    @classmethod
    def load(cls, path: Optional[str]) -> "LearnedRanker":
        if path and os.path.exists(path) and _HAVE_LGB:
            try:
                with open(path) as f:
                    model_str = json.load(f)["booster"]
                return cls(booster=lgb.Booster(model_str=model_str))
            except Exception:
                pass
        return cls()  # linear fallback

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("no trained booster to save")
        with open(path, "w") as f:
            json.dump({"booster": self.booster.model_to_string()}, f)

    def _score(self, feat_rows: List[List[float]]) -> List[float]:
        if self.booster is not None:
            return list(self.booster.predict(feat_rows))
        return [sum(w * x for w, x in zip(self.linear, row)) for row in feat_rows]

    def rank(self, query_critique: str, candidates: List[Exemplar], top_k: int = 3,
             key_mode: str = "both", dense_fn=None) -> List[Tuple[float, Exemplar]]:
        if not candidates:
            return []
        rows = []
        for e in candidates:
            dc = dense_fn(query_critique, e) if dense_fn else 0.0
            rows.append(features(query_critique, e, key_mode, dc))
        scores = self._score(rows)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(float(s), e) for s, e in ranked[:top_k]]
