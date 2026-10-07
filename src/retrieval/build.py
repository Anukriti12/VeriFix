"""
retrieval/build.py - Mine the verified-fix index with perturb-and-invert, on the INDEX split
(disjoint from evaluation, so there is no leakage).

For each index design and each perturbation class (and severity) of perturb/operators.py:
  key      : a templated, judge-style description of the defect (re-key it with the deployed
             judge before use; see retrieval/coverage.rekey_with_judge)
  actions  : the restore recipe computed from Y, i.e. tool calls that are correct by construction
  utility  : a PRIOR. Phi drop when Phi can see the defect, else a small constant. Replace it with
             measured gains logged by run_experiments --log_outcomes (outcome-weighted retrieval).
synthetic=True adds compositions of 2-3 classes for rare multi-defect cases (condition M4).
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..core.design import Document
from ..core.objective import design_penalty
from ..core.tools import execute_actions
from ..perturb.fonts import FontClusters
from ..perturb.generate import sample_spec
from ..perturb.operators import ALL_CLASSES, BINS, CRITIQUE, apply_spec, restore_recipe
from .index import Exemplar, FixIndex


def _prior(Y: Document, fix) -> float:
    before, _ = design_penalty(Y)
    after, _ = design_penalty(execute_actions(Y, fix))
    return max(0.1, round(before - after, 4))


def build_index(designs: List[Document], synthetic: bool = False, seed: int = 1000,
                clusters: Optional[FontClusters] = None) -> FixIndex:
    idx = FixIndex()
    clusters = clusters or FontClusters.fallback()
    for i, X in enumerate(designs):
        for cls in ALL_CLASSES:
            sevs = list(BINS[cls]) if cls in BINS else ["significant" if cls == "style" else "-"]
            for sev in sevs:
                out = apply_spec(X, [(cls, sev)], random.Random(seed + 97 * i + hash(cls) % 997), clusters)
                if out is None:
                    continue
                Y, targets, _, audit = out
                if not targets:
                    continue
                fix = restore_recipe(Y, targets)
                idx.add(Exemplar(defect_class=cls, critique=CRITIQUE[cls], actions=fix,
                                 utility=_prior(Y, fix), source="pi_single",
                                 meta={"severity": sev, "audit": audit}))
    if synthetic:
        for i, X in enumerate(designs):
            for k in (2, 3):
                rnd = random.Random(seed + 31 * i + k)
                spec = sample_spec(rnd, k, "both")
                out = apply_spec(X, spec, rnd, clusters)
                if out is None:
                    continue
                Y, targets, _, audit = out
                classes = sorted({c for c, _ in spec})
                fix = restore_recipe(Y, targets)
                idx.add(Exemplar(defect_class="+".join(classes),
                                 critique="; ".join(CRITIQUE[c] for c in classes), actions=fix,
                                 utility=_prior(Y, fix), source=f"pi_comp{k}",
                                 meta={"audit": audit}))
    return idx
