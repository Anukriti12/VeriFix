"""
perturb/generate.py - Benchmark generation with certification, aligned with Perturb & Invert.

Each datapoint: clean design X, degraded input Y, targets (what must be restored), the exact
inverse recipe (tool calls), the request, and a certificate. A datapoint is KEPT iff:

  1. invertible : executing the recorded inverse recipe on Y gives back X exactly
                  (structural_diff == 0). Guards implementation bugs such as unit mismatches.
  2. violates   : the perturbation measurably violates the design's own intent (P&I Prop. 3):
                  the calibrated measure lies in its bin (size ratio, contrast drop, CIEDE2000,
                  cross-cluster font swap), and every tool step changed the design.
  3. visible    : the render of Y differs from the render of X (catches e.g. a font swap that
                  falls back to the same face because the font file is missing).

Also recorded, but NOT used to filter: phi_visible = Phi(Y) > Phi(X), i.e. whether the
reference-free objective can see the defect. Report its rate per class.

Generation mixes P&I's metadata-grounded classes and single-tool perturbations, k in [lo, hi]
per case, and can curate the pool for coverage of class pairs (perturb/coverage.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.design import Document, structural_diff
from ..core.objective import design_penalty
from ..core.render import render
from ..core.tools import execute_actions
from .coverage import greedy_cover
from .fonts import FontClusters
from .requests import build_request
from .operators import (BINS, METADATA_CLASSES, SEVERITIES, TOOL_CLASSES, apply_spec, query_for,
                        restore_recipe)


@dataclass
class DataPoint:
    X: Document
    Y: Document
    targets: List[Dict]
    inverse: List[Dict]
    pclasses: List[str]
    severities: List[str]
    query: str
    audit: List[Dict] = field(default_factory=list)
    certificate: Dict = field(default_factory=dict)

    def to_dict(self, asset_dir: Optional[str] = None) -> Dict:
        return {"X": self.X.to_dict(asset_dir), "Y": self.Y.to_dict(asset_dir),
                "targets": self.targets, "inverse": self.inverse, "pclasses": self.pclasses,
                "severities": self.severities, "query": self.query, "audit": self.audit,
                "certificate": self.certificate}

    @classmethod
    def from_dict(cls, d: Dict) -> "DataPoint":
        return cls(Document.from_dict(d["X"]), Document.from_dict(d["Y"]), d["targets"],
                   d.get("inverse", []), d["pclasses"], d.get("severities", []), d["query"],
                   d.get("audit", []), d.get("certificate", {}))


VISIBLE_EPS = 0.02   # mean absolute pixel difference, 0..255 scale, on a 256x256 downsample


def _renders_differ(a: Document, b: Document, fonts_dir: str, eps: float = VISIBLE_EPS) -> bool:
    try:
        ra = np.asarray(render(a, fonts_dir=fonts_dir).resize((256, 256)), dtype=np.float32)
        rb = np.asarray(render(b, fonts_dir=fonts_dir).resize((256, 256)), dtype=np.float32)
        return float(np.mean(np.abs(ra - rb))) > eps
    except Exception:
        return True


def _violates(audit: List[Dict]) -> bool:
    for a in audit:
        c, s = a["class"], a["severity"]
        if c == "readability_size":
            lo, hi = BINS[c][s]
            if not (lo - 1e-6 <= a["size_ratio"] <= hi + 1e-6):
                return False
        elif c == "readability_contrast":
            lo, hi = BINS[c][s]
            if not (lo - 1e-6 <= a["delta_cr"] <= hi + 1e-6):
                return False
        elif c == "palette":
            lo, hi = BINS[c][s]
            if not (lo - 1e-6 <= a["delta_e"] <= hi + 1e-6):
                return False
        elif c == "style":
            if a.get("cluster_distance", 0) <= 0:
                return False
    return True


def certify(X: Document, Y: Document, targets: List[Dict], inverse: List[Dict],
            audit: List[Dict], fonts_dir=None) -> Dict:
    invertible = len(structural_diff(execute_actions(Y, inverse), X)) == 0
    oracle_ok = len(structural_diff(execute_actions(Y, restore_recipe(Y, targets)), X)) == 0
    violates = bool(targets) and _violates(audit)
    visible = _renders_differ(X, Y, fonts_dir)
    phi_x, _ = design_penalty(X)
    phi_y, _ = design_penalty(Y)
    keep = invertible and violates and visible
    return {"invertible": invertible, "violates": violates, "visible": visible,
            "oracle_recipe_exact": oracle_ok, "phi_visible": phi_y > phi_x + 1e-9,
            "phi_x": round(phi_x, 4), "phi_y": round(phi_y, 4), "keep": keep}


def sample_spec(rnd: random.Random, k: int, mix: str = "both") -> List[Tuple[str, str]]:
    spec, used = [], set()
    while len(spec) < k:
        fam = mix if mix != "both" else rnd.choice(["metadata", "tools"])
        pool = METADATA_CLASSES if fam == "metadata" else TOOL_CLASSES
        cls = rnd.choice([c for c in pool if c not in used] or list(pool))
        if cls in used:
            break
        used.add(cls)
        sev = rnd.choice(SEVERITIES) if cls in BINS else ("significant" if cls == "style" else "-")
        spec.append((cls, sev))
    return spec


def make_datapoint(X: Document, spec: List[Tuple[str, str]], seed: int,
                   clusters: Optional[FontClusters] = None, query_mode: str = "inverse",
                   fonts_dir=None) -> Optional[DataPoint]:
    out = apply_spec(X, spec, random.Random(seed), clusters)
    if out is None:
        return None
    Y, targets, inverse, audit = out
    classes = [c for c, _ in spec]
    cert = certify(X, Y, targets, inverse, audit, fonts_dir)
    return DataPoint(X=X, Y=Y, targets=targets, inverse=inverse, pclasses=classes,
                     severities=[s for _, s in spec], query=build_request(audit, X, Y, query_mode),
                     audit=audit, certificate=cert)


def generate(designs: List[Document], k_range=(1, 3), seed: int = 42, mix: str = "both",
             sampler: str = "random", pool_per_design: int = 4,
             clusters: Optional[FontClusters] = None, query_mode: str = "inverse",
             keep_only_certified: bool = True, fonts_dir=None,
             max_tries: int = 8, stratify_k: bool = True) -> List[DataPoint]:
    """One datapoint per design. sampler="coverage" draws pool_per_design candidates per design
    and keeps the subset that greedily maximizes class-pair coverage (one per design).
    stratify_k=True gives design i exactly k = lo + (i mod (hi-lo+1)) defects, so the set has
    equal shares of k=1, 2, 3 (coverage selection alone would pick k=3 almost always)."""
    lo, hi = k_range
    clusters = clusters or FontClusters.fallback()
    cands: List[DataPoint] = []
    groups: List[str] = []
    dropped = 0
    n_per = pool_per_design if sampler == "coverage" else 1
    for i, X in enumerate(designs):
        made = 0
        for t in range(max_tries * n_per):
            if made >= n_per:
                break
            rnd = random.Random(seed * 100003 + i * 1009 + t)
            k = lo + (i % (hi - lo + 1)) if stratify_k else rnd.randint(lo, hi)
            spec = sample_spec(rnd, k, mix)
            dp = make_datapoint(X, spec, seed + i * 7919 + t, clusters, query_mode, fonts_dir)
            if dp is None:
                continue
            if keep_only_certified and not dp.certificate.get("keep"):
                dropped += 1
                continue
            cands.append(dp); groups.append(str(i)); made += 1
    if sampler == "coverage":
        idx = greedy_cover([d.pclasses for d in cands], budget=len(designs), groups=groups)
        out = [cands[j] for j in idx]
    else:
        out = cands
    print(f"[generate] kept {len(out)} datapoints from {len(designs)} designs; "
          f"{dropped} candidates failed certification; sampler={sampler}, mix={mix}")
    return out
