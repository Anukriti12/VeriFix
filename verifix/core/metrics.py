"""
core/metrics.py - Judge-independent metrics (the spine of the evaluation). Uses the answer key,
so it runs ONLY after an episode, never inside the loop.

A perturbation records targets: {"target": element id | "canvas", "prop": name,
                                  "x_value": value in the clean design X, "y_value": perturbed value}

Primary: recovery fraction per target, averaged over targets ("quality" Q in the paper).
    frac = clip(1 - dist(final, x) / dist(y, x), 0, 1)
    i.e. the share of the perturbation the agent undid. Undo half a 10% shrink -> 0.5.
    Overshooting past x counts as error too. Categorical properties (font family, alignment,
    filter, effect, asset, existence) score 1 if restored exactly, else 0. Colors use CIEDE2000.
    Why not exact match: Perturb & Invert's calibrated perturbations are small (e.g. a 5-15%
    size reduction); an image-only agent cannot know the heading was exactly 40 px, so exact
    match scores almost everyone ~0 and hides real progress.
Secondary:
    exact_rate      : fraction of targets restored exactly (the strict metric)
    collateral      : number of slots changed that were NOT perturbed (a regression count)
    perceptual_distance : mean abs pixel difference between renders of final and X
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .color import ciede2000, is_hex_color
from .design import ADJUST_KEYS, Document, norm, prop, structural_diff

COLOR_PROPS = {"color", "fill", "background"}
NUMERIC_PROPS = {"font_size", "left", "top", "width", "height", "hue_shift", "sat_scale",
                 "opacity", "contrast", "saturation", "z", "angle"}
VECTOR_PROPS = {"crop"}


def _current(final: Document, tg: Dict) -> Any:
    eid, p = tg["target"], tg["prop"]
    if eid == "canvas":
        if p == "background":
            return final.background
        if p in ADJUST_KEYS:
            return final.adjust.get(p, 0.0)
        return None
    if p == "exists":
        return final.get(eid) is not None
    if p == "z":
        return final.index_of(eid)
    e = final.get(eid)
    return None if e is None else prop(e, p)


def _dist(a: Any, b: Any, p: str) -> Optional[float]:
    if a is None or b is None:
        return None
    if p in COLOR_PROPS and is_hex_color(a) and is_hex_color(b):
        return ciede2000(a, b)
    if p == "angle":
        d = abs(float(a) - float(b)) % 360
        return min(d, 360 - d)
    if p in NUMERIC_PROPS:
        return abs(float(a) - float(b))
    if p in VECTOR_PROPS:
        return float(sum(abs(float(u) - float(v)) for u, v in zip(a, b)))
    return 0.0 if norm(a) == norm(b) else 1.0


def target_fraction(final: Document, tg: Dict) -> float:
    p = tg["prop"]
    cur = _current(final, tg)
    if p == "exists":
        should = tg.get("x_value") is not None
        return 1.0 if bool(cur) == should else 0.0
    x, y = tg.get("x_value"), tg.get("y_value")
    dcx = _dist(cur, x, p)
    if dcx is None:
        return 0.0
    if y is None:
        return 1.0 if dcx == 0 else 0.0
    dyx = _dist(y, x, p)
    if not dyx:
        return 1.0 if dcx == 0 else 0.0
    return float(max(0.0, min(1.0, 1.0 - dcx / dyx)))


def target_distance(final: Document, tg: Dict) -> float:
    """UNCLIPPED distance to the original, in units of the perturbation (0 = restored, 1 = as
    perturbed, >1 = moved further away). Used by the ceiling verifier, where moving away from X
    must count as worse even when the clipped recovery fraction is already 0."""
    p = tg["prop"]
    cur = _current(final, tg)
    if p == "exists":
        return 0.0 if bool(cur) == (tg.get("x_value") is not None) else 1.0
    dcx = _dist(cur, tg.get("x_value"), p)
    if dcx is None:
        return 1.0
    dyx = _dist(tg.get("y_value"), tg.get("x_value"), p) if tg.get("y_value") is not None else None
    return dcx / dyx if dyx else (0.0 if dcx == 0 else 1.0)


def truth_distance(final: Document, targets: List[Dict]) -> float:
    return float(np.mean([target_distance(final, t) for t in targets])) if targets else 0.0


def target_exact(final: Document, tg: Dict) -> bool:
    p = tg["prop"]
    cur = _current(final, tg)
    if p == "exists":
        return bool(cur) == (tg.get("x_value") is not None)
    d = _dist(cur, tg.get("x_value"), p)
    return d is not None and d < 1e-6


def quality(final: Document, targets: List[Dict]) -> float:
    if not targets:
        return 1.0
    return float(np.mean([target_fraction(final, t) for t in targets]))


def recovery_error(final: Document, targets: List[Dict]) -> float:
    return 1.0 - quality(final, targets)


def exact_rate(final: Document, targets: List[Dict]) -> float:
    if not targets:
        return 1.0
    return float(np.mean([target_exact(final, t) for t in targets]))


def collateral(final: Document, X: Document, targets: List[Dict]) -> int:
    perturbed = {(t["target"], t["prop"]) for t in targets}
    if any(t["prop"] == "z" for t in targets):
        perturbed.add(("canvas", "order"))
    return sum(1 for d in structural_diff(final, X) if (d["target"], d["prop"]) not in perturbed)


def perceptual_distance(final: Document, X: Document, size: int = 256) -> float:
    try:
        from .render import render
        a = np.asarray(render(final).resize((size, size)), dtype=np.float32)
        b = np.asarray(render(X).resize((size, size)), dtype=np.float32)
        return float(np.mean(np.abs(a - b)) / 255.0)
    except Exception:
        return float("nan")


class CostCounter:
    """Counts model calls and tokens per case (exact when the API reports usage)."""

    def __init__(self):
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0

    def add(self, in_tok: int, out_tok: int):
        self.calls += 1
        self.in_tok += int(in_tok)
        self.out_tok += int(out_tok)

    def as_dict(self) -> Dict:
        return {"calls": self.calls, "in_tok": self.in_tok, "out_tok": self.out_tok}
