"""
core/objective.py - Phi, the reference-free design objective used by the verifier.

design_penalty(doc) -> (total, breakdown). Lower is better; 0 means no violation found.
Every term is computed from the CURRENT design alone: no original design X, no perturbation
record. That is what makes the verifier usable at deployment, where there is no answer key.

Phi is a checklist of ABSOLUTE design standards. Each term is a hinge: zero when the standard is
met, growing with the size of the violation.

  contrast  : text whose WCAG 2.1 contrast ratio against what is behind it is below 4.5:1
              (3:1 for large text, >= 24 px, or >= 18.7 px bold)
  size      : text smaller than a floor of 1.8% of canvas height
  tilt      : text rotated more than 3 degrees
  bounds    : the fraction of any element's box that falls outside the canvas
  overlap   : text boxes overlapping other text boxes
  occlusion : text covered by an opaque shape or image drawn above it (transparent parts of
              image assets do not count)

What Phi cannot see (report this; it is a result, not a bug): Perturb & Invert perturbations are
calibrated against the design's OWN intent, not against absolute standards. Shrinking a 40 px
heading to 36 px, lowering contrast from 15:1 to 13:1, swapping a font for one from another style
cluster, or shifting the palette hue breaks no absolute rule, so Phi stays the same. The verifier
is therefore a HARM FILTER: it rejects edits that create new violations; it cannot confirm that an
edit restored the original. Retrieval (M3/M4) supplies fixes for what Phi cannot see.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .color import contrast_ratio
from .design import Document, prop

MIN_CONTRAST = 4.5          # WCAG 2.1 AA, normal text
MIN_CONTRAST_LARGE = 3.0    # WCAG 2.1 AA, large text
LARGE_PX, LARGE_BOLD_PX = 24.0, 18.66
SIZE_FLOOR_FRAC = 0.018     # minimum font size as a fraction of canvas height
TILT_TOL_DEG = 3.0
OCCLUDER_MIN_OPACITY = 0.2

TERMS = ("contrast", "size", "tilt", "bounds", "overlap", "occlusion")


def _bbox(e: Dict) -> Tuple[int, int, int, int]:
    x, y = int(prop(e, "left") or 0), int(prop(e, "top") or 0)
    return x, y, x + int(prop(e, "width") or 0), y + int(prop(e, "height") or 0)


def _inter(a, b) -> int:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _area(b) -> int:
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


_REGION_CACHE: Dict[tuple, Tuple[Any, float]] = {}


def _region_stats(doc: Document, u: Dict, box) -> Tuple[Any, float]:
    """(alpha-weighted mean color, opaque fraction) of element u inside the canvas box `box`.
    Shapes are solid. Tinted vector shapes use their fill color. Images are sampled from their
    asset (crop-aware, rotation ignored), so a transparent part of a PNG does not count as
    being behind or on top of text."""
    ub = _bbox(u)
    ix0, iy0, ix1, iy1 = max(box[0], ub[0]), max(box[1], ub[1]), min(box[2], ub[2]), min(box[3], ub[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return None, 0.0
    if u.get("type") == "shape":
        return prop(u, "fill"), 1.0
    img = doc.assets.get(prop(u, "asset_id"))
    if img is None:
        return (prop(u, "fill") or u.get("avg_color") or u.get("color")), 1.0
    uw, uh = max(1, ub[2] - ub[0]), max(1, ub[3] - ub[1])
    c = prop(u, "crop") or [0, 0, 1, 1]
    rel = ((ix0 - ub[0]) / uw, (iy0 - ub[1]) / uh, (ix1 - ub[0]) / uw, (iy1 - ub[1]) / uh)
    key = (id(img), tuple(round(v, 3) for v in c), tuple(round(v, 3) for v in rel))
    if key not in _REGION_CACHE:
        import numpy as np
        sw, sh = img.size
        fx = lambda t: c[0] + t * (c[2] - c[0])
        fy = lambda t: c[1] + t * (c[3] - c[1])
        x0, y0 = int(fx(rel[0]) * sw), int(fy(rel[1]) * sh)
        x1, y1 = max(x0 + 1, int(fx(rel[2]) * sw)), max(y0 + 1, int(fy(rel[3]) * sh))
        a = np.asarray(img.convert("RGBA").crop((x0, y0, x1, y1)).resize((16, 16)), dtype=float)
        w = a[..., 3] / 255.0
        cov = float(w.mean())
        if w.sum() < 1e-6:
            col = None
        else:
            m = (a[..., :3] * w[..., None]).sum(axis=(0, 1)) / w.sum()
            col = "#{:02x}{:02x}{:02x}".format(*[int(round(v)) for v in m])
        if len(_REGION_CACHE) > 200000:
            _REGION_CACHE.clear()
        _REGION_CACHE[key] = (col, cov)
    col, cov = _REGION_CACHE[key]
    if u.get("tint") and prop(u, "fill"):
        col = prop(u, "fill")
    return col, cov


def _bg_behind(doc: Document, idx: int) -> Any:
    """Color behind text element idx: the topmost non-text element BELOW it (earlier in z-order)
    that is mostly opaque under the text box, else the canvas background. For photos and
    gradients the alpha-weighted mean color of the covered region is used."""
    e = doc.elements[idx]
    box = _bbox(e)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    best = None
    for u in doc.elements[:idx]:
        if u.get("type") == "text":
            continue
        ub = _bbox(u)
        if not (ub[0] <= cx <= ub[2] and ub[1] <= cy <= ub[3]):
            continue
        if float(prop(u, "opacity") if prop(u, "opacity") is not None else 1.0) < 0.5:
            continue
        col, cov = _region_stats(doc, u, box)
        if col is not None and cov >= 0.5:
            best = col
    return best if best is not None else doc.background


def design_penalty(doc: Document) -> Tuple[float, Dict[str, float]]:
    """Return (total_penalty, per-term breakdown). Lower is better."""
    floor = SIZE_FLOOR_FRAC * max(1, doc.height)
    canvas = (0, 0, doc.width, doc.height)
    b = {t: 0.0 for t in TERMS}
    text_idx = [i for i, e in enumerate(doc.elements) if e.get("type", "text") == "text"]

    for i in text_idx:
        e = doc.elements[i]
        size = float(prop(e, "font_size") or 24)
        bold = str(prop(e, "font_weight")).lower() in ("bold", "600", "700", "800", "900")
        large = size >= LARGE_PX or (bold and size >= LARGE_BOLD_PX)
        need = MIN_CONTRAST_LARGE if large else MIN_CONTRAST
        ratio = contrast_ratio(prop(e, "color") or "#000000", _bg_behind(doc, i))
        if ratio < need:
            b["contrast"] += (need - ratio) / need
        if size < floor:
            b["size"] += (floor - size) / floor
        ang = abs(float(prop(e, "angle") or 0))
        if ang > TILT_TOL_DEG:
            b["tilt"] += min(1.0, (ang - TILT_TOL_DEG) / 45.0)
        box = _bbox(e)
        for u in doc.elements[i + 1:]:
            if u.get("type") == "text":
                continue
            if float(prop(u, "opacity") if prop(u, "opacity") is not None else 1) < OCCLUDER_MIN_OPACITY:
                continue
            inter = _inter(box, _bbox(u))
            if inter:
                _, cov = _region_stats(doc, u, box)
                b["occlusion"] += cov * inter / _area(box)

    for e in doc.elements:
        box = _bbox(e)
        b["bounds"] += (_area(box) - _inter(box, canvas)) / _area(box)

    for a in range(len(text_idx)):
        for c in range(a + 1, len(text_idx)):
            ba, bc = _bbox(doc.elements[text_idx[a]]), _bbox(doc.elements[text_idx[c]])
            b["overlap"] += _inter(ba, bc) / min(_area(ba), _area(bc))

    return sum(b.values()), b
