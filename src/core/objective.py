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
  occlusion : text covered by an opaque shape or image drawn above it

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


def _bg_behind(doc: Document, idx: int) -> Any:
    """Color behind text element idx: the topmost non-text element BELOW it (earlier in z-order)
    whose box contains the text center and whose color is known, else the canvas background.
    Known limitation: photos and gradients have no single color; their stored mean color is used."""
    e = doc.elements[idx]
    x0, y0, x1, y1 = _bbox(e)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    best = None
    for u in doc.elements[:idx]:
        if u.get("type") == "text":
            continue
        bx0, by0, bx1, by1 = _bbox(u)
        col = prop(u, "fill") if u.get("type") == "shape" else u.get("color")
        if bx0 <= cx <= bx1 and by0 <= cy <= by1 and col is not None:
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
            b["occlusion"] += _inter(box, _bbox(u)) / _area(box)

    for e in doc.elements:
        box = _bbox(e)
        b["bounds"] += (_area(box) - _inter(box, canvas)) / _area(box)

    for a in range(len(text_idx)):
        for c in range(a + 1, len(text_idx)):
            ba, bc = _bbox(doc.elements[text_idx[a]]), _bbox(doc.elements[text_idx[c]])
            b["overlap"] += _inter(ba, bc) / min(_area(ba), _area(bc))

    return sum(b.values()), b
