"""
perturb/requests.py - The user request for each case: the INVERSE of the perturbation, in words.

As in Perturb & Invert, the request asks for the change that undoes the damage, the way a user
would ask for it: which element, and what should happen to it. Two levels of detail:

  inverse        (default) direction and target, no exact values:
                 'Make the text "August 15-17" larger.'   'Move the image at the top left up.'
  inverse_exact  exact values, as an upper-bound / ablation setting:
                 'Make the text "August 15-17" 32 px.'    'Move the image to (120, 340).'

The generic defect-agnostic request (REPAIR_REQUEST) remains available as an ablation.
Elements are named the way a user sees them: text by its words, other elements by kind, color,
and position on the canvas. Nothing here is shown to the agent except the request text.
"""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional

from ..core.color import is_hex_color, rgb
from ..core.design import Document, prop

GENERIC = "Fix the design problems in this design. Keep its content and overall style."
MODES = ("inverse", "inverse_exact", "generic")


def color_name(c) -> str:
    if c is None or not (is_hex_color(c) or isinstance(c, (list, tuple))):
        return "its original color"
    r, g, b = (v / 255.0 for v in rgb(c))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.15:
        return "white" if v > 0.85 else ("light gray" if v > 0.6 else "gray")
    deg = h * 360
    names = [(15, "red"), (45, "orange"), (70, "yellow"), (170, "green"), (200, "teal"),
             (260, "blue"), (300, "purple"), (345, "pink"), (361, "red")]
    name = next(n for lim, n in names if deg < lim)
    if name in ("orange", "red") and v < 0.55:
        name = "brown" if name == "orange" else "dark red"
    elif v < 0.45:
        name = "dark " + name
    elif s < 0.4 and v > 0.8:
        name = "light " + name
    return name


def _where(e: Dict, doc: Document) -> str:
    x = (float(prop(e, "left") or 0) + float(prop(e, "width") or 0) / 2) / max(1, doc.width)
    y = (float(prop(e, "top") or 0) + float(prop(e, "height") or 0) / 2) / max(1, doc.height)
    v = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "middle")
    h = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "center")
    if v == "middle" and h == "center":
        return "in the middle"
    if v == "middle":
        return f"on the {h}"
    return f"at the {v}" + ("" if h == "center" else f" {h}")


def describe(e: Optional[Dict], doc: Document) -> str:
    """How a user would name this element."""
    if e is None:
        return "the element"
    t = e.get("type", "text")
    if t == "text":
        words = " ".join(str(prop(e, "text") or "").split())
        if len(words) > 32:
            words = words[:30].rstrip() + "…"
        return f'the text "{words}"'
    if t == "image" and e.get("tint"):
        return f"the {color_name(prop(e, 'fill'))} shape {_where(e, doc)}"
    if t == "image":
        return f"the image {_where(e, doc)}"
    kind = "circle" if prop(e, "shape") == "ellipse" else "rectangle"
    return f"the {color_name(prop(e, 'fill'))} {kind} {_where(e, doc)}"


def _bigger(a, b) -> bool:
    try:
        return float(a) > float(b)
    except Exception:
        return False


def one_request(entry: Dict, X: Document, Y: Document, exact: bool = False) -> Optional[str]:
    """Words for undoing one perturbation step (an audit entry with its recorded inverse)."""
    cls = entry.get("class")
    inv = entry.get("inverse") or []
    if not inv:
        return None
    c = inv[0]
    a, tgt, p = c.get("action"), c.get("target"), c.get("params") or {}
    ey = Y.get(tgt) if tgt != "canvas" else None
    name = describe(ey or X.get(tgt), Y)

    if a == "resize_text":
        if exact:
            return f"Set the size of {name} to {p['px']:g} px."
        larger = _bigger(p.get("px"), prop(ey, "font_size") if ey else 0)
        return f"Make {name} {'larger' if larger else 'smaller'}."
    if a == "recolor_text":
        if cls == "readability_contrast" and not exact:
            return f"Make {name} easier to read against its background."
        return (f"Change the color of {name} to {p['color']}." if exact
                else f"Change the color of {name} back to {color_name(p.get('color'))}.")
    if a == "change_font":
        return (f"Change the font of {name} to {p['family']}." if exact
                else f"Change the font of {name} to one that matches the rest of the design.")
    if a == "swap_palette":
        if exact:
            pairs = ", ".join(f"{k} to {v}" for k, v in (p.get("mapping") or {}).items())
            return f"Change these colors: {pairs}."
        return "The colors of the design have shifted; restore its original color palette."
    if a == "align_text":
        return f"{str(p.get('alignment', 'left')).capitalize()}-align {name}."
    if a == "reflow_text":
        if exact:
            return f"Set the box around {name} to {p['bbox']}."
        wider = _bigger(p["bbox"][2], prop(ey, "width") if ey else 0)
        return f"Make the box around {name} {'wider' if wider else 'narrower'} so the text wraps as before."
    if a == "recolor_image":
        return (f"Set the hue shift of {name} to {p['hue']:g} and its saturation to {p['sat']:g}." if exact
                else f"Restore the natural colors of {name}.")
    if a == "replace_image":
        return f"Put the original image back in place of {name}."
    if a == "crop_image":
        return (f"Set the crop of {name} to {p['bbox']}." if exact
                else f"Show more of {name}; it is cropped too tightly.")
    if a == "apply_filter":
        f = p.get("filter", "none")
        cur = prop(ey, "filter") if ey else "a"
        return (f"Remove the {cur} filter from {name}." if f == "none"
                else f"Apply a {f} filter to {name}.")
    if a == "apply_effect":
        f = p.get("effect", "none")
        cur = prop(ey, "effect") if ey else "the"
        return (f"Remove the {cur} effect from {name}." if f == "none"
                else f"Give {name} a {f} effect.")
    if a == "reposition":
        if exact:
            return f"Move {name} to x={p['x']:g}, y={p['y']:g}."
        dx = float(p["x"]) - float(prop(ey, "left") or 0) if ey else 0
        dy = float(p["y"]) - float(prop(ey, "top") or 0) if ey else 0
        parts = []
        if abs(dy) >= 1:
            parts.append("down" if dy > 0 else "up")
        if abs(dx) >= 1:
            parts.append("to the right" if dx > 0 else "to the left")
        return f"Move {name} {' and '.join(parts) or 'back'}."
    if a == "resize_element":
        if exact:
            return f"Resize {name} to {p['w']:g} x {p['h']:g} px."
        larger = _bigger(p.get("w"), prop(ey, "width") if ey else 0)
        return f"Make {name} {'larger' if larger else 'smaller'}."
    if a == "rotate":
        if exact:
            return f"Rotate {name} by {p['deg']:g} degrees."
        target = (float(prop(ey, "angle") or 0) + float(p["deg"])) if ey else 0
        if abs(((target + 180) % 360) - 180) < 0.5:
            return f"Straighten {name}; it is tilted."
        return f"Rotate {name} {'clockwise' if float(p['deg']) > 0 else 'counterclockwise'}."
    if a == "reorder_layer":
        cur = Y.index_of(tgt) if tgt else 0
        if exact:
            return f"Move {name} to layer {p['z']} (0 = bottom)."
        return (f"Move {name} further back so it does not cover other elements." if int(p["z"]) < (cur or 0)
                else f"Bring {name} forward so it is not hidden.")
    if a == "remove_element":
        if cls == "duplicate_elem":
            return f"Remove the extra copy of {name}."
        return f"Remove {name}; it does not belong in the design."
    if a == "change_bg":
        return (f"Change the background color to {p['color']}." if exact
                else f"Change the background color back to {color_name(p.get('color'))}.")
    if a == "adj_contrast":
        return (f"Change the overall contrast by {p['delta']:+g}." if exact
                else f"{'Increase' if float(p['delta']) > 0 else 'Reduce'} the overall contrast of the design.")
    if a == "adj_saturation":
        return (f"Change the overall saturation by {p['delta']:+g}." if exact
                else f"Make the colors {'more' if float(p['delta']) > 0 else 'less'} vivid overall.")
    return None


def build_request(audit: List[Dict], X: Document, Y: Document, mode: str = "inverse") -> str:
    if mode == "generic":
        return GENERIC
    parts = [r for r in (one_request(e, X, Y, exact=(mode == "inverse_exact")) for e in audit) if r]
    if not parts:
        return GENERIC
    if len(parts) == 1:
        return parts[0] + " Keep everything else as it is."
    return ("Please make these changes: " + " ".join(f"({i}) {t}" for i, t in enumerate(parts, 1))
            + " Keep everything else as it is.")
