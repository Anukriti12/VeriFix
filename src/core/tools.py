"""
core/tools.py - The action space: the Perturb & Invert tool set (P&I, Table 9).

Every edit the planner or compiler can make, and every perturbation the benchmark applies, is a
call to one of these tools:  {"action": <tool>, "target": <element id | "canvas">, "params": {...}}

P&I Table 9 lists 22 tools in four categories (T=text, I=image, L=layout, D=design). We implement
21 of them with exact inverses at the design-state level. mood_transfer is NOT implemented: it
needs P&I's mood library (bundled font + palette styles), which is not public. See TOOLS_EXCLUDED.

Semantics follow P&I:
  * rotate, adj_contrast, adj_saturation are RELATIVE (inverse = the negated delta);
  * every other tool sets an absolute value (inverse = restore the recorded value);
  * duplicate_elem and add_shape create an element (inverse = remove_element on the new id);
  * remove_element deletes an element. Its P&I inverse ("restore element") needs the full
    element spec, which an image-only agent cannot know, so remove_element is an agent action
    but is never sampled as a perturbation (see PERTURBABLE).

execute_tool() is pure and deterministic. It returns (new_doc, "") on success or (None, reason)
when the call is infeasible: unknown tool, missing target, wrong element type, or invalid params.
This is the executor the verifier runs on a COPY of the design before accepting an edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .color import is_hex_color, to_hex
from .design import Document


@dataclass(frozen=True)
class ToolSpec:
    category: str          # "T" text, "I" image, "L" layout, "D" design (P&I Table 9)
    applies_to: Tuple[str, ...]   # element types, or ("canvas",)
    params: str            # parameter example shown to models
    inverse: str           # P&I's inverse, in words
    perturbable: bool      # may the benchmark sample it as a perturbation?


TOOLS: Dict[str, ToolSpec] = {
    # ---- text ----
    "recolor_text":   ToolSpec("T", ("text",), '{"color": "#1a1a1a"}', "restore color", True),
    "change_font":    ToolSpec("T", ("text",), '{"family": "Montserrat", "weight": "bold"}', "restore font", True),
    "resize_text":    ToolSpec("T", ("text",), '{"px": 32}', "restore size", True),
    "align_text":     ToolSpec("T", ("text",), '{"alignment": "center"}', "restore alignment", True),
    "reflow_text":    ToolSpec("T", ("text",), '{"bbox": [left, top, width, height]}', "restore bbox", True),
    # ---- image ----
    "recolor_image":  ToolSpec("I", ("image", "shape"), '{"hue": 0, "sat": 1.0}', "restore values", True),
    "replace_image":  ToolSpec("I", ("image",), '{"asset": "<asset id>"}', "restore asset", True),
    "crop_image":     ToolSpec("I", ("image",), '{"bbox": [x0, y0, x1, y1]}  (fractions 0..1)', "restore bbox", True),
    "apply_filter":   ToolSpec("I", ("image",), '{"filter": "none|grayscale|sepia|blur|invert"}', "remove filter", True),
    # ---- layout ----
    "reposition":     ToolSpec("L", ("text", "image", "shape"), '{"x": 120, "y": 80}', "restore position", True),
    "resize_element": ToolSpec("L", ("image", "shape"), '{"w": 300, "h": 200}', "restore size", True),
    "rotate":         ToolSpec("L", ("text", "image", "shape"), '{"deg": -15}  (relative)', "rotate -deg", True),
    "reorder_layer":  ToolSpec("L", ("text", "image", "shape"), '{"z": 0}  (0 = bottom)', "restore z", True),
    "duplicate_elem": ToolSpec("L", ("text", "image", "shape"), "{}", "remove copy", True),
    "remove_element": ToolSpec("L", ("text", "image", "shape"), "{}", "restore element", False),
    "add_shape":      ToolSpec("L", ("canvas",), '{"shape": "rect|ellipse", "bbox": [l, t, w, h], "style": {"fill": "#ff0000"}}', "remove shape", True),
    # ---- design ----
    "swap_palette":   ToolSpec("D", ("canvas",), '{"mapping": {"#old": "#new"}}', "restore palette", True),
    "change_bg":      ToolSpec("D", ("canvas",), '{"color": "#ffffff"}', "restore background", True),
    "apply_effect":   ToolSpec("D", ("text",), '{"effect": "none|shadow|outline|glow"}', "remove effect", True),
    "adj_contrast":   ToolSpec("D", ("canvas",), '{"delta": -0.2}  (relative)', "-delta", True),
    "adj_saturation": ToolSpec("D", ("canvas",), '{"delta": 0.3}  (relative)', "-delta", True),
}

TOOLS_EXCLUDED = {
    "mood_transfer": "needs P&I's mood library (bundled font+palette styles), not public",
}

VALID_ACTIONS: List[str] = list(TOOLS.keys())
PERTURBABLE: List[str] = [k for k, v in TOOLS.items() if v.perturbable]

FILTERS = ("none", "grayscale", "sepia", "blur", "invert")
EFFECTS = ("none", "shadow", "outline", "glow")
ALIGNMENTS = ("left", "center", "right")
ADJ_RANGE = (-0.9, 2.0)


def _num(p: Dict, k: str) -> Optional[float]:
    try:
        return float(p[k])
    except Exception:
        return None


def _wrap_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return round(a, 4)


def _new_id(doc: Document, stem: str) -> str:
    ids = {e.get("id") for e in doc.elements}
    i = 0
    while f"{stem}{i}" in ids:
        i += 1
    return f"{stem}{i}"


def execute_tool(doc: Document, call: Dict) -> Tuple[Optional[Document], str]:
    """Apply one tool call to a COPY of doc. Returns (new_doc, "") or (None, reason)."""
    name = call.get("action", "")
    spec = TOOLS.get(name)
    if spec is None:
        return None, "unknown_tool" if name not in TOOLS_EXCLUDED else "tool_not_implemented"
    p = call.get("params", {}) or {}
    d = doc.clone()
    tgt = call.get("target", "")

    # ---------------- canvas-level tools ----------------
    if spec.applies_to == ("canvas",):
        if name == "change_bg":
            if not is_hex_color(p.get("color")):
                return None, "bad_params"
            d.background = to_hex(p["color"])
            return d, ""
        if name in ("adj_contrast", "adj_saturation"):
            delta = _num(p, "delta")
            if delta is None:
                return None, "bad_params"
            key = "contrast" if name == "adj_contrast" else "saturation"
            new = round(d.adjust.get(key, 0.0) + delta, 4)
            if not (ADJ_RANGE[0] <= new <= ADJ_RANGE[1]):
                return None, "out_of_range"
            d.adjust[key] = new
            return d, ""
        if name == "swap_palette":
            mapping = p.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                return None, "bad_params"
            m = {to_hex(k): to_hex(v) for k, v in mapping.items()
                 if is_hex_color(k) and is_hex_color(v)}
            if not m:
                return None, "bad_params"
            changed = False
            if is_hex_color(d.background) and to_hex(d.background) in m:
                d.background = m[to_hex(d.background)]; changed = True
            for e in d.elements:
                for key in ("color", "fill"):
                    v = e.get(key)
                    if is_hex_color(v) and to_hex(v) in m:
                        e[key] = m[to_hex(v)]; changed = True
            return (d, "") if changed else (None, "no_effect")
        if name == "add_shape":
            shape = p.get("shape", "rect")
            bbox = p.get("bbox")
            style = p.get("style", {}) or {}
            if shape not in ("rect", "ellipse") or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                return None, "bad_params"
            l, t, w, h = (int(round(float(v))) for v in bbox)
            if w < 1 or h < 1:
                return None, "bad_params"
            fill = to_hex(style.get("fill", "#888888")) if is_hex_color(style.get("fill", "#888888")) else "#888888"
            eid = p.get("id") or _new_id(d, "shape-")
            d.elements.append({"id": eid, "type": "shape", "shape": shape, "left": l, "top": t,
                               "width": w, "height": h, "fill": fill, "opacity": 1.0, "angle": 0.0})
            return d, ""
        return None, "unknown_tool"

    # ---------------- element-level tools ----------------
    e = d.get(tgt)
    if e is None:
        return None, "missing_target"
    if e.get("type", "text") not in spec.applies_to:
        return None, "wrong_element_type"

    if name == "recolor_text":
        if not is_hex_color(p.get("color")):
            return None, "bad_params"
        e["color"] = to_hex(p["color"])
    elif name == "change_font":
        fam = p.get("family")
        if not isinstance(fam, str) or not fam.strip():
            return None, "bad_params"
        e["font"] = fam.strip()
        if p.get("weight") is not None:
            e["font_weight"] = str(p["weight"])
    elif name == "resize_text":
        px = _num(p, "px")
        if px is None or not (4 <= px <= 600):
            return None, "out_of_range"
        e["font_size"] = round(px, 2)
    elif name == "align_text":
        if p.get("alignment") not in ALIGNMENTS:
            return None, "bad_params"
        e["text_align"] = p["alignment"]
    elif name in ("reflow_text",):
        bbox = p.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None, "bad_params"
        l, t, w, h = (int(round(float(v))) for v in bbox)
        if w < 1 or h < 1:
            return None, "bad_params"
        e["left"], e["top"], e["width"], e["height"] = l, t, w, h
    elif name == "recolor_image":
        hue, sat = _num(p, "hue"), _num(p, "sat")
        if hue is None or sat is None or not (0.0 <= sat <= 3.0):
            return None, "bad_params"
        e["hue_shift"], e["sat_scale"] = _wrap_deg(hue), round(sat, 4)
    elif name == "replace_image":
        asset = p.get("asset")
        if asset not in d.assets:
            return None, "unknown_asset"
        e["asset_id"] = asset
    elif name == "crop_image":
        bb = p.get("bbox")
        try:
            x0, y0, x1, y1 = (float(v) for v in bb)
        except Exception:
            return None, "bad_params"
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            return None, "out_of_range"
        e["crop"] = [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
    elif name == "apply_filter":
        if p.get("filter") not in FILTERS:
            return None, "bad_params"
        e["filter"] = p["filter"]
    elif name == "apply_effect":
        if p.get("effect") not in EFFECTS:
            return None, "bad_params"
        e["effect"] = p["effect"]
    elif name == "reposition":
        x, y = _num(p, "x"), _num(p, "y")
        if x is None or y is None:
            return None, "bad_params"
        e["left"], e["top"] = int(round(x)), int(round(y))
    elif name == "resize_element":
        w, h = _num(p, "w"), _num(p, "h")
        if w is None or h is None or w < 1 or h < 1:
            return None, "bad_params"
        e["width"], e["height"] = int(round(w)), int(round(h))
    elif name == "rotate":
        deg = _num(p, "deg")
        if deg is None or not (-180 <= deg <= 180):
            return None, "out_of_range"
        e["angle"] = _wrap_deg(float(e.get("angle", 0) or 0) + deg)
    elif name == "reorder_layer":
        try:
            z = int(p["z"])
        except Exception:
            return None, "bad_params"
        if not (0 <= z < len(d.elements)):
            return None, "out_of_range"
        d.elements.remove(e)
        d.elements.insert(z, e)
    elif name == "duplicate_elem":
        import copy as _copy
        c = _copy.deepcopy(e)
        c["id"] = p.get("id") or _new_id(d, f"{e['id']}-dup")
        c["left"] = int(e.get("left", 0)) + 16
        c["top"] = int(e.get("top", 0)) + 16
        d.elements.append(c)
    elif name == "remove_element":
        d.elements = [x for x in d.elements if x.get("id") != tgt]
    else:
        return None, "unknown_tool"
    return d, ""


def execute_action(doc: Document, call: Dict) -> Document:
    """Lenient wrapper: an infeasible call leaves the design unchanged (returns a copy)."""
    out, _ = execute_tool(doc, call)
    return out if out is not None else doc.clone()


def execute_actions(doc: Document, calls: List[Dict]) -> Document:
    d = doc
    for c in calls or []:
        d = execute_action(d, c)
    return d


def is_feasible(doc: Document, call: Dict) -> Tuple[bool, str]:
    out, why = execute_tool(doc, call)
    return out is not None, why


def tool_menu() -> str:
    """Tool list with parameter examples, for planner and compiler prompts."""
    return "\n".join(f"  - {k} ({v.category}; on {'/'.join(v.applies_to)}): params {v.params}"
                     for k, v in TOOLS.items())
