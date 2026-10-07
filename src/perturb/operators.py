"""
perturb/operators.py - Invertible perturbations, aligned with Perturb & Invert (P&I).

Two families, as in P&I:

1. METADATA-GROUNDED CLASSES (P&I Sec. 4, Table 8), each with calibrated magnitude bins:
     style                : font swap to the farthest co-occurrence cluster (always significant)
     readability_size     : font size x r,   r in [0.85, 0.95] subtle | [0.60, 0.80] moderate
     readability_contrast : text color interpolated linearly toward its background until the WCAG
                            contrast ratio drops by dCR in [0.5, 1.5] subtle | [1.5, 3.0] moderate
     palette              : hue/saturation shift of the design palette with mean CIEDE2000
                            dE in [5, 15) subtle | [15, 30] moderate
                            (P&I writes "< 15" for subtle; we add a lower bound of 5 so the
                            shift is visible.)
2. TOOL PERTURBATIONS: a single invertible call to one tool of the P&I tool set (core/tools.py),
   with randomized parameters. P&I builds tool COMPOSITIONS and curates them for coverage of tool
   pairs; see perturb/coverage.py.

Every perturbation is applied as a real tool call, so it is executable and its inverse is a list
of tool calls. Targets (what the agent must restore) are read off structural_diff(Y, X), so they
always match exactly what changed; reorder_layer contributes a z target for the moved element.

The user request is DEFECT-AGNOSTIC by default (REPAIR_REQUEST): the critique, not the request,
must reveal what is wrong. Defect-specific requests (P&I-style) are available for ablations.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..core.color import (ciede2000, contrast_ratio, is_hex_color, lerp, shift_hue_sat, to_hex)
from ..core.design import Document, prop, structural_diff
from ..core.objective import _bg_behind
from ..core.tools import FILTERS, EFFECTS, ALIGNMENTS, PERTURBABLE, execute_tool
from .fonts import FontClusters

METADATA_CLASSES = ("style", "readability_size", "readability_contrast", "palette")
TOOL_CLASSES = tuple(t for t in PERTURBABLE if t not in ("swap_palette",))  # palette class covers it
ALL_CLASSES = METADATA_CLASSES + TOOL_CLASSES

BINS = {
    "readability_size": {"subtle": (0.85, 0.95), "moderate": (0.60, 0.80)},
    "readability_contrast": {"subtle": (0.5, 1.5), "moderate": (1.5, 3.0)},
    "palette": {"subtle": (5.0, 15.0), "moderate": (15.0, 30.0)},
}
SEVERITIES = ("subtle", "moderate")

REPAIR_REQUEST = "Fix the design problems in this design. Keep its content and overall style."

SPECIFIC_REQUEST = {
    "style": "The font no longer fits this design. Restore a fitting typeface.",
    "readability_size": "Some text has become too small. Make it readable again.",
    "readability_contrast": "Some text is hard to read against its background. Make it readable again.",
    "palette": "The colors look off. Restore the design's original palette.",
}

# Templated, judge-style descriptions used as index keys (the deployed judge re-keys these).
CRITIQUE = {
    "style": "the font does not match the style of the design",
    "readability_size": "some text is too small for its role and hard to read",
    "readability_contrast": "text has weak contrast against its background and is hard to read",
    "palette": "the colors clash and the palette looks inconsistent",
    "recolor_text": "a text color looks out of place",
    "change_font": "a typeface does not fit the design",
    "resize_text": "a text element has the wrong size for its role",
    "align_text": "text alignment is inconsistent",
    "reflow_text": "a text box is the wrong shape and the text wraps badly",
    "recolor_image": "an image has odd, unnatural colors",
    "replace_image": "an image does not fit the design",
    "crop_image": "an image is cropped badly",
    "apply_filter": "an image has an unwanted filter",
    "reposition": "an element is out of position",
    "resize_element": "an element is the wrong size",
    "rotate": "an element is tilted",
    "reorder_layer": "an element is on the wrong layer and hides another",
    "duplicate_elem": "an element is duplicated",
    "add_shape": "a stray shape clutters the design",
    "change_bg": "the background color does not fit",
    "apply_effect": "a text effect looks cheap or distracting",
    "adj_contrast": "the overall contrast is off",
    "adj_saturation": "the overall colors are washed out or oversaturated",
}


def query_for(classes: List[str], mode: str = "generic") -> str:
    if mode == "specific":
        parts = [SPECIFIC_REQUEST[c] for c in classes if c in SPECIFIC_REQUEST]
        if parts:
            return " ".join(parts)
    return REPAIR_REQUEST


# ------------------------------------------------------------------ helpers
def _texts(doc: Document) -> List[Dict]:
    return [e for e in doc.elements if e.get("type", "text") == "text"]


def _images(doc: Document) -> List[Dict]:
    return [e for e in doc.elements if e.get("type") == "image"]


def _call(action: str, target: str, **params) -> Dict:
    return {"action": action, "target": target, "params": params}


def _apply(doc: Document, call: Dict) -> Optional[Document]:
    out, _ = execute_tool(doc, call)
    return out


Result = Tuple[Optional[Document], List[Dict], Dict]   # (Y, inverse calls, audit)


# ------------------------------------------------------------------ metadata-grounded classes
def p_readability_size(doc, rnd, severity, **_) -> Result:
    texts = _texts(doc)
    if not texts:
        return None, [], {}
    e = rnd.choice(texts)
    old = float(prop(e, "font_size") or 24)
    r = rnd.uniform(*BINS["readability_size"][severity])
    new = round(old * r, 2)
    Y = _apply(doc, _call("resize_text", e["id"], px=new))
    return Y, [_call("resize_text", e["id"], px=old)], {"size_ratio": round(new / old, 4)}


def p_readability_contrast(doc, rnd, severity, **_) -> Result:
    lo, hi = BINS["readability_contrast"][severity]
    cands = []
    for i, e in enumerate(doc.elements):
        if e.get("type", "text") != "text" or not is_hex_color(prop(e, "color")):
            continue
        bg = _bg_behind(doc, i)
        cr0 = contrast_ratio(prop(e, "color"), bg)
        if cr0 - lo >= 1.05:
            cands.append((e, bg, cr0))
    if not cands:
        return None, [], {}
    e, bg, cr0 = rnd.choice(cands)
    want = min(rnd.uniform(lo, hi), cr0 - 1.05)
    target_cr = cr0 - want
    a, b = 0.0, 1.0
    for _ in range(40):                       # bisection on the interpolation weight
        m = (a + b) / 2
        if contrast_ratio(lerp(prop(e, "color"), bg, m), bg) > target_cr:
            a = m
        else:
            b = m
    new = lerp(prop(e, "color"), bg, a)
    dcr = cr0 - contrast_ratio(new, bg)
    if not (lo <= dcr + 1e-6 and dcr <= hi + 1e-6):
        return None, [], {}
    old = to_hex(prop(e, "color"))
    Y = _apply(doc, _call("recolor_text", e["id"], color=new))
    return Y, [_call("recolor_text", e["id"], color=old)], {"delta_cr": round(dcr, 3),
                                                          "cr_before": round(cr0, 3)}


def _palette_colors(doc: Document) -> List[str]:
    cols = set()
    if is_hex_color(doc.background):
        cols.add(to_hex(doc.background))
    for e in doc.elements:
        for k in ("color", "fill"):
            if is_hex_color(e.get(k)):
                cols.add(to_hex(e[k]))
    return sorted(cols)


def _chromatic(c: str) -> bool:
    return shift_hue_sat(c, 90.0) != to_hex(c)


def p_palette(doc, rnd, severity, **_) -> Result:
    lo, hi = BINS["palette"][severity]
    cols = [c for c in _palette_colors(doc) if _chromatic(c)]
    if not cols:
        return None, [], {}
    sign = rnd.choice([-1, 1])
    want = rnd.uniform(lo, hi)

    def mapping(a):
        return {c: shift_hue_sat(c, sign * 180.0 * a, 1.0 + 0.4 * a * rnd_sat) for c in cols}

    rnd_sat = rnd.choice([-1, 1])

    def mean_de(a):
        m = mapping(a)
        return sum(ciede2000(c, m[c]) for c in cols) / len(cols)

    if mean_de(1.0) < lo:
        return None, [], {}
    a, b = 0.0, 1.0
    for _ in range(40):
        mid = (a + b) / 2
        if mean_de(mid) < want:
            a = mid
        else:
            b = mid
    m = {c: n for c, n in mapping(b).items() if n != c}
    if len(set(m.values())) != len(m) or set(m.values()) & (set(_palette_colors(doc)) - set(m)):
        return None, [], {}                    # mapping must be invertible
    de = sum(ciede2000(c, m.get(c, c)) for c in cols) / len(cols)
    if not (lo <= de + 1e-6 and de <= hi + 1e-6):
        return None, [], {}
    Y = _apply(doc, _call("swap_palette", "canvas", mapping=m))
    return Y, [_call("swap_palette", "canvas", mapping={v: k for k, v in m.items()})], \
        {"delta_e": round(de, 3)}


def p_style(doc, rnd, severity="significant", clusters: Optional[FontClusters] = None, **_) -> Result:
    clusters = clusters or FontClusters.fallback()
    texts = [e for e in _texts(doc) if prop(e, "font")]
    if not texts:
        return None, [], {}
    e = rnd.choice(texts)
    old = prop(e, "font")
    new, dist = clusters.swap(old, rnd)
    if not new:
        return None, [], {}
    Y = _apply(doc, _call("change_font", e["id"], family=new))
    return Y, [_call("change_font", e["id"], family=old)], {"cluster_distance": round(dist, 4),
                                                             "font_from": old, "font_to": new}


# ------------------------------------------------------------------ single-tool perturbations
def p_tool(doc, rnd, tool: str, clusters: Optional[FontClusters] = None, **_) -> Result:
    W, H = doc.width, doc.height
    texts, images = _texts(doc), _images(doc)
    els = [e for e in doc.elements]
    pick = rnd.choice

    if tool == "recolor_text" and texts:
        e = pick(texts); old = to_hex(prop(e, "color") or "#000000")
        new = shift_hue_sat(old, rnd.choice([-1, 1]) * rnd.uniform(90, 180), 1.0)
        if new == old:
            new = to_hex((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
        return _apply(doc, _call(tool, e["id"], color=new)), [_call(tool, e["id"], color=old)], {}
    if tool == "change_font" and texts:
        e = pick(texts); old = prop(e, "font") or "Arial"
        pool = [f for f in (clusters or FontClusters.fallback()).fonts if f != old]
        new = pick(pool)
        return _apply(doc, _call(tool, e["id"], family=new)), [_call(tool, e["id"], family=old)], {}
    if tool == "resize_text" and texts:
        e = pick(texts); old = float(prop(e, "font_size") or 24)
        new = round(old * pick([rnd.uniform(0.5, 0.8), rnd.uniform(1.25, 1.6)]), 2)
        return _apply(doc, _call(tool, e["id"], px=new)), [_call(tool, e["id"], px=old)], {}
    if tool == "align_text" and texts:
        e = pick(texts); old = prop(e, "text_align") or "left"
        new = pick([a for a in ALIGNMENTS if a != old])
        return _apply(doc, _call(tool, e["id"], alignment=new)), [_call(tool, e["id"], alignment=old)], {}
    if tool == "reflow_text" and texts:
        e = pick(texts)
        l, t, w, h = (int(prop(e, k) or 0) for k in ("left", "top", "width", "height"))
        f = pick([rnd.uniform(0.55, 0.8), rnd.uniform(1.25, 1.5)])
        new = [l, t, max(1, int(w * f)), max(1, int(h / f))]
        return _apply(doc, _call(tool, e["id"], bbox=new)), [_call(tool, e["id"], bbox=[l, t, w, h])], {}
    if tool == "recolor_image" and (images or [e for e in els if e.get("type") == "shape"]):
        e = pick(images or [e for e in els if e.get("type") == "shape"])
        oh, os_ = float(prop(e, "hue_shift") or 0), float(prop(e, "sat_scale") or 1)
        nh = oh + pick([-1, 1]) * rnd.uniform(40, 150)
        ns = pick([rnd.uniform(0.3, 0.6), rnd.uniform(1.5, 2.0)])
        return (_apply(doc, _call(tool, e["id"], hue=nh, sat=round(ns, 4))),
                [_call(tool, e["id"], hue=oh, sat=os_)], {})
    if tool == "replace_image" and images and len(doc.assets) >= 2:
        e = pick(images); old = prop(e, "asset_id")
        others = [a for a in doc.assets if a != old]
        if others:
            new = pick(others)
            return _apply(doc, _call(tool, e["id"], asset=new)), [_call(tool, e["id"], asset=old)], {}
    if tool == "crop_image" and images:
        e = pick(images); old = list(prop(e, "crop") or [0, 0, 1, 1])
        new = [round(rnd.uniform(0.1, 0.25), 3), round(rnd.uniform(0.1, 0.25), 3),
               round(rnd.uniform(0.75, 0.9), 3), round(rnd.uniform(0.75, 0.9), 3)]
        return _apply(doc, _call(tool, e["id"], bbox=new)), [_call(tool, e["id"], bbox=old)], {}
    if tool == "apply_filter" and images:
        e = pick(images); old = prop(e, "filter") or "none"
        new = pick([f for f in FILTERS if f not in ("none", old)])
        return _apply(doc, _call(tool, e["id"], filter=new)), [_call(tool, e["id"], filter=old)], {}
    if tool == "apply_effect" and texts:
        e = pick(texts); old = prop(e, "effect") or "none"
        new = pick([f for f in EFFECTS if f not in ("none", old)])
        return _apply(doc, _call(tool, e["id"], effect=new)), [_call(tool, e["id"], effect=old)], {}
    if tool == "reposition" and els:
        e = pick(els); ox, oy = int(prop(e, "left") or 0), int(prop(e, "top") or 0)
        nx = ox + pick([-1, 1]) * int(rnd.uniform(0.05, 0.15) * W)
        ny = oy + pick([-1, 1]) * int(rnd.uniform(0.05, 0.15) * H)
        return _apply(doc, _call(tool, e["id"], x=nx, y=ny)), [_call(tool, e["id"], x=ox, y=oy)], {}
    if tool == "resize_element" and [e for e in els if e.get("type") != "text"]:
        e = pick([e for e in els if e.get("type") != "text"])
        ow, oh = int(prop(e, "width") or 100), int(prop(e, "height") or 100)
        f = pick([rnd.uniform(0.6, 0.8), rnd.uniform(1.2, 1.4)])
        return (_apply(doc, _call(tool, e["id"], w=max(1, int(ow * f)), h=max(1, int(oh * f)))),
                [_call(tool, e["id"], w=ow, h=oh)], {})
    if tool == "rotate" and els:
        e = pick(els); deg = round(pick([-1, 1]) * rnd.uniform(8, 30), 2)
        return _apply(doc, _call(tool, e["id"], deg=deg)), [_call(tool, e["id"], deg=-deg)], {}
    if tool == "reorder_layer" and len(els) >= 2:
        e = pick(els); oz = doc.index_of(e["id"])
        nz = len(els) - 1 if oz < len(els) - 1 else 0
        Y = _apply(doc, _call(tool, e["id"], z=nz))
        return Y, [_call(tool, e["id"], z=oz)], {"z_target": (e["id"], oz, nz)}
    if tool == "duplicate_elem" and els:
        e = pick(els)
        Y = _apply(doc, _call(tool, e["id"]))
        if Y is None:
            return None, [], {}
        new_id = Y.elements[-1]["id"]
        return Y, [_call("remove_element", new_id)], {}
    if tool == "add_shape":
        w, h = int(rnd.uniform(0.15, 0.35) * W), int(rnd.uniform(0.1, 0.25) * H)
        if texts and rnd.random() < 0.5:
            t = pick(texts)
            l, tp = int(prop(t, "left") or 0) + int(rnd.uniform(-0.1, 0.3) * w), int(prop(t, "top") or 0)
        else:
            l, tp = rnd.randint(0, max(1, W - w)), rnd.randint(0, max(1, H - h))
        fill = to_hex((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
        Y = _apply(doc, _call(tool, "canvas", shape=pick(["rect", "ellipse"]),
                              bbox=[l, tp, w, h], style={"fill": fill}))
        if Y is None:
            return None, [], {}
        return Y, [_call("remove_element", Y.elements[-1]["id"])], {}
    if tool == "change_bg" and is_hex_color(doc.background):
        old = to_hex(doc.background)
        new = shift_hue_sat(old, rnd.uniform(60, 180), 1.0)
        if ciede2000(new, old) < 8:
            new = lerp(old, pick(["#202020", "#f2f2f2"]), rnd.uniform(0.3, 0.6))
        return _apply(doc, _call(tool, "canvas", color=new)), [_call(tool, "canvas", color=old)], {}
    if tool in ("adj_contrast", "adj_saturation"):
        lo, hi = (0.25, 0.5) if tool == "adj_contrast" else (0.3, 0.6)
        delta = round(pick([-1, 1]) * rnd.uniform(lo, hi), 3)
        return _apply(doc, _call(tool, "canvas", delta=delta)), [_call(tool, "canvas", delta=-delta)], {}
    return None, [], {}


# ------------------------------------------------------------------ composition
def apply_spec(X: Document, spec: List[Tuple[str, str]], rnd: random.Random,
               clusters: Optional[FontClusters] = None):
    """Apply a list of (class, severity) perturbations in order. Returns
    (Y, targets, inverse_recipe, audit) or None if any step was infeasible."""
    Y = X.clone()
    inverse: List[Dict] = []
    audits, z_targets = [], []
    for cls, sev in spec:
        if cls == "readability_size":
            out = p_readability_size(Y, rnd, sev)
        elif cls == "readability_contrast":
            out = p_readability_contrast(Y, rnd, sev)
        elif cls == "palette":
            out = p_palette(Y, rnd, sev)
        elif cls == "style":
            out = p_style(Y, rnd, clusters=clusters)
        else:
            out = p_tool(Y, rnd, cls, clusters=clusters)
        Yn, inv, audit = out
        if Yn is None:
            return None
        Y = Yn
        inverse = inv + inverse                       # undo in reverse order
        if "z_target" in audit:
            z_targets.append(audit.pop("z_target"))
        audits.append({"class": cls, "severity": sev, **audit})
    targets = targets_from_diff(Y, X, z_targets)
    return Y, targets, inverse, audits


def targets_from_diff(Y: Document, X: Document, z_targets=()) -> List[Dict]:
    out = []
    for d in structural_diff(Y, X):
        if d["prop"] == "order":
            continue
        out.append({"target": d["target"], "prop": d["prop"], "x_value": d["to"],
                    "y_value": d["from"]})
    for eid, xz, yz in z_targets:
        if X.get(eid) is not None:
            out.append({"target": eid, "prop": "z", "x_value": X.index_of(eid), "y_value": yz})
    return out


# ------------------------------------------------------------------ oracle recipe
def restore_recipe(doc: Document, targets: List[Dict]) -> List[Dict]:
    """Tool calls that set every target back to its X value, computed RELATIVE TO doc (the
    current state). Used by the oracle condition (from the turn-1 design, where relative tools
    such as rotate need a different delta than at Y) and for mining the verified-fix index."""
    calls: List[Dict] = []
    pos: Dict[str, Dict] = {}
    for tg in targets:
        eid, p, x = tg["target"], tg["prop"], tg.get("x_value")
        e = doc.get(eid) if eid != "canvas" else None
        if p == "exists":
            if x is None and e is not None:
                calls.append(_call("remove_element", eid))
            continue
        if eid == "canvas":
            if p == "background" and is_hex_color(x):
                calls.append(_call("change_bg", "canvas", color=x))
            elif p in ("contrast", "saturation"):
                delta = round(float(x) - float(doc.adjust.get(p, 0.0)), 4)
                if delta:
                    calls.append(_call("adj_" + p, "canvas", delta=delta))
            continue
        if e is None:
            continue
        typ = e.get("type", "text")
        if p == "font":
            calls.append(_call("change_font", eid, family=x))
        elif p == "font_weight":
            calls.append(_call("change_font", eid, family=prop(e, "font") or "Arial", weight=x))
        elif p == "font_size":
            calls.append(_call("resize_text", eid, px=x))
        elif p == "color" and is_hex_color(x):
            calls.append(_call("recolor_text", eid, color=x))
        elif p == "text_align":
            calls.append(_call("align_text", eid, alignment=x))
        elif p == "effect":
            calls.append(_call("apply_effect", eid, effect=x))
        elif p == "filter":
            calls.append(_call("apply_filter", eid, filter=x))
        elif p == "crop":
            calls.append(_call("crop_image", eid, bbox=list(x)))
        elif p == "asset_id":
            calls.append(_call("replace_image", eid, asset=x))
        elif p in ("hue_shift", "sat_scale"):
            col = {t["prop"]: t.get("x_value") for t in targets
                   if t["target"] == eid and t["prop"] in ("hue_shift", "sat_scale")}
            call = _call("recolor_image", eid, hue=col.get("hue_shift", prop(e, "hue_shift")),
                         sat=col.get("sat_scale", prop(e, "sat_scale")))
            if call not in calls:
                calls.append(call)
        elif p == "angle":
            delta = (float(x) - float(prop(e, "angle") or 0) + 180) % 360 - 180
            if abs(delta) > 1e-6:
                calls.append(_call("rotate", eid, deg=round(delta, 4)))
        elif p == "z":
            calls.append(_call("reorder_layer", eid, z=int(x)))
        elif p == "fill" and is_hex_color(x) and is_hex_color(prop(e, "fill")):
            calls.append(_call("swap_palette", "canvas", mapping={to_hex(prop(e, "fill")): x}))
        elif p in ("left", "top", "width", "height"):
            pos.setdefault(eid, {})[p] = x
    for eid, g in pos.items():
        e = doc.get(eid)
        l = g.get("left", prop(e, "left")); t = g.get("top", prop(e, "top"))
        w = g.get("width", prop(e, "width")); h = g.get("height", prop(e, "height"))
        if e.get("type", "text") == "text":
            calls.append(_call("reflow_text", eid, bbox=[l, t, w, h]))
        else:
            if "left" in g or "top" in g:
                calls.append(_call("reposition", eid, x=l, y=t))
            if "width" in g or "height" in g:
                calls.append(_call("resize_element", eid, w=w, h=h))
    # remove elements last-but-one, reorder last (indices shift after removals)
    calls.sort(key=lambda c: (c["action"] == "remove_element", c["action"] == "reorder_layer"))
    removes = [c for c in calls if c["action"] == "remove_element"]
    reorders = [c for c in calls if c["action"] == "reorder_layer"]
    others = [c for c in calls if c not in removes and c not in reorders]
    return others + removes + reorders
