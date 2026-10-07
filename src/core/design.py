"""
core/design.py - Domain-agnostic structured document + normalization + structural diff.

A Document is a canvas (width, height, background, global adjustments) plus a list of typed
elements. Each element is a plain dict with "id", "type" ("text" | "image" | "shape"), and property
keys. The list order is the z-order (index 0 is drawn first, i.e. at the bottom).

This is INTERNAL state. The agent never reads it: it sees rendered images (core/render.py). The
executor that edits it is the Perturb & Invert tool set in core/tools.py.

Image assets (PIL images, e.g. Crello per-element previews) live in doc.assets, keyed by asset id;
an image element points at one through "asset_id" (default: its own element id). to_dict() can
write assets to disk so a saved benchmark still renders its images when reloaded.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

# Properties compared by structural_diff and collateral, with the value an element has when the
# key is absent. Defaults matter: a perturbation can ADD a key (e.g. a filter) that X never had.
PROP_DEFAULTS: Dict[str, Any] = {
    "font": None, "font_size": None, "font_weight": "normal", "color": None,
    "text_align": "left", "line_height": None, "letter_spacing": None, "text": None,
    "opacity": 1.0, "angle": 0.0, "fill": None, "effect": "none",
    "left": 0, "top": 0, "width": None, "height": None,
    "hue_shift": 0.0, "sat_scale": 1.0, "crop": [0.0, 0.0, 1.0, 1.0], "filter": "none",
    "asset_id": None, "shape": None,
}
COMPARABLE_PROPS: List[str] = list(PROP_DEFAULTS.keys())
ADJUST_KEYS = ("contrast", "saturation")


def prop(e: Dict, name: str) -> Any:
    """Read an element property with its default (asset_id defaults to the element id)."""
    if name == "asset_id" and e.get("type") == "image":
        return e.get("asset_id", e.get("id"))
    return e.get(name, PROP_DEFAULTS.get(name))


class Document:
    def __init__(self, width: int = 1024, height: int = 1024, background: Any = "#FFFFFF",
                 elements: Optional[List[Dict]] = None, assets: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict] = None, adjust: Optional[Dict[str, float]] = None):
        self.width = int(width)
        self.height = int(height)
        self.background = background
        self.elements: List[Dict] = elements or []
        self.assets: Dict[str, Any] = assets or {}
        self.metadata: Dict = metadata or {}
        self.adjust: Dict[str, float] = {k: 0.0 for k in ADJUST_KEYS}
        self.adjust.update(adjust or {})

    def get(self, eid: str) -> Optional[Dict]:
        for e in self.elements:
            if e.get("id") == eid:
                return e
        return None

    def index_of(self, eid: str) -> Optional[int]:
        for i, e in enumerate(self.elements):
            if e.get("id") == eid:
                return i
        return None

    def clone(self) -> "Document":
        return Document(self.width, self.height, self.background, copy.deepcopy(self.elements),
                        dict(self.assets), copy.deepcopy(self.metadata), dict(self.adjust))

    # ---- persistence ----
    def uid(self) -> str:
        m = self.metadata
        return f"{m.get('source', 'doc')}-{m.get('split', 'x')}-{m.get('index', id(self))}"

    def to_dict(self, asset_dir: Optional[str] = None) -> Dict:
        d = {"width": self.width, "height": self.height, "background": self.background,
             "elements": self.elements, "metadata": self.metadata, "adjust": self.adjust}
        if asset_dir and self.assets:
            folder = os.path.join(asset_dir, self.uid())
            os.makedirs(folder, exist_ok=True)
            paths = {}
            for aid, img in self.assets.items():
                path = os.path.join(folder, f"{aid}.png")
                if not os.path.exists(path):
                    try:
                        img.save(path)
                    except Exception:
                        continue
                paths[aid] = path
            d["asset_paths"] = paths
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        assets = {}
        for aid, path in (d.get("asset_paths") or {}).items():
            try:
                from PIL import Image
                assets[aid] = Image.open(path).convert("RGBA")
            except Exception:
                continue
        return cls(d.get("width", 1024), d.get("height", 1024), d.get("background", "#FFFFFF"),
                   copy.deepcopy(d.get("elements", [])), assets, d.get("metadata", {}),
                   d.get("adjust", {}))


# ----------------------------------------------------------------------------- #
# Normalization + structural diff (ground-truth machinery; never used inside the loop)
# ----------------------------------------------------------------------------- #
def norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, int):
        return float(v)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (list, tuple)):
        return tuple(norm(x) for x in v)
    return v


def structural_diff(cur: Document, target: Document) -> List[Dict]:
    """Every (element, property) slot where cur differs from target, plus canvas-level slots
    (background, global adjustments, element order). len()==0 iff the designs are identical."""
    out: List[Dict] = []
    if norm(cur.background) != norm(target.background):
        out.append({"target": "canvas", "prop": "background",
                    "from": cur.background, "to": target.background})
    for k in ADJUST_KEYS:
        if norm(cur.adjust.get(k, 0.0)) != norm(target.adjust.get(k, 0.0)):
            out.append({"target": "canvas", "prop": k, "from": cur.adjust.get(k),
                        "to": target.adjust.get(k)})
    tgt = {e["id"]: e for e in target.elements}
    curm = {e["id"]: e for e in cur.elements}
    for eid, te in tgt.items():
        ce = curm.get(eid)
        if ce is None:
            out.append({"target": eid, "prop": "exists", "from": None, "to": "present"})
            continue
        for p in COMPARABLE_PROPS:
            if norm(prop(ce, p)) != norm(prop(te, p)):
                out.append({"target": eid, "prop": p, "from": prop(ce, p), "to": prop(te, p)})
    for eid in curm:
        if eid not in tgt:
            out.append({"target": eid, "prop": "exists", "from": "present", "to": None})
    common_t = [e["id"] for e in target.elements if e["id"] in curm]
    common_c = [e["id"] for e in cur.elements if e["id"] in tgt]
    if common_t != common_c:
        out.append({"target": "canvas", "prop": "order", "from": common_c, "to": common_t})
    return out


# Re-export the executor so older imports (from core.design import execute_actions) keep working.
from .tools import VALID_ACTIONS, execute_action, execute_actions  # noqa: E402
