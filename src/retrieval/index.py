"""
retrieval/index.py - The verified-fix index and the defect detector.

An Exemplar is a VERIFIED (defect -> fix) pair mined by perturb-and-invert:
  defect_class : the perturbation class (or compound key like "color+size")
  critique     : a judge-style defect statement used as the retrieval KEY (text)
  actions      : the concrete fix (typed edits with exact parameter values)
  utility      : measured usefulness of this fix (Delta objective or Delta recovery when a
                 ground-truth target is available in training logs). Starts at a prior and is
                 updated from logged outcomes (see retrieval/outcome.py). This is what lets us
                 retrieve by MEASURED OUTCOME rather than surface similarity (the contribution).
  source       : provenance ("oracle_single", "oracle_multi", "agent_success", ...).

FixIndex stores exemplars keyed by defect_class for O(1) recall, then a ranker (sparse /
dense / outcome) orders candidates. detect_defects maps a judge critique to defect classes
at inference time (no ground-truth log needed), so retrieval is deployable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Exemplar:
    defect_class: str
    critique: str
    actions: List[Dict]
    utility: float = 0.0
    n_uses: int = 0
    source: str = "oracle_single"
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# exemplarKey ablation (item 1): which text becomes the retrieval key.
# The internal finding was that critique+actions is the only key with positive
# headroom; critique-only is worst. Expose the choice so the public repo can
# reproduce that ablation. KEY_MODES: "critique" | "actions" | "both".
# --------------------------------------------------------------------------- #
KEY_MODES = ("critique", "actions", "both")


def actions_to_text(actions: List[Dict]) -> str:
    """Render an action list to a short text used as (part of) the retrieval key."""
    parts = []
    for a in actions or []:
        params = " ".join(f"{k} {v}" for k, v in (a.get("params") or {}).items())
        parts.append(f"{a.get('action','')} {params}".strip())   # element ids are design-specific
    return " ; ".join(parts)


def exemplar_key_text(ex: "Exemplar", key_mode: str = "critique") -> str:
    if key_mode == "actions":
        return actions_to_text(ex.actions)
    if key_mode == "both":
        return (ex.critique + " " + actions_to_text(ex.actions)).strip()
    return ex.critique  # default: "critique"


class FixIndex:
    def __init__(self):
        self.by_class: Dict[str, List[Exemplar]] = defaultdict(list)

    def add(self, ex: Exemplar) -> None:
        self.by_class[ex.defect_class].append(ex)

    def candidates(self, defect_classes: List[str]) -> List[Exemplar]:
        """Recall: every exemplar whose class matches a detected defect, plus compound keys
        that mention any detected class."""
        wanted = set(defect_classes)
        out: List[Exemplar] = []
        for key, exs in self.by_class.items():
            parts = set(key.split("+"))
            if parts & wanted:
                out.extend(exs)
        return out

    # ---- persistence ----
    def save(self, path: str) -> None:
        data = {k: [asdict(e) for e in v] for k, v in self.by_class.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FixIndex":
        idx = cls()
        with open(path) as f:
            data = json.load(f)
        for k, lst in data.items():
            for d in lst:
                idx.by_class[k].append(Exemplar(**d))
        return idx

    def stats(self) -> Dict:
        return {k: len(v) for k, v in self.by_class.items()}


# --------------------------------------------------------------------------- #
# Judge critique -> defect classes (deployable; no ground truth used)
# --------------------------------------------------------------------------- #
DIMS = ("layout", "typography", "color")   # must match agents/judge.JUDGE_DIMS

_KEYWORDS = {
    "style": ["typeface", "font does not", "font doesn't", "font choice", "mismatched font",
              "font style", "decorative font", "inconsistent font", "font clash", "wrong font"],
    "readability_size": ["too small", "tiny", "small text", "illegible", "hard to read", "size"],
    "readability_contrast": ["contrast", "faint", "washed", "blends", "against the background",
                             "hard to see", "low visibility"],
    "palette": ["palette", "colors clash", "colours clash", "color scheme", "hue", "inconsistent color"],
    "recolor_text": ["text color", "text colour"],
    "align_text": ["alignment", "aligned", "align"],
    "reflow_text": ["wraps", "wrapping", "text box", "line break"],
    "recolor_image": ["tint", "unnatural color", "discolor", "image color"],
    "replace_image": ["image does not fit", "wrong image", "irrelevant image"],
    "crop_image": ["cropped", "crop", "cut off"],
    "apply_filter": ["filter", "grayscale", "sepia", "blurry", "blurred", "inverted"],
    "reposition": ["position", "misplaced", "off-center", "misaligned"],
    "resize_element": ["too big", "scaled", "proportion", "oversized"],
    "rotate": ["tilt", "rotated", "angle", "crooked", "slanted", "skew"],
    "reorder_layer": ["hidden", "behind", "covered", "obscured", "layer"],
    "duplicate_elem": ["duplicate", "repeated", "twice", "copy"],
    "add_shape": ["stray shape", "shape", "block", "clutter", "unnecessary element"],
    "change_bg": ["background color", "background colour", "background"],
    "apply_effect": ["shadow", "outline", "glow", "effect"],
    "adj_contrast": ["overall contrast", "flat", "harsh"],
    "adj_saturation": ["saturated", "desaturated", "dull", "muted"],
}


def detect_defects(judge_result: Dict) -> List[str]:
    """Map the judge's failing dimensions and explanations to defect classes (recall-oriented)."""
    hits: List[str] = []
    for dim in DIMS:
        d = judge_result.get(dim, {})
        if float(d.get("score", 1)) >= 1:
            continue
        expl = str(d.get("explanation", "")).lower()
        for cls, kws in _KEYWORDS.items():
            if cls not in hits and any(k in expl for k in kws):
                hits.append(cls)
    return hits
