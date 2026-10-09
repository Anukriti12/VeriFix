"""
agents/judge.py - Vision judge. Sees ONLY the clean render and the user's request.
Never sees the document, the action list, or X. UNTESTED offline (needs an endpoint).

With a specific request (the inverse of the damage, the default), the judge scores FOUR
dimensions: whether the request was carried out ("request") and three design dimensions
(layout, typography, color). With the generic repair request, "request" is omitted. Each score
is 0, 0.5 or 1, with an explanation that must name the element and what is still wrong; together
these form the critique handed to the feedback conditions.
"""
from __future__ import annotations

from typing import Dict, List

from ..core.design import Document
from ..core.render import render
from .llm_client import chat, extract_json, make_client

JUDGE_DIMS = ("request", "layout", "typography", "color")
DESIGN_DIMS = ("layout", "typography", "color")
GENERIC_REQUEST = "Fix the design problems in this design. Keep its content and overall style."

_DESIGN_RUBRIC = """- layout: alignment, spacing, layering, no overlaps, cut-offs, or stray elements
- typography: fitting typefaces, readable sizes, clear hierarchy, tasteful text effects
- color: readable contrast, a coherent palette, natural-looking images"""

JUDGE_SYSTEM = """You are a professional graphic-design quality judge.
You are shown a rendered design. Score THREE dimensions, each 0 (fail), 0.5 (partial), or 1.0 (pass):
""" + _DESIGN_RUBRIC + """
For every score below 1.0, the explanation must say WHICH element is wrong and HOW.
Return ONLY JSON:
{"layout":{"score":0.0,"explanation":"..."},
 "typography":{"score":0.0,"explanation":"..."},
 "color":{"score":0.0,"explanation":"..."}}"""

JUDGE_SYSTEM_REQUEST = """You are a professional graphic-design quality judge.
A user asked for changes to a design. You are shown the design AFTER an editor tried to carry
them out, and the user's request. Score FOUR dimensions, each 0 (fail), 0.5 (partial), or 1.0 (pass):
- request: was EVERY requested change made, on the right element, and by the right amount?
  Give 1.0 only if all of them are clearly done; say which change is missing, too weak, or overdone.
""" + _DESIGN_RUBRIC + """
For every score below 1.0, the explanation must say WHICH element is wrong and HOW.
Return ONLY JSON:
{"request":{"score":0.0,"explanation":"..."},
 "layout":{"score":0.0,"explanation":"..."},
 "typography":{"score":0.0,"explanation":"..."},
 "color":{"score":0.0,"explanation":"..."}}"""


def dims_for(query: str) -> List[str]:
    q = (query or "").strip()
    return list(DESIGN_DIMS) if (not q or q == GENERIC_REQUEST) else list(JUDGE_DIMS)


def present_dims(r: Dict) -> List[str]:
    return [d for d in JUDGE_DIMS if isinstance(r.get(d), dict)]


class Judge:
    def __init__(self, client=None, model=None, counter=None, fonts_dir=None):
        self.client = client or make_client()
        self.model = model
        self.counter = counter
        self.fonts_dir = fonts_dir

    def score(self, doc: Document, query: str = "") -> Dict:
        img = render(doc, marked=False, fonts_dir=self.fonts_dir)
        dims = dims_for(query)
        if "request" in dims:
            system, user = JUDGE_SYSTEM_REQUEST, f'User request: "{query}"\nScore the shown design. Return only the JSON.'
        else:
            system, user = JUDGE_SYSTEM, "Score the shown design. Return only the JSON."
        r = None
        for attempt in range(2):                     # one retry on unparseable output
            text = chat(self.client, self.model, system, user, images=[img],
                        counter=self.counter, temperature=0.1 if attempt == 0 else 0.3,
                        max_tokens=900)
            r = _parse(text, dims)
            if r.get("parse_ok"):
                break
        return finalize(r)


def _score(v) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return min((0.0, 0.5, 1.0), key=lambda s: abs(s - x))   # snap to the 3-level scale


def finalize(r: Dict) -> Dict:
    """Add overall (mean of the dimension scores) and the feedback text given to the planner."""
    dims = present_dims(r) or list(DESIGN_DIMS)
    for d in dims:
        r.setdefault(d, {"score": 0.0, "explanation": ""})
        r[d]["score"] = _score(r[d].get("score", 0))
    scores = [float(r[d]["score"]) for d in dims]
    r["overall"] = sum(scores) / len(scores)
    r["feedback_text"] = "\n".join(
        f"{d} ({'PASS' if r[d]['score'] >= 1 else 'PARTIAL' if r[d]['score'] >= 0.5 else 'FAIL'}, "
        f"{r[d]['score']}): {r[d]['explanation']}" for d in dims)
    return r


def critique_text(r: Dict) -> str:
    return " ".join(str(r.get(d, {}).get("explanation", "")) for d in present_dims(r))


def _parse(text: str, dims=DESIGN_DIMS) -> Dict:
    """Judge JSON -> dict with every dimension; parse_ok=False when the reply had no usable JSON
    (count these: aggregate reports the rate; a high rate means the prompt or model is wrong)."""
    v = extract_json(text, "object")
    if isinstance(v, dict) and any(isinstance(v.get(d), dict) for d in dims):
        for d in dims:
            if not isinstance(v.get(d), dict):
                v[d] = {"score": 0, "explanation": ""}
            v[d].setdefault("explanation", "")
        v["parse_ok"] = True
        return v
    out = {d: {"score": 0, "explanation": "parse error"} for d in dims}
    out["parse_ok"] = False
    return out
