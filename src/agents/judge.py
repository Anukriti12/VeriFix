"""
agents/judge.py - Vision judge. Sees ONLY the clean render. Scores three DESIGN dimensions.
UNTESTED offline (needs an endpoint). Never sees the document, the action list, or X.

The study is about design repair, not instruction following: every case uses the same
defect-agnostic request, so request fidelity is not scored. The critique is the only channel
that tells the planner what is wrong, which is exactly the judge-to-planner translation we study.
"""
from __future__ import annotations

import json
import re
from typing import Dict

from ..core.design import Document
from ..core.render import render
from .llm_client import chat, make_client

JUDGE_DIMS = ("layout", "typography", "color")

JUDGE_SYSTEM = """You are a professional graphic-design quality judge.
You are shown a rendered design. Score THREE dimensions, each 0 (fail), 0.5 (partial), or 1.0 (pass):
- layout: alignment, spacing, layering, no overlaps, cut-offs, or stray elements
- typography: fitting typefaces, readable sizes, clear hierarchy, tasteful text effects
- color: readable contrast, a coherent palette, natural-looking images
For every score below 1.0, the explanation must say WHICH element is wrong and HOW.
Return ONLY JSON:
{"layout":{"score":0.0,"explanation":"..."},
 "typography":{"score":0.0,"explanation":"..."},
 "color":{"score":0.0,"explanation":"..."}}"""


class Judge:
    def __init__(self, client=None, model=None, counter=None, fonts_dir="fonts"):
        self.client = client or make_client()
        self.model = model
        self.counter = counter
        self.fonts_dir = fonts_dir

    def score(self, doc: Document, query: str = "") -> Dict:
        img = render(doc, marked=False, fonts_dir=self.fonts_dir)
        text = chat(self.client, self.model, JUDGE_SYSTEM,
                    "Score the shown design. Return only the JSON.", images=[img],
                    counter=self.counter, temperature=0.1, max_tokens=800)
        return finalize(_parse(text))


def finalize(r: Dict) -> Dict:
    """Add overall (mean of the dimension scores) and the feedback text given to the planner."""
    scores = [float(r[d]["score"]) for d in JUDGE_DIMS]
    r["overall"] = sum(scores) / len(scores)
    r["feedback_text"] = "\n".join(
        f"{d} ({'PASS' if r[d]['score'] >= 1 else 'PARTIAL' if r[d]['score'] >= 0.5 else 'FAIL'}, "
        f"{r[d]['score']}): {r[d]['explanation']}" for d in JUDGE_DIMS)
    return r


def critique_text(r: Dict) -> str:
    return " ".join(str(r.get(d, {}).get("explanation", "")) for d in JUDGE_DIMS)


def _parse(text: str) -> Dict:
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    try:
        r = json.loads(text)
        for d in JUDGE_DIMS:
            r.setdefault(d, {"score": 0, "explanation": ""})
        return r
    except Exception:
        return {d: {"score": 0, "explanation": "parse error"} for d in JUDGE_DIMS}
