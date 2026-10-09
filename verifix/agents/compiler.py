"""
agents/compiler.py - Critique compiler (M1). Rewrites the judge's prose critique into concrete
Perturb & Invert tool calls with exact parameters. UNTESTED offline (needs an endpoint).
M1 hands its output to the planner unchecked; M2 runs the verifier over it first.
"""
from __future__ import annotations

from typing import Dict, List

from ..core.design import Document
from ..core.render import render
from ..core.tools import tool_menu
from .judge import present_dims
from .llm_client import chat, make_client
from .planner import map_marks, parse_calls

COMPILER_SYSTEM = """You turn a design judge's critique into concrete, executable edits.
You see the rendered design (red number tags at each element's top-left corner, 0 = bottom
layer) and the judge's per-dimension scores and explanations. Positions and sizes are in canvas
pixels (canvas size given below); text sizes ("px") are font heights in canvas pixels.
Output a JSON array. Each item:
{"action": <tool>, "target": <element number or "canvas">, "params": {...},
 "reason": "<one line tying it to the critique>"}
Tools:
{tools}
Rules: address failing dimensions first; give exact values (pixel sizes, hex colors, degrees);
change only what the critique implicates. Return ONLY the JSON array."""


class Compiler:
    def __init__(self, client=None, model=None, counter=None, fonts_dir=None):
        self.client = client or make_client()
        self.model = model
        self.counter = counter
        self.fonts_dir = fonts_dir

    def compile(self, doc: Document, judge_result: Dict, query: str = "") -> List[Dict]:
        img = render(doc, marked=True, fonts_dir=self.fonts_dir)
        crit = "\n".join(f"  {d} (score={judge_result.get(d, {}).get('score', 0)}): "
                         f"{judge_result.get(d, {}).get('explanation', '')}" for d in present_dims(judge_result))
        req = f'User request: "{query}"\n' if query else ""
        user = (req + f"Canvas: {doc.width} x {doc.height} px, {len(doc.elements)} elements tagged "
                f"0-{len(doc.elements) - 1}.\nJudge critique:\n{crit}\n"
                "Return the JSON array of edits ([] if the critique asks for nothing).")
        text = chat(self.client, self.model, COMPILER_SYSTEM.replace("{tools}", tool_menu()), user,
                    images=[img], counter=self.counter, temperature=0.2, max_tokens=1000)
        return map_marks(parse_calls(text), doc)
