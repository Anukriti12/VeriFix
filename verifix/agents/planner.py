"""
agents/planner.py - Vision planner. Sees the MARKED render (numbered element tags) and the request,
emits Perturb & Invert tool calls that refer to elements by number. UNTESTED offline.
Never sees the document or X. Element numbers in the output are mapped back to element ids.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from ..core.design import Document
from ..core.render import element_index_map, render
from ..core.tools import tool_menu
from .llm_client import chat, extract_json, make_client

PLANNER_SYSTEM = """You are a graphic-design editing agent.
You see a rendered design with small RED number tags at each element's top-left corner
(0 = bottom layer). Refer to elements by their number; use "canvas" for design-level tools.
All positions and sizes are in CANVAS PIXELS (the canvas size is given with each request);
x grows to the right, y grows downward. Text sizes ("px") are font heights in canvas pixels.
Change only what needs fixing. If nothing needs fixing, return [].

Tools (use ONLY these):
{tools}

Each call: {"action": <tool>, "target": <element number as string, or "canvas">, "params": {...}}
Return ONLY a JSON array of calls. No prose."""


class Planner:
    def __init__(self, client=None, model=None, counter=None, fonts_dir=None):
        self.client = client or make_client()
        self.model = model
        self.counter = counter
        self.fonts_dir = fonts_dir

    def plan(self, doc: Document, query: str, feedback: Optional[str] = None,
             extra_block: Optional[str] = None) -> List[Dict]:
        img = render(doc, marked=True, fonts_dir=self.fonts_dir)
        user = (f'User request: "{query}"\nCanvas: {doc.width} x {doc.height} px, '
                f'{len(doc.elements)} elements tagged 0-{len(doc.elements) - 1}.\n'
                'Edit the design shown.')
        if feedback:
            user += f"\n\nFeedback on the current design:\n{feedback}\nFix the issues; keep what works."
        if extra_block:
            user += f"\n\n{extra_block}"
        user += "\n\nReturn ONLY the JSON array of tool calls."
        text = chat(self.client, self.model, PLANNER_SYSTEM.replace("{tools}", tool_menu()), user,
                    images=[img], counter=self.counter, temperature=0.2, max_tokens=1200)
        return map_marks(parse_calls(text), doc)


def parse_calls(text: str) -> List[Dict]:
    """Tool calls from a model reply; [] when no JSON array can be found."""
    v = extract_json(text, "array")
    return [c for c in v if isinstance(c, dict) and c.get("action")] if isinstance(v, list) else []


def map_marks(calls: List[Dict], doc: Document) -> List[Dict]:
    idx = element_index_map(doc)
    for c in calls:
        t = str(c.get("target", ""))
        if t in idx:
            c["target"] = idx[t]
    return calls
