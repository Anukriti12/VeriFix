"""
agents/stub.py - A FAKE model endpoint for the offline smoke test. Never use it for results.

Activated by VERIFIX_STUB=1 (make_client returns StubClient and prints a warning). It answers
in the same formats as the real prompts so every code path runs without a GPU or network:
  judge     -> fixed partial scores with a templated critique
  compiler  -> a few tool calls on the first text element (one deliberately harmful)
  planner   -> applies whatever CHECKED / SUGGESTED / EXACT edits it was shown, else []
  other     -> a short sentence
"""
from __future__ import annotations

import ast
import json
import re
from types import SimpleNamespace


def _reply(system: str, user: str) -> str:
    sl = system.lower()
    if "what color fills" in user.lower():
        return "Red"
    if "quality judge" in sl:
        out = {
            "layout": {"score": 0.5, "explanation": "element 2 is misplaced and slightly rotated"},
            "typography": {"score": 0.5, "explanation": "the subtitle text is too small and hard to read"},
            "color": {"score": 1.0, "explanation": ""}}
        if "user request:" in user.lower():
            out["request"] = {"score": 0.5, "explanation": "the subtitle is still too small"}
        return json.dumps(out)
    if "turn a design judge's critique" in sl:
        m = re.search(r"Canvas: (\d+) x (\d+) px", user)
        h = int(m.group(2)) if m else 1000
        return json.dumps([
            {"action": "resize_text", "target": "3", "params": {"px": round(h * 0.05, 1)}, "reason": "too small"},
            {"action": "resize_text", "target": "3", "params": {"px": 6}, "reason": "harmful: below the floor"},
            {"action": "rotate", "target": "2", "params": {"deg": 0}, "reason": "no-op"}])
    if "editing agent" in sl:
        calls = []
        for line in user.splitlines():
            mm = re.match(r"- (\w+) on ([\w-]+): (.*)$", line.strip())
            if not mm:
                continue
            params = {}
            for k, v in re.findall(r"(\w+)=(\[.*?\]|\{.*?\}|[^,]+)", mm.group(3)):
                try:
                    params[k] = ast.literal_eval(v.strip())
                except Exception:
                    params[k] = v.strip()
            calls.append({"action": mm.group(1), "target": mm.group(2), "params": params})
        return json.dumps(calls)
    return "The subtitle is small; increase its size and keep the layout."


class _Completions:
    def create(self, model=None, messages=None, **_):
        system = messages[0]["content"] if messages else ""
        content = messages[-1]["content"] if messages else ""
        user = " ".join(c.get("text", "") for c in content if isinstance(c, dict)) \
            if isinstance(content, list) else str(content)
        text = _reply(system, user)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=len(user) // 4,
                                                     completion_tokens=len(text) // 4))


class StubClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())
