"""
agents/llm_client.py - OpenAI-compatible vision client (Qwen3-VL via vLLM / DashScope / etc).

UNTESTED in the build sandbox (no endpoint reachable). Set:
  VERIFIX_BASE_URL   e.g. http://<endpoint>/v1   (note the /v1 suffix for vLLM)
  VERIFIX_MODEL      the served model id, EXACTLY as GET /v1/models reports it
  VERIFIX_API_KEY    "EMPTY" for self-hosted, or your token
  VERIFIX_EXTRA_HEADERS  optional JSON, e.g. {"Authorization":"Bearer ..."}
  VERIFIX_MAX_IMAGE_SIDE longest side (px) of images sent to the model (default 1280); renders
                         are downscaled before sending so large Crello canvases stay affordable

The model is a vision-language MODEL. It does NOT render; it reads rendered PNGs (core/render).
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Dict, List, Optional

from PIL import Image

try:
    from openai import OpenAI
except Exception:  # keep import-safe offline
    OpenAI = None  # type: ignore

_WARNED = False

# Optional record of every model call (set by scripts/run_experiments.py --log_io).
IO_LOG_PATH: Optional[str] = None
IO_CONTEXT: Dict = {}


def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    if os.environ.get("VERIFIX_STUB") == "1":
        from .stub import StubClient
        print("[llm_client] VERIFIX_STUB=1: using the FAKE offline model. Numbers are meaningless.")
        return StubClient()
    if OpenAI is None:
        raise RuntimeError("openai not installed: pip install openai")
    global _WARNED
    api_key = api_key or os.environ.get("VERIFIX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("VERIFIX_BASE_URL")
    if not api_key:
        api_key = "EMPTY"
        if not _WARNED:
            print("[llm_client] no API key; using 'EMPTY' (fine for self-hosted vLLM).")
            _WARNED = True
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    eh = os.environ.get("VERIFIX_EXTRA_HEADERS")
    if eh:
        try:
            kwargs["default_headers"] = json.loads(eh)
        except Exception:
            pass
    return OpenAI(**kwargs)


def _data_url(img: Image.Image) -> str:
    img = img.convert("RGB")
    side = int(os.environ.get("VERIFIX_MAX_IMAGE_SIDE", "1280"))
    if max(img.size) > side:
        r = side / max(img.size)
        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(client, model: str, system: str, user_text: str,
         images: Optional[List[Image.Image]] = None, counter=None,
         temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """One vision+text turn. Returns assistant text. Records exact tokens into `counter`
    when the API reports usage, else a chars/4 proxy."""
    content = [{"type": "text", "text": user_text}]
    for im in images or []:
        content.append({"type": "image_url", "image_url": {"url": _data_url(im)}})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    params = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    resp = None
    import time
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**params)
            break
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("timeout", "timed out", "connection", "rate limit", "429",
                                      "502", "503", "504", "overloaded")) and attempt < 5:
                time.sleep(min(60, 2 ** attempt * 2)); continue
            if "max_tokens" in params and ("max_completion_tokens" in msg or "unsupported" in msg):
                params["max_completion_tokens"] = params.pop("max_tokens"); continue
            if "temperature" in params and ("temperature" in msg and
                                            ("unsupported" in msg or "default" in msg)):
                params.pop("temperature", None); continue
            raise
    text = (resp.choices[0].message.content or "") if resp else ""
    text = strip_thinking(text)
    if IO_LOG_PATH:
        try:
            role = system.strip().splitlines()[0][:80] if system else ""
            with open(IO_LOG_PATH, "a") as f:
                f.write(json.dumps({**IO_CONTEXT, "role": role, "user": user_text,
                                    "reply": text, "n_images": len(images or [])}) + "\n")
        except Exception:
            pass
    if counter is not None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            counter.add(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        else:
            counter.add(len(system) + len(user_text), len(text))
    return text


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models prepend to their answer."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str, kind: str = "array"):
    """Find the first JSON array (kind="array") or object (kind="object") in a model reply,
    tolerating code fences and prose around it. Returns the parsed value or None."""
    import re
    text = re.sub(r"```(?:json)?", "", strip_thinking(text))
    open_c, close_c = ("[", "]") if kind == "array" else ("{", "}")
    start = text.find(open_c)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find(open_c, start + 1)
    return None
