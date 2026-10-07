"""
agents/llm_client.py - OpenAI-compatible vision client (Qwen3-VL via vLLM / DashScope / etc).

UNTESTED in the build sandbox (no endpoint reachable). Set:
  VERIFIX_BASE_URL   e.g. http://<endpoint>/v1   (note the /v1 suffix for vLLM)
  VERIFIX_MODEL      the served model id, EXACTLY as GET /v1/models reports it
  VERIFIX_API_KEY    "EMPTY" for self-hosted, or your token
  VERIFIX_EXTRA_HEADERS  optional JSON, e.g. {"Authorization":"Bearer ..."}

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


def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
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
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
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
    for _ in range(3):
        try:
            resp = client.chat.completions.create(**params)
            break
        except Exception as e:
            msg = str(e).lower()
            if "max_tokens" in params and ("max_completion_tokens" in msg or "unsupported" in msg):
                params["max_completion_tokens"] = params.pop("max_tokens"); continue
            if "temperature" in params and ("temperature" in msg and
                                            ("unsupported" in msg or "default" in msg)):
                params.pop("temperature", None); continue
            raise
    text = (resp.choices[0].message.content or "") if resp else ""
    if counter is not None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            counter.add(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        else:
            counter.add(len(system) + len(user_text), len(text))
    return text
