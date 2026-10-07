"""
domains/graphic_design.py - Crello-backed graphic design domain (the core domain).

load() pulls cyberagent/crello (revision 5.0.0) with `datasets` and converts each template into a
core.Document: text elements with font, size (px), color (hex), alignment, angle (degrees), line
height, letter spacing, string; image elements whose per-element preview becomes an asset; the
canvas background. Categorical fields are decoded with the dataset's label maps.

If `datasets` or the dataset is unavailable, load() falls back to synthetic designs and SAYS SO,
so the pipeline still runs offline. Check the log line before trusting a run: real numbers must
come from "loaded N Crello designs", never from the synthetic fallback.

real_defects(): hook for the real-defect transfer split (generator outputs, human-flagged issues).
"""
from __future__ import annotations

import math
import random
from typing import List

from ..core.color import to_hex
from ..core.design import Document
from .base import Domain


class GraphicDesignDomain(Domain):
    name = "graphic_design"

    def __init__(self, revision: str = "5.0.0"):
        self.revision = revision

    def load(self, n: int, split: str = "test") -> List[Document]:
        try:
            from datasets import load_dataset
            ds = load_dataset("cyberagent/crello", revision=self.revision, split=split)
            feats = ds.features
            out: List[Document] = []
            for i, s in enumerate(ds):
                if i >= n:
                    break
                out.append(_crello_to_doc(s, feats, i, split))
            print(f"[graphic_design] loaded {len(out)} Crello designs ({split}, rev {self.revision})")
            return out
        except Exception as e:
            print(f"[graphic_design] WARNING: Crello unavailable ({e}); using SYNTHETIC designs")
            return synthetic_designs(n, split=split)

    def real_defects(self, n: int, split: str = "test") -> List[Document]:
        # TODO(transfer split): load generator outputs / human-flagged defects here.
        return []


def _cat(feats, key, val):
    try:
        return feats[key].feature.int2str(val)
    except Exception:
        return val


def _crello_to_doc(s, feats, idx, split) -> Document:
    W = int(s.get("canvas_width", 1024) or 1024)
    H = int(s.get("canvas_height", 1024) or 1024)
    m = len(s.get("type", []))
    elements, assets = [], {}
    for j in range(m):
        etype = str(_cat(feats, "type", s["type"][j])).lower()
        etype = "text" if etype in ("text", "textelement") else "image"
        eid = f"el-{j}"
        col = list(s.get("color", [[0, 0, 0]] * m)[j] or [0, 0, 0])
        el = {
            "id": eid, "type": etype,
            "left": int(float(s["left"][j]) * W), "top": int(float(s["top"][j]) * H),
            "width": int(float(s["width"][j]) * W), "height": int(float(s["height"][j]) * H),
            "opacity": float(s["opacity"][j]) if "opacity" in s else 1.0,
            "angle": round(math.degrees(float(s["angle"][j])), 4) if "angle" in s else 0.0,
        }
        if etype == "text":
            el["text"] = s.get("text", [""] * m)[j] or ""
            el["color"] = to_hex(col)
            el["font"] = str(_cat(feats, "font", s["font"][j])) if "font" in s else "Arial"
            el["font_size"] = float(s.get("font_size", [24] * m)[j] or 24)
            el["text_align"] = str(_cat(feats, "text_align", s["text_align"][j])).lower() \
                if "text_align" in s else "left"
            if el["text_align"] not in ("left", "center", "right"):
                el["text_align"] = "left"
            el["line_height"] = float(s.get("line_height", [1.2] * m)[j] or 1.2)
            el["letter_spacing"] = float(s.get("letter_spacing", [0] * m)[j] or 0)
            el["font_weight"] = "normal"
        else:
            el["color"] = col                  # mean color, used as the background behind text
            img = s.get("image", [None] * m)[j]
            if img is not None:
                assets[eid] = img
                el["asset_id"] = eid
            else:
                el["fill"] = to_hex(col)
        elements.append(el)
    bg = "#FFFFFF"
    return Document(W, H, bg, elements, assets,
                    metadata={"source": "crello", "split": split, "index": idx})


def _asset(seed: int, w: int = 160, h: int = 120):
    """A small deterministic gradient image, so image tools work offline."""
    from PIL import Image
    rnd = random.Random(seed)
    c1 = [rnd.randrange(256) for _ in range(3)]
    c2 = [rnd.randrange(256) for _ in range(3)]
    img = Image.new("RGBA", (w, h))
    px = img.load()
    for x in range(w):
        t = x / (w - 1)
        col = tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3)) + (255,)
        for y in range(h):
            px[x, y] = col
    return img


def synthetic_designs(n: int, split: str = "synthetic") -> List[Document]:
    rnd = random.Random(42)
    fonts = ["Roboto", "Playfair Display", "Montserrat", "Oswald", "Lora", "Poppins"]
    palettes = [("#1A1A2E", "#E94560", "#F5F5F5"), ("#FFFFFF", "#0F3460", "#333333"),
                ("#F4EBD0", "#B85042", "#2C3E50"), ("#0B3D2E", "#F2C14E", "#FFFFFF")]
    out = []
    for i in range(n):
        bg, accent, ink = palettes[i % len(palettes)]
        W, H = 800, 1000
        assets = {f"asset-{k}": _asset(1000 * i + k) for k in range(3)}
        els = [
            {"id": "el-0", "type": "image", "asset_id": "asset-0", "left": 120, "top": 340,
             "width": 560, "height": 380, "opacity": 1.0, "angle": 0.0, "color": [128, 128, 128]},
            {"id": "el-1", "type": "text", "text": "SUMMER FEST", "font": rnd.choice(fonts),
             "font_size": 64, "font_weight": "bold", "color": accent, "text_align": "center",
             "line_height": 1.1, "letter_spacing": 1, "angle": 0.0, "opacity": 1.0,
             "left": 90, "top": 80, "width": 620, "height": 90},
            {"id": "el-2", "type": "text", "text": "August 15-17", "font": rnd.choice(fonts),
             "font_size": 32, "font_weight": "normal", "color": ink, "text_align": "center",
             "line_height": 1.2, "letter_spacing": 0, "angle": 0.0, "opacity": 1.0,
             "left": 220, "top": 210, "width": 360, "height": 48},
            {"id": "el-3", "type": "shape", "shape": "rect", "fill": accent, "left": 120,
             "top": 760, "width": 560, "height": 8, "opacity": 1.0, "angle": 0.0},
        ]
        out.append(Document(W, H, bg, els, assets,
                            metadata={"source": "synthetic", "split": split, "index": i}))
    print(f"[graphic_design] generated {n} synthetic designs")
    return out
