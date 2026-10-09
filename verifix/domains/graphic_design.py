"""
domains/graphic_design.py - Crello-backed graphic design domain.

Loads cyberagent/crello (v5.x schema, default revision 5.1.0) and converts each template into a
core.Document. Field handling follows the dataset card (huggingface.co/datasets/cyberagent/crello):

  canvas_width/height   pixels
  left/top/width/height pixels in v5 (normalized to [0,1] in v4 and earlier; detected and scaled)
  angle                 degrees in v5 (radians in v4 and earlier; converted)
  type                  SvgElement | TextElement | ImageElement | ColoredBackground | SvgMaskElement
  text_color            per-character RGBA  -> the element's dominant text color
  font_bold/italic      per-character flags -> majority vote per element
  text_line             per-character line index -> explicit line breaks
  capitalize            -> text is upper-cased
  color                 palette of vector elements (only present when the original had one)
  image                 pre-rendered PNG of each element -> image asset
  preview               pre-rendered PNG of the whole design -> render-fidelity check

Element mapping:
  TextElement        -> "text"
  ImageElement       -> "image" with its pre-rendered asset
  Svg/SvgMaskElement -> "image" with its asset; when the palette has exactly one color the element
                        is TINTED (its alpha mask is filled with "fill"), so palette and recolor
                        tools change it exactly and invertibly
  ColoredBackground  -> the canvas background color; kept as a bottom image element only when it
                        is not a flat color (gradient or texture)

Filters applied in load() (all configurable, stats printed):
  at least one text element; at most max_elements elements; every text font available as a file
  (require_fonts); optional render-fidelity threshold against Crello's own preview.

If `datasets` or the data is unavailable, load() falls back to SYNTHETIC designs and says so in
capital letters. Real numbers must come from a run that printed "loaded N Crello designs".
"""
from __future__ import annotations

import glob
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.color import parse_color, to_hex
from ..core.design import Document
from .base import Domain

DEFAULT_REVISION = "5.1.0"


# ----------------------------------------------------------------------------- helpers
def _decoder(features):
    """Return decode(key, value) that maps class-label ints to names when features know them."""
    def decode(key: str, val: Any) -> Any:
        if features is None or key not in features:
            return val
        f = features[key]
        f = getattr(f, "feature", f)            # Sequence(ClassLabel) -> ClassLabel
        if hasattr(f, "int2str") and isinstance(val, (int, np.integer)):
            try:
                return f.int2str(int(val))
            except Exception:
                return val
        return val
    return decode


def _seq(s: Dict, key: str, m: int, default: Any = None) -> List[Any]:
    v = s.get(key)
    if v is None:
        return [default] * m
    v = list(v)
    return v + [default] * (m - len(v)) if len(v) < m else v


CRELLO_TYPES = ["SvgElement", "TextElement", "ImageElement", "ColoredBackground", "SvgMaskElement"]


def _etype(name: Any) -> str:
    if isinstance(name, (int, np.integer)) and 0 <= int(name) < len(CRELLO_TYPES):
        name = CRELLO_TYPES[int(name)]           # v5 class order, from the dataset card
    n = str(name).lower()
    if "text" in n:
        return "text"
    if "background" in n:
        return "background"
    if "svg" in n or "mask" in n:
        return "svg"
    return "image"


def _major(revision: str) -> int:
    try:
        return int(str(revision).split(".")[0])
    except Exception:
        return 5


def _looks_normalized(s: Dict, W: int, H: int) -> bool:
    """True when geometry is in [0,1] (Crello v4 and earlier)."""
    vals = [float(v) for k in ("left", "top", "width", "height") for v in (s.get(k) or [])]
    return bool(vals) and max(abs(v) for v in vals) <= 1.5 and max(W, H) > 64


def _asset_stats(img) -> Tuple[Optional[str], float, float]:
    """(alpha-weighted mean color, color std, alpha coverage) of an RGBA image."""
    try:
        a = np.asarray(img.convert("RGBA").resize((64, 64)), dtype=np.float32)
    except Exception:
        return None, 0.0, 0.0
    w = a[..., 3] / 255.0
    cov = float(w.mean())
    if w.sum() < 1e-6:
        return None, 0.0, 0.0
    mean = (a[..., :3] * w[..., None]).sum(axis=(0, 1)) / w.sum()
    std = float(np.sqrt((((a[..., :3] - mean) ** 2) * w[..., None]).sum(axis=(0, 1)) / w.sum()).mean())
    return to_hex([int(round(x)) for x in mean]), std, cov


def _dominant_color(chars: str, colors: List[Any]) -> Optional[Tuple[str, float]]:
    votes: Counter = Counter()
    alphas: Dict[str, List[float]] = {}
    for ch, c in zip(chars, colors):
        if ch.isspace():
            continue
        pc = parse_color(c)
        if pc:
            votes[pc[0]] += 1
            alphas.setdefault(pc[0], []).append(pc[1])
    if not votes:
        return None
    hexc = votes.most_common(1)[0][0]
    return hexc, float(np.mean(alphas[hexc]))


def _majority(chars: str, flags: Optional[List[Any]]) -> bool:
    if not flags:
        return False
    vals = [bool(f) for ch, f in zip(chars, flags) if not ch.isspace()]
    return bool(vals) and sum(vals) * 2 > len(vals)


def _with_line_breaks(text: str, lines: Optional[List[Any]]) -> str:
    """Insert explicit line breaks where Crello's per-character line index changes."""
    if not lines or len(lines) != len(text):
        return text
    out: List[str] = []
    prev = None
    for ch, li in zip(text, lines):
        if ch == "\n":
            out.append("\n")
            prev = li
            continue
        if prev is not None and li != prev and out and out[-1] != "\n":
            while out and out[-1] == " ":
                out.pop()
            out.append("\n")
            if ch == " ":
                prev = li
                continue
        out.append(ch)
        prev = li
    return "".join(out)


def _palette(raw: Any) -> List[str]:
    out = []
    for c in raw or []:
        pc = parse_color(c)
        if pc and pc[0] not in out:
            out.append(pc[0])
    return out


# ----------------------------------------------------------------------------- conversion
def crello_to_doc(s: Dict, features=None, idx: int = 0, split: str = "test",
                  revision: str = DEFAULT_REVISION) -> Document:
    """Convert one Crello record (a dict as `datasets` yields it) into a Document."""
    decode = _decoder(features)
    W = int(s.get("canvas_width") or 1024)
    H = int(s.get("canvas_height") or 1024)
    types = list(s.get("type") or [])
    m = len(types)
    scale = _looks_normalized(s, W, H)
    deg = _major(revision) >= 5
    L, T = _seq(s, "left", m, 0.0), _seq(s, "top", m, 0.0)
    Wd, Hd = _seq(s, "width", m, 0.0), _seq(s, "height", m, 0.0)
    ANG, OP = _seq(s, "angle", m, 0.0), _seq(s, "opacity", m, 1.0)
    TXT, FONT, FS = _seq(s, "text", m, ""), _seq(s, "font", m, None), _seq(s, "font_size", m, 24.0)
    ALIGN, TCOL = _seq(s, "text_align", m, "left"), _seq(s, "text_color", m, None)
    BOLD, ITAL = _seq(s, "font_bold", m, None), _seq(s, "font_italic", m, None)
    LINE, CAP = _seq(s, "text_line", m, None), _seq(s, "capitalize", m, False)
    LH, LS = _seq(s, "line_height", m, 1.0), _seq(s, "letter_spacing", m, 0.0)
    COL, IMG = _seq(s, "color", m, None), _seq(s, "image", m, None)

    background = "#ffffff"
    elements: List[Dict] = []
    assets: Dict[str, Any] = {}
    skipped = Counter()
    for j in range(m):
        kind = _etype(decode("type", types[j]))
        x, y, w, h = (float(v or 0.0) for v in (L[j], T[j], Wd[j], Hd[j]))
        if scale:
            x, y, w, h = x * W, y * H, w * W, h * H
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        if w < 1 or h < 1:
            skipped["zero_size"] += 1
            continue
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            skipped["off_canvas"] += 1
            continue
        ang = float(ANG[j] or 0.0)
        ang = ang if deg else math.degrees(ang)
        ang = round((ang + 180.0) % 360.0 - 180.0, 4)
        op = float(OP[j] if OP[j] is not None else 1.0)
        eid = f"el-{j}"
        base = {"id": eid, "left": x, "top": y, "width": w, "height": h,
                "opacity": round(op, 4), "angle": ang}
        img = IMG[j]
        if img is not None and hasattr(img, "convert"):
            img = img.convert("RGBA")
        else:
            img = None
        pal = _palette(COL[j])

        if kind == "text":
            raw = str(TXT[j] or "")
            if not raw.strip():
                skipped["empty_text"] += 1
                continue
            text = _with_line_breaks(raw, LINE[j])
            if CAP[j]:
                text = text.upper()
            dom = _dominant_color(raw, list(TCOL[j] or [])) if TCOL[j] else None
            color = dom[0] if dom else (pal[0] if pal else "#000000")
            if dom and dom[1] < 0.999:
                base["opacity"] = round(op * dom[1], 4)
            align = str(decode("text_align", ALIGN[j]) or "left").lower()
            el = dict(base, type="text", text=text, color=color,
                      font=str(decode("font", FONT[j]) or "Roboto"),
                      font_size=round(float(FS[j] or 24.0), 2),
                      font_weight="bold" if _majority(raw, BOLD[j]) else "normal",
                      italic=_majority(raw, ITAL[j]),
                      text_align=align if align in ("left", "center", "right") else "left",
                      line_height=round(float(LH[j] or 1.0), 3),
                      letter_spacing=round(float(LS[j] or 0.0), 3))
            elements.append(el)
            continue

        mean, std, cov = _asset_stats(img) if img is not None else (None, 0.0, 0.0)
        if kind == "background":
            flat = img is None or std < 6.0
            background = pal[0] if pal else (mean or background)
            if flat:
                continue                              # a flat color is the canvas background
        if kind == "svg" and len(pal) == 1:
            el = dict(base, type="image", fill=pal[0], tint=True, avg_color=pal[0])
            if img is not None:
                assets[eid] = img
                el["asset_id"] = eid
            else:
                el["type"], el["shape"] = "shape", "rect"
                el.pop("tint")
            elements.append(el)
            continue
        if img is None:
            if pal:
                elements.append(dict(base, type="shape", shape="rect", fill=pal[0]))
            else:
                skipped["no_asset"] += 1
            continue
        assets[eid] = img
        el = dict(base, type="image", asset_id=eid, avg_color=mean or "#808080")
        if pal:
            el["palette"] = pal
        elements.append(el)

    meta = {"source": "crello", "split": split, "index": idx, "crello_id": s.get("id"),
            "revision": revision, "skipped": dict(skipped),
            "format": str(decode("format", s.get("format"))) if s.get("format") is not None else None,
            "group": str(decode("group", s.get("group"))) if s.get("group") is not None else None}
    return Document(W, H, background, elements, assets, metadata=meta)


def render_fidelity(doc: Document, preview, fonts_dir: Optional[str] = None,
                    size: int = 256) -> Optional[float]:
    """Mean absolute pixel difference (0..1) between our render and Crello's own preview."""
    if preview is None or not hasattr(preview, "convert"):
        return None
    from ..core.render import render
    a = np.asarray(render(doc, fonts_dir=fonts_dir).convert("RGB").resize((size, size)), dtype=np.float32)
    b = np.asarray(_flatten(preview).resize((size, size)), dtype=np.float32)
    return float(np.abs(a - b).mean() / 255.0)


def _flatten(img):
    """Composite a possibly transparent preview over white."""
    from PIL import Image
    rgba = img.convert("RGBA")
    base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    base.alpha_composite(rgba)
    return base.convert("RGB")


# ----------------------------------------------------------------------------- domain
class GraphicDesignDomain(Domain):
    """Crello loader with filters. Typical use:
        dom = GraphicDesignDomain(data_dir=None)        # None -> download from the HF Hub
        designs = dom.load(1000, split="test")
        print(dom.last_stats)
    """
    name = "graphic_design"

    def __init__(self, revision: str = DEFAULT_REVISION, data_dir: Optional[str] = None,
                 streaming: bool = False, seed: int = 0, max_elements: int = 30,
                 require_fonts: bool = True, fonts_dir: Optional[str] = None,
                 max_render_diff: Optional[float] = None, compute_fidelity: bool = True,
                 allow_synthetic: bool = True):
        self.revision = revision
        self.data_dir = data_dir or os.environ.get("VERIFIX_CRELLO_DIR")
        self.streaming = streaming
        self.seed = seed
        self.max_elements = max_elements
        self.require_fonts = require_fonts
        self.fonts_dir = fonts_dir
        self.max_render_diff = max_render_diff
        self.compute_fidelity = compute_fidelity or max_render_diff is not None
        self.allow_synthetic = allow_synthetic
        self.last_stats: Dict[str, Any] = {}

    # ---- data access ----
    def open(self, split: str):
        from datasets import load_dataset
        if self.data_dir:
            files = sorted(glob.glob(os.path.join(self.data_dir, "**", f"*{split}*.parquet"),
                                     recursive=True))
            if files:
                ds = load_dataset("parquet", data_files=files, split="train",
                                  streaming=self.streaming)
            else:
                ds = load_dataset(self.data_dir, split=split, streaming=self.streaming)
        else:
            ds = load_dataset("cyberagent/crello", revision=self.revision, split=split,
                              streaming=self.streaming)
        if self.streaming:
            return ds.shuffle(seed=self.seed, buffer_size=1000)
        return ds.shuffle(seed=self.seed)

    def iter_records(self, split: str):
        ds = self.open(split)
        feats = getattr(ds, "features", None)
        font_f = getattr((feats or {}).get("font"), "feature", None) if feats else None
        if not hasattr(font_f, "int2str"):
            raise RuntimeError(
                "Crello records have no class-label metadata for 'font', so font ids cannot be "
                "turned into names. Load from the Hub, or download the parquet files with "
                "scripts/download_crello.py (which keeps the metadata).")
        for i, s in enumerate(ds):
            yield i, s, feats

    # ---- load with filters ----
    def load(self, n: int, split: str = "test", max_scan: Optional[int] = None,
             skip: int = 0) -> List[Document]:
        """Return n designs of `split` that pass the filters, in a fixed shuffled order (seed).
        skip=K drops the first K designs that pass, so two calls on the same split can take
        disjoint subsets (e.g. index designs [0, 300) and ranker-training designs [300, 500))."""
        try:
            records = self.iter_records(split)
            first = next(records)
        except Exception as e:
            if not self.allow_synthetic:
                raise
            print(f"[graphic_design] WARNING: CRELLO UNAVAILABLE ({type(e).__name__}: {e}); "
                  f"USING SYNTHETIC DESIGNS. Do not report numbers from this run.")
            return synthetic_designs(n, split=split)
        from ..core.render import font_available, resolve_fonts_dir
        fonts_dir = resolve_fonts_dir(self.fonts_dir)
        out: List[Document] = []
        drops: Counter = Counter()
        missing_fonts: Counter = Counter()
        fidelity: List[float] = []
        max_scan = max_scan or max(5 * (n + skip), n + skip + 200)
        passed = 0

        def chain():
            yield first
            yield from records

        for scanned, (i, s, feats) in enumerate(chain()):
            if len(out) >= n or scanned >= max_scan:
                break
            try:
                doc = crello_to_doc(s, feats, i, split, self.revision)
            except Exception as e:  # malformed record
                drops[f"convert_error:{type(e).__name__}"] += 1
                continue
            texts = [e for e in doc.elements if e.get("type") == "text"]
            if not texts:
                drops["no_text"] += 1
                continue
            if self.max_elements and len(doc.elements) > self.max_elements:
                drops["too_many_elements"] += 1
                continue
            if self.require_fonts:
                miss = sorted({e["font"] for e in texts if not font_available(e["font"], fonts_dir)})
                if miss:
                    missing_fonts.update(miss)
                    drops["missing_font"] += 1
                    continue
            if self.compute_fidelity:
                fd = render_fidelity(doc, s.get("preview"), fonts_dir)
                if fd is not None:
                    doc.metadata["render_diff"] = round(fd, 4)
                    fidelity.append(fd)
                    if self.max_render_diff is not None and fd > self.max_render_diff:
                        drops["render_mismatch"] += 1
                        continue
            passed += 1
            if passed <= skip:
                continue
            out.append(doc)
        scanned_total = passed + sum(drops.values())
        self.last_stats = {"split": split, "kept": len(out), "skipped_first": skip, "scanned": scanned_total,
                           "drops": dict(drops), "missing_fonts": dict(missing_fonts.most_common()),
                           "render_diff": _summary(fidelity)}
        print(f"[graphic_design] loaded {len(out)} Crello designs ({split}, rev {self.revision}); "
              f"scanned {scanned_total}; dropped {dict(drops)}")
        if missing_fonts:
            print(f"[graphic_design] {len(missing_fonts)} missing font families, most common: "
                  f"{', '.join(f for f, _ in missing_fonts.most_common(15))}")
        if fidelity:
            print(f"[graphic_design] render vs Crello preview (mean abs diff): {_summary(fidelity)}")
        if len(out) < n:
            print(f"[graphic_design] WARNING: asked for {n}, kept {len(out)}. Fetch missing fonts, "
                  f"relax filters, or raise max_scan.")
        return out

    def real_defects(self, n: int, split: str = "test") -> List[Document]:
        # Transfer split: designs with defects that no tool call produced (annotated Crello
        # designs or generator outputs). Not built yet; see README "What is still missing".
        return []


def _summary(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4),
            "p50": round(float(np.percentile(a, 50)), 4), "p90": round(float(np.percentile(a, 90)), 4),
            "max": round(float(a.max()), 4)}


# ----------------------------------------------------------------------------- synthetic
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
    """Offline stand-ins for tests and smoke runs. Never report numbers from these."""
    rnd = random.Random(42)
    fonts = ["Roboto", "Playfair Display", "Montserrat", "Oswald", "Lora", "Poppins"]
    palettes = [("#1A1A2E", "#E94560", "#F5F5F5"), ("#FFFFFF", "#0F3460", "#333333"),
                ("#F4EBD0", "#B85042", "#2C3E50"), ("#0B3D2E", "#F2C14E", "#FFFFFF")]
    out = []
    for i in range(n):
        bg, accent, ink = palettes[i % len(palettes)]
        W, H = 800, 1000
        assets = {f"asset-{k}": _asset(1000 * i + k) for k in range(3)}
        mean, _, _ = _asset_stats(assets["asset-0"])
        els = [
            {"id": "el-0", "type": "image", "asset_id": "asset-0", "left": 120, "top": 340,
             "width": 560, "height": 380, "opacity": 1.0, "angle": 0.0, "avg_color": mean},
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
    print(f"[graphic_design] generated {n} SYNTHETIC designs")
    return out
