"""
core/render.py - Deterministic renderer + set-of-marks overlay.

The agent observes IMAGES, never the document. Two renders:
  - clean  (marked=False): what the JUDGE sees.
  - marked (marked=True) : numbered red tags on each element, what the PLANNER and COMPILER see,
                           so an image-only agent can say "element 3" (Set-of-Mark prompting).

Supports every property the Perturb & Invert tool set (core/tools.py) can change: text color,
font, size, alignment, box geometry, effects; image asset, crop, filter, hue/saturation; shapes;
position, size, rotation, z-order; background; global contrast and saturation.

Font fidelity depends on the .ttf files in fonts_dir (see scripts/fetch_fonts.py). A family with
no file falls back to DejaVuSans; the certification's visibility check catches perturbations that
the fallback makes invisible. For camera-ready fidelity, swap in an HTML + headless-browser
renderer behind the same render(doc, marked) signature.
"""
from __future__ import annotations

import os
from typing import Dict

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .color import rgb
from .design import Document, prop

_MAX_SIDE = 8192


def _font_path(family: str, weight: str, fonts_dir: str):
    stems = []
    if family:
        f = str(family)
        stems = [f.replace(" ", ""), f, f.replace(" ", "_"), f.replace(" ", "-")]
    bold = str(weight).lower() in ("bold", "600", "700", "800", "900")
    for s in stems:
        cands = ([f"{s}-Bold.ttf", f"{s}-SemiBold.ttf"] if bold else []) + \
                [f"{s}-Regular.ttf", f"{s}.ttf", f"{s}[wght].ttf"]
        for c in cands:
            p = os.path.join(fonts_dir, c)
            if os.path.exists(p):
                return p
    return None


def _load_font(family: str, size: float, weight: str, fonts_dir: str):
    size = max(4, int(round(float(size or 24))))
    path = _font_path(family, weight, fonts_dir)
    for c in [path, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
              if str(weight).lower() in ("bold", "700") else None,
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def font_available(family: str, fonts_dir: str = "fonts") -> bool:
    return _font_path(family, "normal", fonts_dir) is not None


def _text_tile(e: Dict, w: int, h: int, fonts_dir: str) -> Image.Image:
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    font = _load_font(prop(e, "font"), prop(e, "font_size") or 24, prop(e, "font_weight"), fonts_dir)
    text = str(prop(e, "text") or "")
    color = rgb(prop(e, "color") or "#000000")
    align = prop(e, "text_align") or "left"
    try:
        x0, y0, x1, y1 = draw.multiline_textbbox((0, 0), text, font=font, align=align)
        tw = x1 - x0
    except Exception:
        tw = 0
    x = {"left": 0, "center": max(0, (w - tw) // 2), "right": max(0, w - tw)}.get(align, 0)
    effect = prop(e, "effect")
    if effect == "shadow":
        draw.multiline_text((x + 3, 3), text, fill=(0, 0, 0, 140), font=font, align=align)
    if effect == "glow":
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).multiline_text((x, 0), text, fill=(255, 255, 210, 220), font=font,
                                           align=align, stroke_width=4, stroke_fill=(255, 255, 210, 220))
        tile.alpha_composite(glow.filter(ImageFilter.GaussianBlur(4)))
    if effect == "outline":
        draw.multiline_text((x, 0), text, fill=color, font=font, align=align,
                            stroke_width=2, stroke_fill=(0, 0, 0))
    else:
        draw.multiline_text((x, 0), text, fill=color, font=font, align=align)
    return tile


def _image_tile(doc: Document, e: Dict, w: int, h: int) -> Image.Image:
    asset = doc.assets.get(prop(e, "asset_id")) if e.get("type") == "image" else None
    if asset is not None:
        src = asset.convert("RGBA")
        c = prop(e, "crop") or [0, 0, 1, 1]
        sw, sh = src.size
        box = (int(c[0] * sw), int(c[1] * sh), max(int(c[0] * sw) + 1, int(c[2] * sw)),
               max(int(c[1] * sh) + 1, int(c[3] * sh)))
        tile = src.crop(box).resize((w, h))
    else:
        tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fill = rgb(prop(e, "fill") or "#CCCCCC") + (255,)
        d = ImageDraw.Draw(tile)
        if e.get("type") == "shape" and prop(e, "shape") == "ellipse":
            d.ellipse([0, 0, w - 1, h - 1], fill=fill)
        else:
            d.rectangle([0, 0, w - 1, h - 1], fill=fill)
    alpha = tile.getchannel("A")
    rgbimg = tile.convert("RGB")
    flt = prop(e, "filter")
    if flt == "grayscale":
        rgbimg = ImageOps.grayscale(rgbimg).convert("RGB")
    elif flt == "sepia":
        g = ImageOps.grayscale(rgbimg)
        rgbimg = ImageOps.colorize(g, (40, 26, 13), (255, 240, 192))
    elif flt == "blur":
        rgbimg = rgbimg.filter(ImageFilter.GaussianBlur(3))
    elif flt == "invert":
        rgbimg = ImageOps.invert(rgbimg)
    hue, sat = float(prop(e, "hue_shift") or 0.0), float(prop(e, "sat_scale") or 1.0)
    if hue or sat != 1.0:
        hsv = rgbimg.convert("HSV")
        hch, sch, vch = hsv.split()
        shift = int(round(hue / 360.0 * 255)) % 256
        hch = hch.point(lambda v: (v + shift) % 256)
        sch = sch.point(lambda v: max(0, min(255, int(v * sat))))
        rgbimg = Image.merge("HSV", (hch, sch, vch)).convert("RGB")
    out = rgbimg.convert("RGBA")
    out.putalpha(alpha)
    return out


def render(doc: Document, marked: bool = False, fonts_dir: str = "fonts") -> Image.Image:
    img = Image.new("RGBA", (doc.width, doc.height), rgb(doc.background) + (255,))
    for e in doc.elements:
        x, y = int(prop(e, "left") or 0), int(prop(e, "top") or 0)
        w = max(1, min(int(prop(e, "width") or 100), _MAX_SIDE))
        h = max(1, min(int(prop(e, "height") or 40), _MAX_SIDE))
        if e.get("type", "text") == "text":
            tile = _text_tile(e, w, h, fonts_dir)
        else:
            tile = _image_tile(doc, e, w, h)
        op = float(prop(e, "opacity") if prop(e, "opacity") is not None else 1.0)
        if op < 1.0:
            tile.putalpha(tile.getchannel("A").point(lambda v: int(v * max(0.0, op))))
        ang = float(prop(e, "angle") or 0.0)
        if ang:
            cx, cy = x + w / 2, y + h / 2
            tile = tile.rotate(-ang, expand=True, resample=Image.BICUBIC)
            x, y = int(cx - tile.width / 2), int(cy - tile.height / 2)
        img.paste(tile, (x, y), tile)
    out = img.convert("RGB")
    c, s = doc.adjust.get("contrast", 0.0), doc.adjust.get("saturation", 0.0)
    if c:
        out = ImageEnhance.Contrast(out).enhance(max(0.05, 1.0 + c))
    if s:
        out = ImageEnhance.Color(out).enhance(max(0.0, 1.0 + s))
    if marked:
        draw = ImageDraw.Draw(out)
        for idx, e in enumerate(doc.elements):
            x, y = int(prop(e, "left") or 0), int(prop(e, "top") or 0)
            draw.rectangle([x, y, x + 22, y + 18], fill=(255, 0, 0))
            draw.text((x + 4, y + 2), str(idx), fill=(255, 255, 255))
    return out


def element_index_map(doc: Document) -> Dict[str, str]:
    """mark number (as string) -> element id, matching the marked render order."""
    return {str(i): e["id"] for i, e in enumerate(doc.elements)}
