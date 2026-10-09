"""
core/render.py - Deterministic renderer + set-of-marks overlay.

The agent observes IMAGES, never the document. Two renders:
  - clean  (marked=False): what the JUDGE sees.
  - marked (marked=True) : numbered red tags on each element, what the PLANNER and COMPILER see,
                           so an image-only agent can say "element 3" (Set-of-Mark prompting).

Supports every property the Perturb & Invert tool set (core/tools.py) can change: text color,
font, weight, size, alignment, box geometry, effects; image asset, crop, filter, hue/saturation,
tint (single-color vector shapes); shapes; position, size, rotation, z-order; background; global
contrast and saturation. Text is drawn line by line with the element's line height and alignment.

Fonts: looked up in the fonts directory (resolve_fonts_dir: argument, else $VERIFIX_FONTS_DIR,
else <repo>/fonts). Names such as "Montserrat Bold" are split into family + weight. Variable fonts
are set to their Bold / Italic instance when needed. A family with no file falls back to
DejaVuSans; the Crello loader drops designs whose fonts are missing (require_fonts), and the
certification's visibility check drops perturbations a fallback would make invisible.
Not rendered: letter spacing (kept as a property), per-character styling (the loader keeps each
text element's dominant color and weight).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .color import rgb
from .design import Document, prop

_MAX_SIDE = 8192
_REPO_FONTS = Path(__file__).resolve().parents[2] / "fonts"
_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_WEIGHT_WORDS = {
    "thin": "100", "hairline": "100", "extralight": "200", "ultralight": "200", "light": "300",
    "regular": "400", "normal": "400", "book": "400", "medium": "500", "semibold": "600",
    "demibold": "600", "bold": "700", "extrabold": "800", "ultrabold": "800", "black": "900",
    "heavy": "900",
}
_BOLD = {"bold", "600", "700", "800", "900"}


def resolve_fonts_dir(fonts_dir: Optional[str] = None) -> str:
    if fonts_dir:
        return str(fonts_dir)
    env = os.environ.get("VERIFIX_FONTS_DIR")
    if env:
        return env
    if _REPO_FONTS.is_dir():
        return str(_REPO_FONTS)
    return "fonts"


def split_family_weight(name: str) -> Tuple[str, Optional[str], bool]:
    """'Montserrat Bold Italic' -> ('Montserrat', '700', True). Unknown words stay in the family."""
    words = str(name or "").replace("-", " ").replace("_", " ").split()
    weight, italic = None, False
    while words:
        w = words[-1].lower()
        two = (words[-2] + words[-1]).lower() if len(words) >= 2 else ""
        if w in ("italic", "oblique"):
            italic = True
            words.pop()
        elif two in _WEIGHT_WORDS and len(words) > 2:
            weight = weight or _WEIGHT_WORDS[two]
            words = words[:-2]
        elif w in _WEIGHT_WORDS and len(words) > 1:
            weight = weight or _WEIGHT_WORDS[w]
            words.pop()
        else:
            break
    return (" ".join(words) or str(name)), weight, italic


def _key(family: str) -> str:
    return "".join(ch for ch in str(family).lower() if ch.isalnum())


@lru_cache(maxsize=64)
def _font_index(fonts_dir: str, _mtime: float) -> Dict[str, list]:
    """family key -> [(path, style, variable)] for every .ttf/.otf in fonts_dir.
    File names follow Google Fonts: Family-Style.ttf (static) or Family[axes].ttf /
    Family-Italic[axes].ttf (variable)."""
    idx: Dict[str, list] = {}
    if not os.path.isdir(fonts_dir):
        return idx
    for fn in os.listdir(fonts_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".ttf", ".otf"):
            continue
        variable = "[" in stem
        base = stem.split("[")[0]
        fam, _, style = base.partition("-")
        idx.setdefault(_key(fam), []).append((os.path.join(fonts_dir, fn), style.lower(), variable))
    return idx


def _style_weight(style: str) -> int:
    s = style.replace("italic", "")
    for word, w in sorted(_WEIGHT_WORDS.items(), key=lambda kv: -len(kv[0])):
        if word in s:
            return int(w)
    return 400


def _pick(files, bold: bool, italic: bool):
    """Prefer: matching italic, then a variable file or the static weight closest to the target."""
    target = 700 if bold else 400

    def score(f):
        _, style, var = f
        sc = 4.0 if (("italic" in style) == italic) else 0.0
        sc += 3.0 if var else 3.0 - abs(_style_weight(style) - target) / 200.0
        return sc
    return max(files, key=score) if files else None


@lru_cache(maxsize=4096)
def _font_path(family: str, weight: str, fonts_dir: str, italic: bool = False):
    """(path, is_variable) of the best file for family/weight/italic, or None."""
    if not family:
        return None
    try:
        mtime = os.path.getmtime(fonts_dir)
    except OSError:
        return None
    idx = _font_index(fonts_dir, mtime)
    fam, w2, it2 = split_family_weight(family)
    bold = str(w2 or weight).lower() in _BOLD
    for name in dict.fromkeys([str(family), fam]):
        f = _pick(idx.get(_key(name), []), bold, italic or it2)
        if f:
            return f[0], f[2]
    return None


@lru_cache(maxsize=4096)
def _font_obj(found, size: int, bold: bool, italic: bool):
    path, variable = found if found else (None, False)
    for c in [path, _DEJAVU_B if bold else None, _DEJAVU]:
        if not c or not os.path.exists(c):
            continue
        try:
            f = ImageFont.truetype(c, size)
        except Exception:
            continue
        if c == path and variable:
            try:
                axes = f.get_variation_axes()
                vals = []
                for ax in axes:
                    nm = ax.get("name", b"")
                    nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                    v = ax.get("default", ax.get("minimum", 0))
                    if nm.lower().startswith("weight") or nm.lower() == "wght":
                        v = 700 if bold else 400
                    elif nm.lower().startswith("italic") or nm.lower() == "ital":
                        v = 1 if italic else 0
                    vals.append(max(ax.get("minimum", v), min(ax.get("maximum", v), v)))
                f.set_variation_by_axes(vals)
            except Exception:
                pass
        return f
    return ImageFont.load_default()


def _load_font(family: str, size: float, weight: str, fonts_dir: str, italic: bool = False):
    size = max(4, int(round(float(size or 24))))
    _, w2, it2 = split_family_weight(family or "")
    bold = str(w2 or weight).lower() in _BOLD
    return _font_obj(_font_path(family or "", str(weight), fonts_dir, bool(italic)), size, bold,
                     bool(italic or it2))


def font_available(family: str, fonts_dir: Optional[str] = None) -> bool:
    return _font_path(str(family or ""), "normal", resolve_fonts_dir(fonts_dir)) is not None


def _text_tile(e: Dict, w: int, h: int, fonts_dir: str):
    """Returns (tile, ox, oy): the tile is centered on the element box and padded so text that
    overflows the box stays visible (as in a browser); (ox, oy) is the box origin inside it."""
    size = float(prop(e, "font_size") or 24)
    font = _load_font(prop(e, "font"), size, prop(e, "font_weight"), fonts_dir, bool(e.get("italic")))
    lines = str(prop(e, "text") or "").split("\n")
    color = rgb(prop(e, "color") or "#000000") + (255,)
    align = prop(e, "text_align") or "left"
    step = max(1.0, size * float(prop(e, "line_height") or 1.2))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = []
    for line in lines:
        try:
            x0, _, x1, _ = probe.textbbox((0, 0), line, font=font)
            widths.append(x1 - x0)
        except Exception:
            widths.append(0)
    try:
        asc, desc = font.getmetrics()
    except Exception:
        asc, desc = int(size), int(size * 0.25)
    text_h = int((len(lines) - 1) * step + max(step, asc + desc))
    effect = prop(e, "effect")
    extra = 8 if effect in ("shadow", "glow", "outline") else 2
    pw = max(0, max(widths + [0]) - w) + extra
    ph = max(0, text_h - h) + extra + int(max(0, (asc + desc) - step))
    ox, oy = pw, ph
    tile = Image.new("RGBA", (w + 2 * pw, h + 2 * ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    glow = Image.new("RGBA", tile.size, (0, 0, 0, 0)) if effect == "glow" else None
    for i, line in enumerate(lines):
        tw = widths[i]
        x = ox + {"left": 0, "center": (w - tw) // 2, "right": w - tw}.get(align, 0)
        # CSS-style line box: the glyphs sit centered in a line of height size * line_height
        # ("half-leading"), which is how Crello's own renderer places text.
        y = oy + int(round(i * step + (step - (asc + desc)) / 2))
        if effect == "shadow":
            draw.text((x + 3, y + 3), line, fill=(0, 0, 0, 140), font=font)
        if glow is not None:
            ImageDraw.Draw(glow).text((x, y), line, fill=(255, 255, 210, 220), font=font,
                                      stroke_width=4, stroke_fill=(255, 255, 210, 220))
        if effect == "outline":
            draw.text((x, y), line, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        else:
            draw.text((x, y), line, fill=color, font=font)
    if glow is not None:
        base = glow.filter(ImageFilter.GaussianBlur(4))
        base.alpha_composite(tile)
        tile = base
    return tile, ox, oy


def _image_tile(doc: Document, e: Dict, w: int, h: int) -> Image.Image:
    asset = doc.assets.get(prop(e, "asset_id")) if e.get("type") == "image" else None
    if asset is not None:
        src = asset.convert("RGBA")
        c = prop(e, "crop") or [0, 0, 1, 1]
        sw, sh = src.size
        box = (int(c[0] * sw), int(c[1] * sh), max(int(c[0] * sw) + 1, int(c[2] * sw)),
               max(int(c[1] * sh) + 1, int(c[3] * sh)))
        tile = src.crop(box).resize((w, h))
        if e.get("tint") and prop(e, "fill"):
            solid = Image.new("RGBA", (w, h), rgb(prop(e, "fill")) + (255,))
            solid.putalpha(tile.getchannel("A"))
            tile = solid
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


def render(doc: Document, marked: bool = False, fonts_dir: Optional[str] = None) -> Image.Image:
    fonts_dir = resolve_fonts_dir(fonts_dir)
    img = Image.new("RGBA", (doc.width, doc.height), rgb(doc.background) + (255,))
    for e in doc.elements:
        x, y = int(prop(e, "left") or 0), int(prop(e, "top") or 0)
        w = max(1, min(int(prop(e, "width") or 100), _MAX_SIDE))
        h = max(1, min(int(prop(e, "height") or 40), _MAX_SIDE))
        ox = oy = 0
        if e.get("type", "text") == "text":
            tile, ox, oy = _text_tile(e, w, h, fonts_dir)
        else:
            tile = _image_tile(doc, e, w, h)
        op = float(prop(e, "opacity") if prop(e, "opacity") is not None else 1.0)
        if op < 1.0:
            tile.putalpha(tile.getchannel("A").point(lambda v: int(v * max(0.0, op))))
        ang = float(prop(e, "angle") or 0.0)
        if ang:
            cx, cy = x + w / 2, y + h / 2
            tile = tile.rotate(-ang, expand=True, resample=Image.BICUBIC)
            x, y = int(round(cx - tile.width / 2)), int(round(cy - tile.height / 2))
        else:
            x, y = x - ox, y - oy
        _composite(img, tile, x, y)
    out = img.convert("RGB")
    c, s = doc.adjust.get("contrast", 0.0), doc.adjust.get("saturation", 0.0)
    if c:
        out = ImageEnhance.Contrast(out).enhance(max(0.05, 1.0 + c))
    if s:
        out = ImageEnhance.Color(out).enhance(max(0.0, 1.0 + s))
    if marked:
        draw_marks(out, doc)
    return out


def _composite(base: Image.Image, tile: Image.Image, x: int, y: int) -> None:
    """Alpha-composite tile onto base at (x, y), clipping to the canvas."""
    l, t = max(0, -x), max(0, -y)
    r, b = min(tile.width, base.width - x), min(tile.height, base.height - y)
    if r <= l or b <= t:
        return
    base.alpha_composite(tile.crop((l, t, r, b)), (x + l, y + t))


def draw_marks(img: Image.Image, doc: Document) -> None:
    """Numbered tags at each element's top-left corner, scaled to the canvas size."""
    draw = ImageDraw.Draw(img)
    s = max(18, int(round(max(doc.width, doc.height) / 45)))
    try:
        font = ImageFont.truetype(_DEJAVU_B, int(s * 0.75))
    except Exception:
        font = ImageFont.load_default()
    for idx, e in enumerate(doc.elements):
        x = min(max(0, int(prop(e, "left") or 0)), doc.width - s)
        y = min(max(0, int(prop(e, "top") or 0)), doc.height - s)
        label = str(idx)
        bw = int(s * (0.6 + 0.5 * len(label)))
        draw.rectangle([x, y, x + bw, y + s], fill=(255, 0, 0))
        draw.text((x + s * 0.15, y + s * 0.08), label, fill=(255, 255, 255), font=font)


def element_index_map(doc: Document) -> Dict[str, str]:
    """mark number (as string) -> element id, matching the marked render order."""
    return {str(i): e["id"] for i, e in enumerate(doc.elements)}
