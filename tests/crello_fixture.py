"""
tests/crello_fixture.py - A tiny dataset in the Crello v5 schema, written as parquet.

Used by the tests and by scripts/smoke_test.py to exercise the REAL loader code path
(datasets + parquet + class-label decoding + pixel geometry + per-character text attributes)
without network access. The designs are invented; never report numbers from them.

    python -m tests.crello_fixture --out data/crello_fixture --n 12
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

TYPES = ["SvgElement", "TextElement", "ImageElement", "ColoredBackground", "SvgMaskElement"]
FONTS = ["Montserrat", "Playfair Display", "Roboto", "Oswald", "Lora", "Poppins"]
ALIGN = ["left", "center", "right"]


def features():
    from datasets import ClassLabel, Features, Image as HFImage, Sequence, Value
    f32 = Sequence(Value("float32"))
    return Features({
        "id": Value("string"), "length": Value("int64"),
        "group": ClassLabel(names=["SM", "HC"]), "format": ClassLabel(names=["Instagram", "Poster"]),
        "canvas_width": Value("int64"), "canvas_height": Value("int64"),
        "category": ClassLabel(names=["holidays", "business"]), "title": Value("string"),
        "suitability": Sequence(ClassLabel(names=["mobile"])), "keywords": Sequence(Value("string")),
        "industries": Sequence(ClassLabel(names=["marketingAds"])), "preview": HFImage(),
        "cluster_index": Value("int64"),
        "type": Sequence(ClassLabel(names=TYPES)),
        "left": f32, "top": f32, "width": f32, "height": f32, "angle": f32, "opacity": f32,
        "color": Sequence(Sequence(Value("string"))), "image": Sequence(HFImage()),
        "text": Sequence(Value("string")), "font": Sequence(ClassLabel(names=FONTS)),
        "font_size": f32, "text_align": Sequence(ClassLabel(names=ALIGN)),
        "font_bold": Sequence(Sequence(Value("bool"))), "font_italic": Sequence(Sequence(Value("bool"))),
        "text_color": Sequence(Sequence(Value("string"))), "text_line": Sequence(Sequence(Value("int64"))),
        "capitalize": Sequence(Value("bool")), "line_height": f32, "letter_spacing": f32,
    })


def _photo(rnd, w=256, h=192):
    img = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(img)
    c1 = [rnd.randrange(256) for _ in range(3)]
    c2 = [rnd.randrange(256) for _ in range(3)]
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3)) + (255,))
    return img


def _blob(w=200, h=200):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 0, w - 1, h - 1], fill=(30, 30, 30, 255))
    return img


def record(i: int, rnd: random.Random):
    W, H = rnd.choice([(1080, 1080), (1080, 1350), (800, 1200)])
    bg = rnd.choice(["rgb(250, 245, 235)", "rgb(20, 24, 40)", "rgb(235, 242, 250)"])
    dark = bg.startswith("rgb(20")
    ink = "rgba(245, 245, 245, 1)" if dark else "rgba(25, 25, 35, 1)"
    accent = rnd.choice(["rgba(220, 60, 70, 1)", "rgba(30, 110, 200, 1)", "rgba(240, 170, 30, 1)"])
    title = rnd.choice(["Grand opening", "Summer sale now on", "Join our workshop"])
    words = title.split(" ")
    # title over two lines: per-character line index
    first = " ".join(words[:max(1, len(words) // 2)])
    lines = [0] * (len(first) + 1) + [1] * (len(title) - len(first) - 1)
    sub = rnd.choice(["Saturday 10 am", "Up to 40% off", "Free entry for members"])
    els = [
        dict(type=3, left=0, top=0, width=W, height=H, angle=0, opacity=1, color=[bg],
             image=Image.new("RGBA", (64, 64), tuple(int(v) for v in bg[4:-1].split(",")) + (255,)),
             text="", font=0, font_size=0, text_align=0, font_bold=[], font_italic=[],
             text_color=[], text_line=[], capitalize=False, line_height=1, letter_spacing=0),
        dict(type=2, left=int(W * .1), top=int(H * .42), width=int(W * .8), height=int(H * .38),
             angle=0, opacity=1, color=[], image=_photo(rnd), text="", font=0, font_size=0,
             text_align=0, font_bold=[], font_italic=[], text_color=[], text_line=[],
             capitalize=False, line_height=1, letter_spacing=0),
        dict(type=0, left=int(W * .78), top=int(H * .05), width=int(W * .14), height=int(W * .14),
             angle=rnd.choice([0.0, 12.0]), opacity=1, color=[accent.replace("rgba", "rgb")[:-4] + ")"],
             image=_blob(), text="", font=0, font_size=0, text_align=0, font_bold=[], font_italic=[],
             text_color=[], text_line=[], capitalize=False, line_height=1, letter_spacing=0),
        dict(type=1, left=int(W * .1), top=int(H * .08), width=int(W * .66), height=int(H * .22),
             angle=0, opacity=1, color=[], image=Image.new("RGBA", (8, 8)), text=title,
             font=rnd.randrange(len(FONTS)), font_size=round(H * 0.05, 1), text_align=0,
             font_bold=[True] * len(title), font_italic=[False] * len(title),
             text_color=[accent] * len(title), text_line=lines, capitalize=rnd.random() < 0.5,
             line_height=1.1, letter_spacing=0),
        dict(type=1, left=int(W * .1), top=int(H * .84), width=int(W * .8), height=int(H * .06),
             angle=0, opacity=1, color=[], image=Image.new("RGBA", (8, 8)), text=sub,
             font=rnd.randrange(len(FONTS)), font_size=round(H * 0.035, 1), text_align=1,
             font_bold=[False] * len(sub), font_italic=[False] * len(sub),
             text_color=[ink] * len(sub), text_line=[0] * len(sub), capitalize=False,
             line_height=1.2, letter_spacing=0),
    ]
    rec = {"id": f"fixture{i:04d}", "length": len(els), "group": 0, "format": i % 2,
           "canvas_width": W, "canvas_height": H, "category": 0, "title": title,
           "suitability": [0], "keywords": ["fixture"], "industries": [0], "cluster_index": i,
           "preview": Image.new("RGB", (8, 8))}
    for k in ("type", "left", "top", "width", "height", "angle", "opacity", "color", "image", "text",
              "font", "font_size", "text_align", "font_bold", "font_italic", "text_color",
              "text_line", "capitalize", "line_height", "letter_spacing"):
        rec[k] = [e[k] for e in els]
    return rec


def write(out_dir: str, n: int = 12, seed: int = 0, splits=("test", "validation")) -> str:
    """Write <out_dir>/data/<split>-00000-of-00001.parquet for each split. Previews are rendered
    with our own renderer, so the render-fidelity check sees (near) zero difference."""
    from datasets import Dataset
    from verifix.domains.graphic_design import crello_to_doc
    from verifix.core.render import render
    feats = features()
    Path(out_dir, "data").mkdir(parents=True, exist_ok=True)
    for si, split in enumerate(splits):
        rnd = random.Random(seed + 1000 * si)
        recs = []
        for i in range(n):
            r = record(i, rnd)
            doc = crello_to_doc(r, feats, i, split)
            r["preview"] = render(doc)
            recs.append(r)
        Dataset.from_list(recs, features=feats).to_parquet(
            os.path.join(out_dir, "data", f"{split}-00000-of-00001.parquet"))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crello_fixture")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    print("wrote", write(a.out, a.n))
