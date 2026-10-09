"""
Inspect Crello BEFORE generating data: which fonts the designs use, how many are missing, what
the designs look like, and how close our renderer gets to Crello's own preview.

Writes to --out (default reports/inspect_<split>/):
  fonts_needed.txt     every font family used by text elements, with counts (input for fetch_fonts)
  stats.json           element types, canvas sizes, element counts, render-difference summary
  side_by_side/*.png   Crello preview (left) | our render (middle) | our marked render (right),
                       file name starts with the render difference so the worst ones sort last

Usage:
  python -m scripts.inspect_crello --split validation --n 200
  python -m scripts.fetch_fonts --families_file reports/inspect_validation/fonts_needed.txt --out fonts
  python -m scripts.inspect_crello --split validation --n 200     # again, now with fonts
Look at the side-by-side images with the highest differences and choose --max_render_diff.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.core.render import font_available, render, resolve_fonts_dir  # noqa: E402
from verifix.domains.cli import add_domain_args, domain_from_args  # noqa: E402
from verifix.domains.graphic_design import _etype, _decoder, _flatten, crello_to_doc, render_fidelity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200, help="records to inspect")
    ap.add_argument("--pairs", type=int, default=30, help="side-by-side images to save")
    ap.add_argument("--out", default=None)
    add_domain_args(ap)
    args = ap.parse_args()
    out = Path(args.out or f"reports/inspect_{args.split}")
    (out / "side_by_side").mkdir(parents=True, exist_ok=True)
    dom = domain_from_args(args)
    fonts_dir = resolve_fonts_dir(args.fonts_dir)

    font_count, type_count, sizes, n_elems = Counter(), Counter(), Counter(), []
    diffs_all, diffs_fonts_ok, designs_fonts_ok = [], [], 0
    saved = 0
    for k, (i, s, feats) in enumerate(dom.iter_records(args.split)):
        if k >= args.n:
            break
        decode = _decoder(feats)
        for t in s.get("type") or []:
            type_count[str(decode("type", t))] += 1
        doc = crello_to_doc(s, feats, i, args.split, args.revision)
        sizes[f"{doc.width}x{doc.height}"] += 1
        n_elems.append(len(doc.elements))
        fams = {e["font"] for e in doc.elements if e.get("type") == "text"}
        font_count.update(fams)
        ok = all(font_available(f, fonts_dir) for f in fams)
        designs_fonts_ok += int(ok)
        fd = render_fidelity(doc, s.get("preview"), fonts_dir)
        if fd is not None:
            diffs_all.append(fd)
            if ok:
                diffs_fonts_ok.append(fd)
        if saved < args.pairs and s.get("preview") is not None:
            h = 512
            prev = _flatten(s["preview"])
            ours, marked = render(doc, fonts_dir=fonts_dir), render(doc, marked=True, fonts_dir=fonts_dir)
            ims = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in (prev, ours, marked)]
            canvas = Image.new("RGB", (sum(im.width for im in ims) + 20, h), (255, 255, 255))
            x = 0
            for im in ims:
                canvas.paste(im, (x, 0)); x += im.width + 10
            tag = f"{fd:.3f}" if fd is not None else "na"
            canvas.save(out / "side_by_side" / f"{tag}_{s.get('id', i)}{'' if ok else '_MISSINGFONT'}.png")
            saved += 1
    inspected = len(n_elems)
    missing = [(f, c) for f, c in font_count.most_common() if not font_available(f, fonts_dir)]
    with open(out / "fonts_needed.txt", "w", encoding="utf-8") as f:
        f.write("# family\tdesigns using it\tavailable\n")
        for fam, c in font_count.most_common():
            f.write(f"{fam}\t{c}\t{'yes' if font_available(fam, fonts_dir) else 'no'}\n")

    def summ(xs):
        if not xs:
            return {}
        a = np.asarray(xs)
        return {"n": len(a), "mean": round(float(a.mean()), 4), "p50": round(float(np.median(a)), 4),
                "p75": round(float(np.percentile(a, 75)), 4), "p90": round(float(np.percentile(a, 90)), 4),
                "max": round(float(a.max()), 4)}

    stats = {"split": args.split, "inspected": inspected, "element_types": dict(type_count),
             "canvas_sizes": dict(sizes.most_common(10)),
             "elements_per_design": summ(n_elems),
             "font_families": len(font_count), "font_families_missing": len(missing),
             "designs_with_all_fonts": designs_fonts_ok,
             "render_diff_all": summ(diffs_all), "render_diff_fonts_ok": summ(diffs_fonts_ok)}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\n{len(missing)} of {len(font_count)} font families missing; "
          f"designs with every font available: {designs_fonts_ok}/{inspected}")
    print(f"fonts list -> {out/'fonts_needed.txt'}; images -> {out/'side_by_side'}/")


if __name__ == "__main__":
    main()
