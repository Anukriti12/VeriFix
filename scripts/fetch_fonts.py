"""
Download font files for the families the benchmark uses, from the Google Fonts GitHub repository
(raw.githubusercontent.com/google/fonts). Without the real files, every family renders as the same
fallback face: style perturbations become invisible and the Crello loader drops the design.

For each family we read the family's METADATA.pb and download:
  * every variable file (Family[wght].ttf, Family-Italic[wght].ttf), or
  * the static Regular, Bold, Italic and BoldItalic files (closest weights when exact ones are absent).
Files keep their Google Fonts names; core/render.py indexes them by family and style.

Families come from (all that are given are merged):
  --families_file reports/fonts_needed.txt   (written by scripts/inspect_crello.py)
  --clusters data/font_clusters.json         (written by scripts/generate_data.py)
  --families "Roboto" "Playfair Display"
  nothing given -> the offline fallback families in perturb/fonts.py

Usage:
  python -m scripts.fetch_fonts --families_file reports/fonts_needed.txt --out fonts
Families that are not on Google Fonts (commercial faces) are listed as missing. Add those .ttf files
to fonts/ yourself if you have a license, or let the loader drop the designs that use them.
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.core.render import split_family_weight  # noqa: E402

BASE = "https://raw.githubusercontent.com/google/fonts/main"
LICENSES = ("ofl", "apache", "ufl")


def _get(url: str, timeout: int = 30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def metadata(family: str):
    """(license dir, family dir, [(filename, style, weight)]) or None."""
    d = re.sub(r"[^a-z0-9]", "", family.lower())
    for lic in LICENSES:
        raw = _get(f"{BASE}/{lic}/{d}/METADATA.pb")
        if not raw:
            continue
        text = raw.decode("utf-8", "replace")
        fonts = []
        for block in re.findall(r"fonts\s*\{(.*?)\}", text, flags=re.S):
            fn = re.search(r'filename:\s*"([^"]+)"', block)
            st = re.search(r'style:\s*"([^"]+)"', block)
            wt = re.search(r"weight:\s*(\d+)", block)
            if fn:
                fonts.append((fn.group(1), st.group(1) if st else "normal",
                              int(wt.group(1)) if wt else 400))
        return lic, d, fonts
    return None


def choose(fonts):
    variable = [f for f in fonts if "[" in f[0]]
    if variable:
        return sorted({f[0] for f in variable})
    out = set()
    for style in ("normal", "italic"):
        cands = [f for f in fonts if f[1] == style]
        for target in (400, 700):
            if cands:
                out.add(min(cands, key=lambda f: abs(f[2] - target))[0])
    return sorted(out)


def fetch(family: str, out_dir: str) -> bool:
    base, _, _ = split_family_weight(family)
    for name in dict.fromkeys([family, base]):
        meta = metadata(name)
        if not meta:
            continue
        lic, d, fonts = meta
        got = 0
        stem = re.sub(r"[^A-Za-z0-9]", "", name.title() if name.islower() else name)
        for fn in choose(fonts):
            # save under the REQUESTED family name, so the renderer finds e.g. "Old Standard TT"
            # even when Google names its files "OldStandard-Regular.ttf"
            head = re.split(r"[-\[]", os.path.splitext(fn)[0], maxsplit=1)[0]
            suffix = fn[len(head):]          # keeps "-Bold.ttf", "[wght].ttf" or just ".ttf"
            dest = os.path.join(out_dir, stem + suffix)
            if os.path.exists(dest):
                got += 1
                continue
            data = _get(f"{BASE}/{lic}/{d}/{urllib.parse.quote(fn)}")
            if data and data[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
                with open(dest, "wb") as f:
                    f.write(data)
                got += 1
        if got:
            return True
    return False


def read_families(args):
    fams = []
    if args.families_file:
        for line in open(args.families_file, encoding="utf-8"):
            name = line.split("\t")[0].strip()
            if name and not name.startswith("#"):
                fams.append(name)
    if args.clusters:
        fams += json.load(open(args.clusters))["fonts"]
    if args.families:
        fams += args.families
    if not fams:
        from verifix.perturb.fonts import FALLBACK_CLUSTERS
        fams = [f for fs in FALLBACK_CLUSTERS.values() for f in fs]
    return list(dict.fromkeys(fams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families_file", default=None)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--out", default="fonts")
    args = ap.parse_args()
    families = read_families(args)
    os.makedirs(args.out, exist_ok=True)
    ok, missing = [], []
    for i, fam in enumerate(families, 1):
        (ok if fetch(fam, args.out) else missing).append(fam)
        if i % 25 == 0:
            print(f"  {i}/{len(families)} ...")
    print(f"fetched {len(ok)}/{len(families)} families into {args.out}/")
    if missing:
        Path(args.out, "MISSING.txt").write_text("\n".join(missing) + "\n")
        print(f"{len(missing)} not on Google Fonts (listed in {args.out}/MISSING.txt):",
              ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""))


if __name__ == "__main__":
    main()
