"""
Download .ttf files for the font families the benchmark uses, from the Google Fonts GitHub repo
(raw.githubusercontent.com/google/fonts). Without real font files every family renders as the same
fallback face, so style perturbations become invisible and certification drops them.

Families come from (first that is given):
  --clusters data/font_clusters.json   (written by scripts/generate_data.py)
  --families "Roboto" "Playfair Display" ...
  otherwise the offline fallback clusters in perturb/fonts.py

Usage:
  python -m scripts.fetch_fonts --clusters data/font_clusters.json --out fonts
Families that are not on Google Fonts (commercial faces in Crello) are reported as missing; supply
those .ttf files yourself or drop them from the clusters.
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/google/fonts/main"
LICENSES = ("ofl", "apache", "ufl")
AXES = ("[wght]", "[wdth,wght]", "[opsz,wght]", "[opsz,wdth,wght]", "[ital,wght]", "[slnt,wght]", "[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght]")


def candidates(family: str):
    stem = family.replace(" ", "")
    d = family.lower().replace(" ", "")
    names = [f"{stem}{a}.ttf" for a in AXES] + [f"{stem}-Regular.ttf", f"{stem}.ttf"]
    for lic in LICENSES:
        for n in names:
            yield f"{BASE}/{lic}/{d}/{urllib.parse.quote(n)}"


def fetch(family: str, out_dir: str, timeout: int = 20) -> bool:
    dest = os.path.join(out_dir, family.replace(" ", "") + ".ttf")
    if os.path.exists(dest):
        return True
    for url in candidates(family):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            if data[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
                with open(dest, "wb") as f:
                    f.write(data)
                return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--out", default="fonts")
    args = ap.parse_args()
    if args.clusters:
        families = json.load(open(args.clusters))["fonts"]
    elif args.families:
        families = args.families
    else:
        from verifix.perturb.fonts import FALLBACK_CLUSTERS
        families = [f for fs in FALLBACK_CLUSTERS.values() for f in fs]
    os.makedirs(args.out, exist_ok=True)
    ok, missing = [], []
    for fam in families:
        (ok if fetch(fam, args.out) else missing).append(fam)
    print(f"fetched {len(ok)}/{len(families)} families into {args.out}/")
    if missing:
        print("missing (not on Google Fonts or unusual file name):", ", ".join(missing))


if __name__ == "__main__":
    main()
