"""
Download only the Crello splits the experiments use (test + validation, about 3 GB) instead of the
whole dataset (about 18 GB with the training split).

Usage:
  python -m scripts.download_crello --out crello_v5            # then pass --crello_dir crello_v5
  python -m scripts.download_crello --out crello_v5 --splits test validation train
"""
import argparse

DEFAULT_REVISION = "5.1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crello_v5")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="cyberagent/crello", repo_type="dataset",
                             revision=args.revision, local_dir=args.out,
                             allow_patterns=[f"data/{s}-*" for s in args.splits] + ["README.md", "LICENSE"])
    print(f"downloaded Crello {args.revision} ({', '.join(args.splits)}) to {path}")
    print(f"use it with:  --crello_dir {args.out}   (or export VERIFIX_CRELLO_DIR={args.out})")


if __name__ == "__main__":
    main()
