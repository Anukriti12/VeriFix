"""
domains/cli.py - Shared command-line options for loading graphic designs, used by every script
that reads Crello (inspect_crello, build_index, generate_data, smoke_test).
"""
from __future__ import annotations

import argparse

from .graphic_design import DEFAULT_REVISION, GraphicDesignDomain


def add_domain_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Crello loading")
    g.add_argument("--crello_dir", default=None,
                   help="local folder with Crello parquet files (default: download from the "
                        "Hugging Face Hub, or $VERIFIX_CRELLO_DIR)")
    g.add_argument("--revision", default=DEFAULT_REVISION, help="Crello dataset revision")
    g.add_argument("--streaming", action="store_true",
                   help="stream records instead of downloading the whole split first")
    g.add_argument("--data_seed", type=int, default=0, help="shuffle seed for picking designs")
    g.add_argument("--skip", type=int, default=0,
                   help="skip the first K designs that pass the filters (disjoint subsets)")
    g.add_argument("--max_elements", type=int, default=30)
    g.add_argument("--max_scan", type=int, default=None,
                   help="stop after scanning this many records (default 5 x requested)")
    g.add_argument("--max_render_diff", type=float, default=None,
                   help="drop designs whose render differs from Crello's preview by more than "
                        "this (0..1); set it after looking at scripts.inspect_crello output")
    g.add_argument("--no_require_fonts", action="store_true",
                   help="keep designs whose fonts are missing (style defects become invisible)")
    g.add_argument("--fonts_dir", default=None, help="default: $VERIFIX_FONTS_DIR or <repo>/fonts")
    g.add_argument("--allow_synthetic", action="store_true",
                   help="fall back to synthetic designs if Crello cannot be loaded (tests only)")


def domain_from_args(args) -> GraphicDesignDomain:
    return GraphicDesignDomain(revision=args.revision, data_dir=args.crello_dir,
                               streaming=args.streaming, seed=args.data_seed,
                               max_elements=args.max_elements,
                               require_fonts=not args.no_require_fonts, fonts_dir=args.fonts_dir,
                               max_render_diff=args.max_render_diff,
                               allow_synthetic=args.allow_synthetic)


def load_designs(args, n: int, split: str):
    dom = domain_from_args(args)
    return dom.load(n, split=split, skip=args.skip, max_scan=args.max_scan), dom
