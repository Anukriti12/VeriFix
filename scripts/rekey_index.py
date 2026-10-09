"""
Re-key the verified-fix index with the DEPLOYED judge (needs the model endpoint).

Index keys start as templated defect descriptions. The judge writes critiques in its own words,
so evaluation critiques may not match the keys (low coverage; see scripts/coverage.py). This
script renders each entry's stored degraded design, asks the judge to critique it, and uses that
critique as the entry's new key.

Usage:
  python -m scripts.rekey_index --index data/index.json --out data/index_rekeyed.json \
      --max_per_class 200
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.agents.judge import Judge, critique_text  # noqa: E402
from verifix.agents.llm_client import make_client  # noqa: E402
from verifix.core.design import Document  # noqa: E402
from verifix.retrieval.index import Exemplar, FixIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--out", default="data/index_rekeyed.json")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="re-key at most this many entries per class (the rest are dropped)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idx = FixIndex.load(args.index)
    y_path = Path(args.index).with_name(Path(args.index).stem + "_Y.jsonl")
    ys = open(y_path).read().splitlines()
    judge = Judge(client=make_client(), model=args.model)
    rnd = random.Random(args.seed)
    new = FixIndex()
    n_parse_fail = 0
    for cls, exs in idx.by_class.items():
        exs = list(exs)
        if args.max_per_class and len(exs) > args.max_per_class:
            exs = rnd.sample(exs, args.max_per_class)
        for e in tqdm(exs, desc=cls):
            ref = e.meta.get("y_ref")
            if ref is None:
                continue
            Y = Document.from_dict(json.loads(ys[ref]))
            jr = judge.score(Y)
            n_parse_fail += int(not jr.get("parse_ok", True))
            crit = critique_text(jr).strip() or e.critique
            new.add(Exemplar(defect_class=e.defect_class, critique=crit, actions=e.actions,
                             utility=e.utility, n_uses=e.n_uses, source=e.source + "+rekeyed",
                             meta=dict(e.meta, template_key=e.critique)))
    new.save(args.out)
    print(f"wrote {args.out}: {sum(new.stats().values())} entries; judge parse failures: {n_parse_fail}")


if __name__ == "__main__":
    main()
