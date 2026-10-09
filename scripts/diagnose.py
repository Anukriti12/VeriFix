"""
Diagnose a run WITHOUT the model: what turn 1 did, how good the judge was, and example cases.

Reads <results>/turn1_cache.json, <results>/results_<cond>.json and the data file the run used.
Writes <out>/diagnosis.txt and <out>/cases/*.png (X | Y | D1 side by side) + *.txt (what was
damaged, what turn 1 did, what the judge said).

Usage:
  python -m scripts.diagnose --results results_gonogo/ --data data/eval.json --out reports/diagnose_gonogo
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.agents.judge import JUDGE_DIMS  # noqa: E402
from verifix.core.design import structural_diff  # noqa: E402
from verifix.core.metrics import collateral, quality  # noqa: E402
from verifix.core.render import render  # noqa: E402
from verifix.core.tools import execute_tool  # noqa: E402
from verifix.perturb.generate import DataPoint  # noqa: E402
from verifix.retrieval.index import detect_defects  # noqa: E402

# which judge dimension each defect class affects (paper supplement, judge audit)
DIM_OF = {}
for c in ("style", "readability_size", "change_font", "resize_text", "align_text", "reflow_text",
          "apply_effect"):
    DIM_OF[c] = "typography"
for c in ("readability_contrast", "palette", "recolor_text", "recolor_image", "apply_filter",
          "change_bg", "adj_contrast", "adj_saturation"):
    DIM_OF[c] = "color"
for c in ("reposition", "resize_element", "rotate", "reorder_layer", "duplicate_elem", "add_shape",
          "crop_image", "replace_image"):
    DIM_OF[c] = "layout"


def apply_calls(doc, calls):
    """Apply calls one by one, counting how many execute."""
    ok, why = 0, Counter()
    for c in calls:
        out, reason = execute_tool(doc, c)
        if out is None:
            why[reason] += 1
            continue
        doc, ok = out, ok + 1
    return doc, ok, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_gonogo/")
    ap.add_argument("--data", default="data/eval.json")
    ap.add_argument("--out", default="reports/diagnose_gonogo")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cache = json.load(open(R / "turn1_cache.json"))
    raw = json.load(open(args.data))
    cases = {}
    for k, c in cache.items():
        dp = DataPoint.from_dict(raw[int(k)])
        if c.get("case_uid") and c["case_uid"] != dp.X.uid():
            raise SystemExit(f"{args.data} is not the file this run used (case {k} differs). "
                             "Use the data file from the time of the run.")
        cases[int(k)] = (dp, c)
    rows = {p.stem.replace("results_", ""): {r["case_id"]: r for r in json.load(open(p))}
            for p in sorted(R.glob("results_*.json"))}
    L = []
    say = L.append

    # ---------------- turn 1 ----------------
    n_calls, n_ok, coll1, q1, fails, acts, hit_target = [], [], [], [], Counter(), Counter(), []
    changed_props = Counter()
    for k, (dp, c) in cases.items():
        D1, ok, why = apply_calls(dp.Y, c["actions"])
        fails.update(why)
        n_calls.append(len(c["actions"])); n_ok.append(ok)
        coll1.append(collateral(D1, dp.X, dp.targets)); q1.append(quality(D1, dp.targets))
        acts.update(a.get("action") for a in c["actions"])
        tgt_ids = {t["target"] for t in dp.targets}
        hit_target.append(sum(1 for a in c["actions"] if a.get("target") in tgt_ids) / max(1, len(c["actions"])))
        perturbed = {(t["target"], t["prop"]) for t in dp.targets}
        for d in structural_diff(D1, dp.X):
            if (d["target"], d["prop"]) not in perturbed:
                changed_props[d["prop"]] += 1
    say(f"=== TURN 1 ({len(cases)} cases) ===")
    say(f"tool calls per case: mean {np.mean(n_calls):.1f} (executed {np.mean(n_ok):.1f}); "
        f"failed calls: {dict(fails)}")
    say(f"Q(D1) mean {np.mean(q1):.3f}; collateral (untouched properties changed) mean {np.mean(coll1):.1f}")
    say(f"share of turn-1 calls aimed at a damaged element: {np.mean(hit_target):.0%}")
    say("most used tools: " + ", ".join(f"{a}={n}" for a, n in acts.most_common(10)))
    say("properties changed that were NOT damaged: " +
        ", ".join(f"{p}={n}" for p, n in changed_props.most_common(12)))

    # ---------------- judge on D1 ----------------
    say("\n=== JUDGE ON D1 (does it notice the remaining damage?) ===")
    fp, n_def, parse_fail, detected, n_cls = Counter(), Counter(), 0, 0, 0
    for k, (dp, c) in cases.items():
        j = c["judge"]
        parse_fail += int(j.get("parse_ok") is False)
        damaged_dims = {DIM_OF.get(cl) for cl in dp.pclasses}
        for d in JUDGE_DIMS:
            if d in damaged_dims:
                n_def[d] += 1
                fp[d] += int(float(j.get(d, {}).get("score", 0)) >= 1.0)
        hits = set(detect_defects(j))
        detected += len(hits & set(dp.pclasses)); n_cls += len(dp.pclasses)
    for d in JUDGE_DIMS:
        if n_def[d]:
            say(f"{d:<11} damaged in {n_def[d]:>3} cases; judge still gave 1.0 (pass) in "
                f"{fp[d]:>3} ({fp[d] / n_def[d]:.0%})")
    say(f"judge replies that could not be parsed: {parse_fail}")
    say(f"keyword defect detector found {detected}/{n_cls} true classes ({detected / max(1, n_cls):.0%}) "
        f"in the critiques (this is what retrieval M3 would search with)")

    # ---------------- turn 2 per condition ----------------
    say("\n=== TURN 2 BY CONDITION ===")
    say(f"{'cond':<10}{'dQ':>8}{'improved':>10}{'worse':>8}{'edits':>8}{'collat D2-D1':>14}")
    for cond, rs in rows.items():
        dq = [r.get("delta_q", 0) for r in rs.values()]
        coll_delta = [r.get("collateral", 0) - coll1[list(cases).index(cid)] for cid, r in rs.items()
                      if cid in cases]
        say(f"{cond:<10}{np.mean(dq):>+8.3f}{sum(x > 1e-9 for x in dq):>10}{sum(x < -1e-9 for x in dq):>8}"
            f"{np.mean([r.get('n_edits_turn2', 0) for r in rs.values()]):>8.1f}{np.mean(coll_delta):>+14.1f}")

    # ---------------- examples ----------------
    say(f"\n=== EXAMPLES -> {out/'cases'} ===")
    order = sorted(cases, key=lambda k: q1[list(cases).index(k)])
    pick = order[: args.examples // 2] + order[-(args.examples - args.examples // 2):]
    for k in pick:
        dp, c = cases[k]
        D1, _, _ = apply_calls(dp.Y, c["actions"])
        ims = [render(d) for d in (dp.X, dp.Y, D1)]
        h = 420
        ims = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in ims]
        canvas = Image.new("RGB", (sum(i.width for i in ims) + 20, h + 22), "white")
        x = 0
        for lab, im in zip(("X (clean)", "Y (damaged)", "D1 (after turn 1)"), ims):
            canvas.paste(im, (x, 22)); ImageDraw.Draw(canvas).text((x + 4, 4), lab, fill="black")
            x += im.width + 10
        canvas.save(out / "cases" / f"case{k:03d}.png")
        txt = [f"case {k}: classes {list(zip(dp.pclasses, dp.severities))}",
               f"Q(D1)={quality(D1, dp.targets):.3f}  collateral(D1)={collateral(D1, dp.X, dp.targets)}",
               "\nDAMAGE (target, property: clean -> damaged):"]
        txt += [f"  {t['target']}.{t['prop']}: {t.get('x_value')} -> {t.get('y_value')}" for t in dp.targets]
        txt += ["\nTURN-1 TOOL CALLS:"] + [f"  {json.dumps(a)}" for a in c["actions"]]
        txt += ["\nJUDGE ON D1:"] + [f"  {d} {c['judge'].get(d, {}).get('score')}: "
                                     f"{c['judge'].get(d, {}).get('explanation', '')}" for d in JUDGE_DIMS]
        txt += ["\nTURN 2 dQ: " + ", ".join(f"{cond}={rs[k].get('delta_q', 0):+.3f}"
                                            for cond, rs in rows.items() if k in rs)]
        (out / "cases" / f"case{k:03d}.txt").write_text("\n".join(txt))
        say(f"  case {k:03d}: Q1={quality(D1, dp.targets):.2f}, classes {dp.pclasses}")
    (out / "diagnosis.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
