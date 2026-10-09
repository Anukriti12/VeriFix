"""
Offline tests for the deterministic core (no model endpoint needed).
Run: python tests/test_core.py   (or python -m pytest tests/ -q)

Style and font-swap visibility needs real font files: run `python -m scripts.fetch_fonts` once;
without them those checks are skipped, because every family renders as the same fallback face.
"""
import json
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FONTS = os.path.join(ROOT, "fonts")
HAVE_FONTS = os.path.isdir(FONTS) and len([f for f in os.listdir(FONTS) if f.endswith(".ttf")]) >= 10

from verifix.core.color import ciede2000_lab, contrast_ratio
from verifix.core.design import Document, structural_diff
from verifix.core.metrics import collateral, exact_rate, quality, target_fraction
from verifix.core.objective import design_penalty
from verifix.core.tools import TOOLS, TOOLS_EXCLUDED, execute_actions, execute_tool
from verifix.domains.graphic_design import synthetic_designs
from verifix.perturb.coverage import coverage_rate, greedy_cover
from verifix.perturb.fonts import FontClusters
from verifix.perturb.generate import DataPoint, generate, make_datapoint
from verifix.perturb.operators import (ALL_CLASSES, BINS, REPAIR_REQUEST, apply_spec,
                                       restore_recipe)
from verifix.agents.verifier import verify, verify_against_truth


def _designs(n=6):
    return synthetic_designs(n)


# ------------------------------------------------------------------ color
def test_color_math():
    # Sharma, Wu & Dalal (2005), test pair 1: dE00 = 2.0425
    assert abs(ciede2000_lab((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)) - 2.0425) < 1e-3
    assert abs(contrast_ratio("#000000", "#ffffff") - 21.0) < 1e-6
    print("ok: CIEDE2000 matches the published test pair; WCAG black/white = 21:1")


# ------------------------------------------------------------------ tools
def test_tool_set_matches_pi():
    assert len(TOOLS) == 21 and "mood_transfer" in TOOLS_EXCLUDED
    assert {t.category for t in TOOLS.values()} == {"T", "I", "L", "D"}
    print("ok: 21 of P&I's 22 tools implemented; mood_transfer excluded and documented")


def test_every_tool_executes_and_rejects_bad_calls():
    X = _designs(1)[0]
    calls = {
        "recolor_text": ("el-1", {"color": "#123456"}), "change_font": ("el-1", {"family": "Lora"}),
        "resize_text": ("el-1", {"px": 40}), "align_text": ("el-1", {"alignment": "right"}),
        "reflow_text": ("el-1", {"bbox": [10, 10, 300, 80]}),
        "recolor_image": ("el-0", {"hue": 90, "sat": 1.5}), "replace_image": ("el-0", {"asset": "asset-1"}),
        "crop_image": ("el-0", {"bbox": [0.1, 0.1, 0.9, 0.9]}), "apply_filter": ("el-0", {"filter": "sepia"}),
        "reposition": ("el-2", {"x": 10, "y": 20}), "resize_element": ("el-0", {"w": 100, "h": 50}),
        "rotate": ("el-1", {"deg": 15}), "reorder_layer": ("el-1", {"z": 0}),
        "duplicate_elem": ("el-2", {}), "remove_element": ("el-3", {}),
        "add_shape": ("canvas", {"shape": "ellipse", "bbox": [5, 5, 50, 50], "style": {"fill": "#ff0000"}}),
        "swap_palette": ("canvas", {"mapping": {X.background: "#336699"}}),
        "change_bg": ("canvas", {"color": "#abcdef"}), "apply_effect": ("el-1", {"effect": "shadow"}),
        "adj_contrast": ("canvas", {"delta": 0.3}), "adj_saturation": ("canvas", {"delta": -0.3}),
    }
    assert set(calls) == set(TOOLS)
    for name, (tgt, params) in calls.items():
        out, why = execute_tool(X, {"action": name, "target": tgt, "params": params})
        assert out is not None, (name, why)
        assert structural_diff(out, X), f"{name} changed nothing"
    bad = [({"action": "resize_text", "target": "el-0", "params": {"px": 20}}, "wrong_element_type"),
           ({"action": "resize_text", "target": "nope", "params": {"px": 20}}, "missing_target"),
           ({"action": "resize_text", "target": "el-1", "params": {"px": 9999}}, "out_of_range"),
           ({"action": "mood_transfer", "target": "canvas", "params": {}}, "tool_not_implemented")]
    for call, reason in bad:
        out, why = execute_tool(X, call)
        assert out is None and why == reason, (call, why)
    print("ok: all 21 tools execute and change the design; infeasible calls are rejected with a reason")


# ------------------------------------------------------------------ perturbations
def test_every_class_is_invertible_and_scored():
    for cls in ALL_CLASSES:
        sevs = list(BINS[cls]) if cls in BINS else (["significant"] if cls == "style" else ["-"])
        for sev in sevs:
            n_ok = 0
            for i, X in enumerate(_designs(4)):
                dp = make_datapoint(X, [(cls, sev)], seed=i, fonts_dir=FONTS)
                if dp is None:
                    continue
                c = dp.certificate
                assert c["invertible"] and c["oracle_recipe_exact"], (cls, sev, c)
                assert abs(quality(dp.Y, dp.targets)) < 1e-9, (cls, quality(dp.Y, dp.targets))
                assert quality(execute_actions(dp.Y, dp.inverse), dp.targets) == 1.0
                assert dp.query != REPAIR_REQUEST and "Keep everything else" in dp.query, dp.query
                g = make_datapoint(X, [(cls, sev)], seed=i, query_mode="generic", fonts_dir=FONTS)
                assert g is None or g.query == REPAIR_REQUEST
                n_ok += 1
            assert n_ok > 0, f"{cls}:{sev} never applicable on synthetic designs"
    print(f"ok: all {len(ALL_CLASSES)} classes invert exactly; Q(Y)=0, Q(inverse)=1; request is defect-agnostic")


def test_calibrated_bins():
    for sev, (lo, hi) in BINS["readability_size"].items():
        for i, X in enumerate(_designs(4)):
            *_, audit = apply_spec(X, [("readability_size", sev)], random.Random(i))
            assert lo - 1e-6 <= audit[0]["size_ratio"] <= hi + 1e-6
    for cls, key in (("readability_contrast", "delta_cr"), ("palette", "delta_e")):
        for sev, (lo, hi) in BINS[cls].items():
            got = 0
            for i, X in enumerate(_designs(4)):
                out = apply_spec(X, [(cls, sev)], random.Random(i))
                if out is None:
                    continue
                assert lo - 1e-6 <= out[3][0][key] <= hi + 1e-6, (cls, sev, out[3])
                got += 1
            assert got > 0, (cls, sev)
    print("ok: size ratio, contrast drop, and CIEDE2000 shift all fall inside P&I's calibrated bins")


def test_style_visibility_needs_fonts():
    if not HAVE_FONTS:
        print("skip: no font files (run scripts/fetch_fonts.py) -> style swaps cannot be visible")
        return
    dp = make_datapoint(_designs(1)[0], [("style", "significant")], seed=0, fonts_dir=FONTS)
    assert dp.certificate["visible"] and dp.certificate["keep"]
    print("ok: with real font files a style swap is visible and certified")


def test_font_clusters_recover_cooccurrence_groups():
    groups = [["A1", "A2", "A3", "A4"], ["B1", "B2", "B3", "B4"], ["C1", "C2", "C3", "C4"]]
    docs = []
    rnd = random.Random(0)
    for g in groups:
        for _ in range(10):
            fs = rnd.sample(g, 2)
            docs.append(Document(100, 100, "#ffffff", [
                {"id": f"t{j}", "type": "text", "font": f, "text": "x"} for j, f in enumerate(fs)]))
    fc = FontClusters.from_designs(docs, k=3, min_fonts=4)
    for g in groups:
        assert len({fc.cluster_of(f) for f in g}) == 1, g
    new, dist = fc.swap("A1", random.Random(1))
    assert fc.cluster_of(new) != fc.cluster_of("A1") and dist > 0
    print("ok: co-occurrence clustering recovers the font groups; a swap leaves the cluster")


# ------------------------------------------------------------------ metrics and objective
def test_recovery_fraction():
    X = _designs(1)[0]
    Y = execute_actions(X, [{"action": "resize_text", "target": "el-1", "params": {"px": 32}}])
    tg = [{"target": "el-1", "prop": "font_size", "x_value": 64.0, "y_value": 32.0}]
    half = execute_actions(Y, [{"action": "resize_text", "target": "el-1", "params": {"px": 48}}])
    over = execute_actions(Y, [{"action": "resize_text", "target": "el-1", "params": {"px": 100}}])
    assert quality(Y, tg) == 0.0 and quality(X, tg) == 1.0
    assert abs(quality(half, tg) - 0.5) < 1e-9 and quality(over, tg) == 0.0
    assert exact_rate(half, tg) == 0.0 and exact_rate(X, tg) == 1.0
    cat = {"target": "el-1", "prop": "font", "x_value": "Lora", "y_value": "Oswald"}
    assert target_fraction(X, cat) in (0.0, 1.0)
    print("ok: recovery fraction gives partial credit (half undo = 0.5); categorical is 0/1")


def test_objective_terms():
    base = Document(600, 400, "#ffffff", [
        {"id": "t", "type": "text", "text": "Hi", "font_size": 40, "color": "#222222",
         "left": 40, "top": 40, "width": 200, "height": 60}])
    assert design_penalty(base)[0] == 0.0
    low = execute_actions(base, [{"action": "recolor_text", "target": "t", "params": {"color": "#eeeeee"}}])
    assert design_penalty(low)[1]["contrast"] > 0
    covered = execute_actions(base, [{"action": "add_shape", "target": "canvas", "params": {
        "shape": "rect", "bbox": [30, 30, 120, 80], "style": {"fill": "#000000"}}}])
    assert design_penalty(covered)[1]["occlusion"] > 0
    under = execute_actions(covered, [{"action": "reorder_layer", "target": "shape-0", "params": {"z": 0}}])
    assert design_penalty(under)[1]["occlusion"] == 0
    tilted = execute_actions(base, [{"action": "rotate", "target": "t", "params": {"deg": 20}}])
    assert design_penalty(tilted)[1]["tilt"] > 0
    print("ok: Phi sees contrast, occlusion (and reordering under), and tilt; 0 on a clean design")


def test_verifier_is_a_harm_filter():
    base = Document(600, 400, "#ffffff", [
        {"id": "t", "type": "text", "text": "Hi", "font_size": 40, "color": "#222222",
         "left": 40, "top": 40, "width": 200, "height": 60}])
    props = [{"action": "change_font", "target": "t", "params": {"family": "Lora"}},          # neutral
             {"action": "recolor_text", "target": "t", "params": {"color": "#f5f5f5"}},    # harmful
             {"action": "resize_text", "target": "ghost", "params": {"px": 30}},           # infeasible
             {"action": "resize_text", "target": "t", "params": {"px": 40}}]               # no change
    kept, log = verify(base, props)
    assert [k["action"] for k in kept] == ["change_font"], log
    assert [r["reason"] for r in log] == ["accepted", "worsens_objective", "missing_target", "no_change"]
    X = base
    Y = execute_actions(X, [{"action": "resize_text", "target": "t", "params": {"px": 30}}])
    tg = [{"target": "t", "prop": "font_size", "x_value": 40.0, "y_value": 30.0}]
    kept, _ = verify_against_truth(Y, [{"action": "resize_text", "target": "t", "params": {"px": 20}},
                                        {"action": "resize_text", "target": "t", "params": {"px": 40}}], X, tg)
    assert [k["params"]["px"] for k in kept] == [40]
    print("ok: verifier keeps neutral edits, drops harmful/infeasible/no-op; ceiling uses the answer key")


def test_collateral_counts_unperturbed_changes():
    X = _designs(1)[0]
    dp = make_datapoint(X, [("rotate", "-")], seed=3, fonts_dir=FONTS)
    fixed = execute_actions(dp.Y, dp.inverse)
    assert collateral(fixed, X, dp.targets) == 0
    extra = execute_actions(fixed, [{"action": "align_text", "target": "el-2", "params": {"alignment": "left"}}])
    assert collateral(extra, X, dp.targets) == 1
    print("ok: collateral counts changes outside the perturbed slots")


# ------------------------------------------------------------------ data, coverage, assets
def test_coverage_greedy_and_generate():
    cands = [["a"], ["a", "b"], ["a", "b"], ["c", "d"], ["b", "c"]]
    pick = greedy_cover(cands, budget=2)
    assert set(pick) == {1, 3}
    one_per = greedy_cover(cands, budget=3, groups=["g1", "g1", "g2", "g3", "g3"])
    assert len({["g1", "g1", "g2", "g3", "g3"][i] for i in one_per}) == 3
    dps = generate(_designs(6), k_range=(1, 3), sampler="coverage", pool_per_design=3, fonts_dir=FONTS)
    assert dps and all(d.certificate["keep"] for d in dps)
    assert 0.0 <= coverage_rate([d.pclasses for d in dps], ALL_CLASSES) <= 1.0
    print(f"ok: greedy max-coverage picks complementary sets; generated {len(dps)} certified datapoints")


def test_asset_roundtrip_keeps_images():
    X = _designs(1)[0]
    d = tempfile.mkdtemp()
    back = Document.from_dict(json.loads(json.dumps(X.to_dict(asset_dir=d))))
    assert set(back.assets) == set(X.assets)
    dp = make_datapoint(X, [("replace_image", "-")], seed=1, fonts_dir=FONTS)
    back_dp = DataPoint.from_dict(json.loads(json.dumps(dp.to_dict(asset_dir=d))))
    assert quality(execute_actions(back_dp.Y, back_dp.inverse), back_dp.targets) == 1.0
    print("ok: saved datapoints keep their image assets (replace_image still inverts after reload)")


# ------------------------------------------------------------------ retrieval
def test_index_and_rankers():
    from verifix.retrieval.build import build_index
    from verifix.retrieval.index import detect_defects
    from verifix.retrieval import sparse, outcome
    from verifix.retrieval.learned import LearnedRanker
    idx = build_index(_designs(2), synthetic=True)
    assert "readability_contrast" in idx.by_class and any("+" in k for k in idx.by_class)
    jr = {"layout": {"score": 1, "explanation": "ok"},
          "typography": {"score": 0.5, "explanation": "the subtitle text is too small"},
          "color": {"score": 0, "explanation": "weak contrast against the background"}}
    d = detect_defects(jr)
    assert "readability_size" in d and "readability_contrast" in d, d
    cands = idx.candidates(d)
    crit = "the subtitle text is too small; weak contrast against the background"
    for ranked in (sparse.rank(crit, cands, top_k=3, key_mode="both"),
                   outcome.rank(crit, cands, top_k=3, key_mode="both"),
                   LearnedRanker.load(None).rank(crit, cands, top_k=3, key_mode="both")):
        assert ranked
    print("ok: index mined with P&I classes; defect detection; sparse/outcome/learned rank")


def test_exemplar_key_modes_drop_element_ids():
    from verifix.retrieval.index import Exemplar, exemplar_key_text
    e = Exemplar("readability_size", "text too small", [{"action": "resize_text", "target": "el-7",
                                                         "params": {"px": 64}}])
    assert "el-7" not in exemplar_key_text(e, "actions")
    assert "resize_text" in exemplar_key_text(e, "both") and "too small" in exemplar_key_text(e, "both")
    print("ok: exemplar keys (critique/actions/both) carry no design-specific element ids")


def test_learned_ranker_trains():
    from verifix.retrieval.index import Exemplar
    from verifix.retrieval.learned import LearnedRanker, _HAVE_LGB, features, train_from_groups
    c = [Exemplar("readability_size", "text too small", [{"action": "resize_text", "target": "x",
                                                         "params": {"px": 64}}], utility=0.3),
         Exemplar("rotate", "element tilted", [{"action": "rotate", "target": "x",
                                                 "params": {"deg": -10}}], utility=0.0)]
    assert len(features("text too small", c[0], "both")) == 4
    if _HAVE_LGB:
        X = [features("text too small", e, "both") for e in c] * 10
        b = train_from_groups(X, [3, 0] * 10, [2] * 10, num_boost_round=20)
        assert len(LearnedRanker(booster=b).rank("text too small", c, top_k=2)) == 2
    print("ok: learned ranker features and training")


def test_coverage_diagnostic_and_rekey():
    from verifix.retrieval.build import build_index
    from verifix.retrieval.coverage import coverage, rekey_with_judge
    idx = build_index(_designs(2))
    cov = coverage(idx, [(["readability_contrast"], "weak contrast against its background")],
                   key_mode="critique", threshold=0.05)
    assert 0.0 <= cov["overall"] <= 1.0
    new = rekey_with_judge(idx, judge_fn=lambda e: "JUDGE: " + e.defect_class)
    assert next(iter(new.by_class.values()))[0].critique.startswith("JUDGE:")
    print("ok: coverage diagnostic and judge re-keying")


# ------------------------------------------------------------------ two-turn pipeline (stubs)
class _Counter:
    def __init__(self): self.calls = 0
    def add(self, a, b): self.calls += 1
    def as_dict(self): return {"calls": self.calls, "in_tok": 0, "out_tok": 0}


class StubJudge:
    fonts_dir = FONTS
    def __init__(self): self.counter, self.client, self.model = _Counter(), None, "stub"
    def score(self, doc, query=""):
        from verifix.agents.judge import finalize
        r = {d: {"score": 0.5, "explanation": "the subtitle text is too small"}
             for d in ("layout", "typography", "color")}
        return finalize(r)


class StubPlanner:
    """Turn 1: no edits. Turn 2: applies whatever the condition handed it via the 'fix' hook."""
    fonts_dir = FONTS
    def __init__(self): self.counter, self.client, self.model = _Counter(), None, "stub"
    def plan(self, doc, query, feedback=None, extra_block=None):
        self.counter.add(0, 0)
        self.seen = (feedback, extra_block)
        return list(getattr(self, "fix", []))


class StubCompiler:
    def __init__(self, proposals): self.proposals = proposals
    def compile(self, doc, jr, query=""): return list(self.proposals)


def test_two_turn_pipeline():
    from verifix.loop.pipeline import Pipeline
    X = _designs(1)[0]
    dp = make_datapoint(X, [("readability_size", "moderate")], seed=2, fonts_dir=FONTS)
    planner, judge = StubPlanner(), StubJudge()
    t1 = Pipeline(planner, judge).run_turn1(dp.Y, dp.query)
    r0 = Pipeline(planner, judge).run(dp.Y, dp.query, "B0", dp.targets, X, turn1=t1)
    assert r0["delta_q"] == 0.0 and r0["q_turn1"] == 0.0
    good = restore_recipe(t1["doc"], dp.targets)
    bg = X.background                                     # text recolored to the background: harmful anywhere
    bad = [{"action": "recolor_text", "target": "el-2", "params": {"color": bg}}]
    pipe = Pipeline(planner, judge, StubCompiler(bad + good))
    planner.fix = good
    r2 = pipe.run(dp.Y, dp.query, "M2", dp.targets, X, turn1=t1)
    assert r2["verifier"]["accepted"] == len(good) and r2["verifier"].get("worsens_objective") == 1
    assert "CHECKED EDITS" in planner.seen[1] and "recolor_text" not in planner.seen[1]
    assert abs(r2["q_turn2"] - 1.0) < 1e-9 and abs(r2["delta_q"] - 1.0) < 1e-9
    ro = pipe.run(dp.Y, dp.query, "oracle", dp.targets, X, turn1=t1)
    assert "EXACT FIX" in planner.seen[1]
    rb1 = pipe.run(dp.Y, dp.query, "B1", dp.targets, X, turn1=t1)
    assert "too small" in planner.seen[0]
    planner.fix = []
    rm = Pipeline(planner, judge, StubCompiler(good), max_passes=3).run_multi(
        dp.Y, dp.query, "M2", dp.targets, X, turn1=t1)
    assert rm["n_turns"] == 3 and len(rm["history"]) == 3
    print("ok: shared turn 1; B0 dQ=0; M2 filters the harmful edit and shows only checked edits; "
          "oracle/B1 inputs; multi-pass runs to the budget")


def test_aggregate_reports_paired_gain():
    from verifix.eval.aggregate import report
    d = tempfile.mkdtemp()

    def row(cid, q1, q2, cond):
        return {"condition": cond, "case_id": cid, "pclasses": ["rotate"], "severities": ["-"],
                "q_turn1": q1, "q_turn2": q2, "delta_q": q2 - q1, "exact_turn2": 0.0,
                "judge_turn2": 0.5, "collateral": 0, "phi_visible": True,
                "cost": {"calls": 3}, "verifier": {"proposed": 2, "accepted": 1, "worsens_objective": 1}}
    json.dump([row(i, 0.2, 0.6, "B2") for i in range(5)], open(os.path.join(d, "results_B2.json"), "w"))
    json.dump([row(i, 0.2, 0.8, "M2") for i in range(5)], open(os.path.join(d, "results_M2.json"), "w"))
    json.dump([row(i, 0.2, 1.0, "oracle") for i in range(5)], open(os.path.join(d, "results_oracle.json"), "w"))
    report(d)
    print("ok: aggregate prints dQ with bootstrap CI, go/no-go, OracleHeadroom, verifier stats")


# ----------------------------------------------------------------------------- Crello loader
def test_parse_color_formats():
    from verifix.core.color import parse_color
    assert parse_color("#FFF") == ("#ffffff", 1.0)
    assert parse_color("rgba(255, 0, 0, 0.5)") == ("#ff0000", 0.5)
    assert parse_color("rgb(10%, 20%, 30%)")[0] == "#1a334d"
    assert parse_color([0, 128, 255]) == ("#0080ff", 1.0)
    assert parse_color([0.0, 0.5, 1.0])[0] == "#0080ff"
    assert parse_color("#11223380")[1] == 128 / 255
    assert parse_color("not a color") is None and parse_color(None) is None
    print("ok: dataset color strings parse (hex, rgb(), rgba(), lists)")


def test_crello_loader_on_v5_schema():
    """The real loader code path (datasets + parquet + class labels) on a fixture in the v5 schema."""
    import tempfile
    from tests.crello_fixture import write
    from verifix.domains.graphic_design import GraphicDesignDomain, _with_line_breaks
    d = tempfile.mkdtemp()
    write(d, n=4)
    # without font files every design would be dropped as "missing font"; fetch_fonts fixes that
    dom = GraphicDesignDomain(data_dir=d, allow_synthetic=False, require_fonts=HAVE_FONTS)
    docs = dom.load(3, split="test")
    assert len(docs) == 3 and dom.last_stats["kept"] == 3
    x = docs[0]
    assert x.width in (800, 1080) and x.background.startswith("#")
    texts = [e for e in x.elements if e["type"] == "text"]
    assert len(texts) == 2 and texts[0]["font_weight"] == "bold" and "\n" in texts[0]["text"]
    assert all(e["left"] > 1 and e["width"] > 1 for e in x.elements), "geometry must be pixels"
    tint = [e for e in x.elements if e.get("tint")]
    assert tint and tint[0]["fill"].startswith("#"), "single-color SVG becomes a tinted shape"
    assert not any(e.get("type") == "image" and e.get("left") == 0 and e.get("width") == x.width
                   for e in x.elements), "flat ColoredBackground becomes the canvas color"
    assert x.metadata["render_diff"] < 0.01
    skipped = dom.load(2, split="test", skip=1)
    assert skipped[0].metadata["crello_id"] == docs[1].metadata["crello_id"], "skip gives disjoint subsets"
    assert _with_line_breaks("Hello big world", [0] * 6 + [1] * 9) == "Hello\nbig world"
    print("ok: Crello v5 loader: pixels, degrees, text_color, bold, line breaks, tint, background, skip")


def test_crello_v4_geometry_is_rescaled():
    from verifix.domains.graphic_design import crello_to_doc
    rec = {"canvas_width": 1000, "canvas_height": 500, "type": ["TextElement"], "left": [0.1],
           "top": [0.2], "width": [0.5], "height": [0.1], "angle": [0.5], "text": ["Hi"],
           "font": ["Roboto"], "font_size": [20.0]}
    e = crello_to_doc(rec, None, 0, "test", revision="4.0.0").elements[0]
    assert (e["left"], e["top"], e["width"]) == (100, 100, 500) and abs(e["angle"] - 28.6479) < 1e-3
    print("ok: v4 normalized geometry and radians are converted")


def test_font_names_and_styles():
    from verifix.core.render import font_available, split_family_weight
    assert split_family_weight("Montserrat Bold Italic") == ("Montserrat", "700", True)
    assert split_family_weight("Open Sans") == ("Open Sans", None, False)
    assert split_family_weight("Roboto Semi Bold")[1] == "600"
    if font_available("Roboto"):
        assert font_available("Roboto Bold")
    print("ok: font names split into family, weight, italic")


def test_phi_ignores_transparent_parts_of_images():
    from PIL import Image
    from verifix.core.design import Document
    from verifix.core.objective import design_penalty, _bg_behind
    ring = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(100):
        for y in list(range(5)) + list(range(95, 100)):
            ring.putpixel((x, y), (0, 0, 0, 255))
    els = [{"id": "t", "type": "text", "text": "Hi", "color": "#ffffff", "font_size": 40,
            "left": 30, "top": 30, "width": 40, "height": 40},
           {"id": "frame", "type": "image", "asset_id": "frame", "left": 0, "top": 0,
            "width": 100, "height": 100}]
    d = Document(100, 100, "#111111", els, {"frame": ring})
    assert design_penalty(d)[1]["occlusion"] == 0.0, "a transparent frame does not occlude"
    d2 = Document(100, 100, "#111111", list(reversed(els)), {"frame": ring})
    assert _bg_behind(d2, 1) == "#111111", "text over a transparent region sees the canvas"
    print("ok: Phi samples image alpha for occlusion and background color")


def test_applicability_filter_and_mmr():
    from verifix.core.design import Document
    from verifix.retrieval.index import Exemplar
    from verifix.retrieval.select import applicable, mmr
    only_text = Document(100, 100, "#fff", [{"id": "a", "type": "text", "text": "x"}])
    crop = Exemplar("crop_image", "cropped", [{"action": "crop_image", "target": "e", "params": {"bbox": [0, 0, 1, 1]}}])
    font = Exemplar("style", "font", [{"action": "change_font", "target": "e", "params": {"family": "Lora"}}])
    bg = Exemplar("change_bg", "bg", [{"action": "change_bg", "target": "canvas", "params": {"color": "#fff"}}])
    assert not applicable(crop, only_text) and applicable(font, only_text) and applicable(bg, only_text)
    font2 = Exemplar("style", "font 2", [{"action": "change_font", "target": "e", "params": {"family": "Oswald"}}])
    picked = mmr([(1.0, font), (0.95, font2), (0.6, bg)], k=2, lam=0.5)
    assert [e.defect_class for _, e in picked] == ["style", "change_bg"], "MMR prefers a different fix"
    print("ok: applicability filter drops impossible fixes; MMR diversifies the top-k")


def test_model_output_parsing():
    from verifix.agents.judge import _parse, finalize
    from verifix.agents.planner import parse_calls
    calls = parse_calls('<think>maybe [x]</think> Here: ```json\n[{"action": "rotate", "target": "1", '
                        '"params": {"deg": -4}}]\n``` done')
    assert calls == [{"action": "rotate", "target": "1", "params": {"deg": -4}}]
    r = finalize(_parse('Sure. {"layout": {"score": "0.5", "explanation": "a"}, '
                        '"typography": {"score": 1}, "color": {"score": 0.4, "explanation": "b"}}'))
    assert r["parse_ok"] and r["layout"]["score"] == 0.5 and r["color"]["score"] == 0.5
    assert _parse("no json here")["parse_ok"] is False
    print("ok: tool calls and judge JSON are extracted from messy model replies")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} TESTS PASSED")
