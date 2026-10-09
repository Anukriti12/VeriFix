"""
Offline smoke test of the WHOLE graphic-design pipeline: no GPU, no network, no Crello download.

It writes a tiny dataset in the Crello v5 schema (tests/crello_fixture.py), then runs every step
of the real workflow through the same command-line scripts you will use, with a FAKE model
(VERIFIX_STUB=1). If this finishes with "SMOKE TEST PASSED", your installation works and the
remaining steps are getting real Crello data, fonts, and a real model endpoint.

Usage:
  python -m scripts.smoke_test                # writes everything under data/_smoke/
The numbers it prints are meaningless (the fake model is scripted).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args, env):
    cmd = [sys.executable, "-m"] + args
    print("\n$ python -m " + " ".join(args), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)}")


def main():
    work = ROOT / "data" / "_smoke"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    env = dict(os.environ, VERIFIX_STUB="1", PYTHONPATH=str(ROOT))
    crello = work / "crello"
    common = ["--crello_dir", str(crello)]
    W = lambda name: str(work / name)

    run(["tests.crello_fixture", "--out", str(crello), "--n", "10"], env)
    run(["scripts.inspect_crello", "--split", "validation", "--n", "10", "--pairs", "3",
         "--out", W("inspect")] + common, env)
    run(["scripts.build_index", "--split", "validation", "--n", "5", "--build_clusters",
         "--clusters", W("font_clusters.json"), "--asset_dir", W("assets"),
         "--out", W("index.json")] + common, env)
    run(["scripts.build_index", "--split", "validation", "--n", "5", "--synthetic",
         "--clusters", W("font_clusters.json"), "--asset_dir", W("assets"),
         "--out", W("index_syn.json")] + common, env)
    run(["scripts.generate_data", "--split", "test", "--n", "6", "--clusters", W("font_clusters.json"),
         "--asset_dir", W("assets"), "--out", W("eval.json")] + common, env)
    run(["scripts.generate_data", "--split", "validation", "--skip", "5", "--n", "4",
         "--clusters", W("font_clusters.json"), "--asset_dir", W("assets"),
         "--out", W("rank_train.json")] + common, env)
    run(["scripts.check_endpoint"], env)
    run(["scripts.rekey_index", "--index", W("index.json"), "--out", W("index_rekeyed.json"),
         "--max_per_class", "3"], env)
    run(["scripts.collect_ranker_logs", "--data", W("rank_train.json"), "--index", W("index_rekeyed.json"),
         "--log", W("ranker_logs.jsonl"), "--out_index", W("index_measured.json"), "--pool", "3"], env)
    run(["scripts.train_ranker", "--log", W("ranker_logs.jsonl"), "--out", W("ranker.json"),
         "--rounds", "5"], env)
    run(["scripts.run_experiments", "--data", W("eval.json"), "--index", W("index_measured.json"),
         "--index_syn", W("index_syn.json"), "--learned_model", W("ranker.json"),
         "--conditions", "B0", "B1", "B2", "B3", "B4", "M1", "M2", "M3", "M4", "ceiling", "oracle",
         "--out", W("results")], env)
    run(["scripts.run_experiments", "--data", W("eval.json"), "--index", W("index_measured.json"),
         "--conditions", "M3", "--retriever", "oracle", "--tag", "oracle_ranker",
         "--out", W("results")], env)
    run(["scripts.coverage", "--data", W("eval.json"), "--index", W("index_rekeyed.json")], env)
    run(["scripts.aggregate_results", "--results", W("results")], env)
    print("\nSMOKE TEST PASSED (fake model: the numbers above mean nothing).")


if __name__ == "__main__":
    main()
