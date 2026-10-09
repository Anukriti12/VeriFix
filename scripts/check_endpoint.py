"""
Check the model endpoint before a real run: text, vision, and the judge / planner JSON formats.

Set VERIFIX_BASE_URL, VERIFIX_MODEL and VERIFIX_API_KEY first (see env.example), then:
  python -m scripts.check_endpoint
All four checks should print OK. Fix any FAIL before running experiments.
"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifix.agents.judge import Judge  # noqa: E402
from verifix.agents.llm_client import chat, make_client  # noqa: E402
from verifix.agents.planner import Planner  # noqa: E402
from verifix.domains.graphic_design import synthetic_designs  # noqa: E402
from verifix.perturb.operators import REPAIR_REQUEST  # noqa: E402


def main():
    model = os.environ.get("VERIFIX_MODEL")
    print("base_url =", os.environ.get("VERIFIX_BASE_URL"), "| model =", model)
    if not model and os.environ.get("VERIFIX_STUB") != "1":
        raise SystemExit("VERIFIX_MODEL is not set (see env.example)")
    client = make_client()
    ok = True

    print("[1/4] text ...")
    r = chat(client, model, "Answer in one word.", "Reply with the word ok.", max_tokens=20)
    print("  ->", repr(r.strip()[:60]))

    print("[2/4] vision ...")
    img = Image.new("RGB", (160, 160), (220, 40, 40))
    ImageDraw.Draw(img).rectangle([0, 120, 160, 160], fill=(40, 60, 220))
    r = chat(client, model, "Answer in one word.", "What color fills MOST of this image?",
             images=[img], max_tokens=20)
    v = "red" in r.lower()
    ok &= v
    print("  ->", repr(r.strip()[:60]), "OK" if v else "FAIL: not vision-capable or ignoring images")

    design = synthetic_designs(1)[0]
    print("[3/4] judge JSON ...")
    jr = Judge(client=client, model=model).score(design)
    v = bool(jr.get("parse_ok"))
    ok &= v
    print("  -> scores", {d: jr[d]["score"] for d in ("layout", "typography", "color")},
          "OK" if v else "FAIL: judge reply was not valid JSON")

    print("[4/4] planner tool calls ...")
    calls = Planner(client=client, model=model).plan(
        design, REPAIR_REQUEST, extra_block="## CHECKED EDITS\n- rotate on 2: deg=5")
    v = isinstance(calls, list)
    ok &= v
    print("  ->", calls[:3], "OK" if v else "FAIL")
    print("\nENDPOINT OK" if ok else "\nENDPOINT NOT READY: fix the FAIL lines above")


if __name__ == "__main__":
    main()
