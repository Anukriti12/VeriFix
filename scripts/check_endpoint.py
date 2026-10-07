"""Validate the model endpoint (text + VISION) before a real run.
Usage: python -m scripts.check_endpoint   (set VERIFIX_BASE_URL/MODEL/API_KEY first)"""
import os, sys
from PIL import Image, ImageDraw
from verifix.agents.llm_client import make_client, chat

def main():
    model = os.environ.get("VERIFIX_MODEL")
    print("base_url =", os.environ.get("VERIFIX_BASE_URL"), "| model =", model)
    client = make_client()
    print("[1/2] text ...")
    print("  ->", repr(chat(client, model, "Answer in one word.", "Reply: ok", max_tokens=10).strip()))
    print("[2/2] vision ...")
    img = Image.new("RGB", (160, 160), (220, 40, 40)); ImageDraw.Draw(img).rectangle([0,120,160,160], fill=(40,60,220))
    r = chat(client, model, "Answer in one word.", "What color fills MOST of this image?", images=[img], max_tokens=10)
    print("  ->", repr(r.strip()))
    print("VISION OK" if "red" in r.lower() else "WARNING: not vision-capable or ignoring images")

if __name__ == "__main__":
    main()
