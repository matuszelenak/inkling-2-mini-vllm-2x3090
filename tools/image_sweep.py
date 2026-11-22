#!/usr/bin/env python3
"""Image-shape sweep: does the vision path survive every size, and at what cost?

Sends one four-quadrant PNG per shape (a distinct colour per quadrant) and
asks for the four colours back. Scores the answer, records the prompt tokens
the request cost (text plus image) and the per-GPU memory after it. The colours are
checkable without OCR, so a wrong encode or a silent downscale bug shows up
as a wrong answer, not just as a crash.

Usage: tools/image_sweep.py <label> [--url http://localhost:8000] [--shapes 64x64 512x512 ...]
Appends one row per shape to bench-results/images.csv unless --csv is given.
"""

import argparse
import base64
import csv
import json
import pathlib
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

COLOURS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 40)}
ORDER = ("red", "green", "blue", "yellow")  # top-left, top-right, bottom-left, bottom-right
SHAPES = ["64x64", "512x512", "1024x768", "768x1024", "2048x2048", "4096x1024", "4096x4096"]
FIELDS = ["label", "width", "height", "ok", "correct", "prompt_n", "gen_n", "wall_s", "gpu0_used_mib", "gpu1_used_mib", "error"]


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    half_w, half_h = width // 2, height // 2
    top = b"\x00" + bytes(COLOURS[ORDER[0]]) * half_w + bytes(COLOURS[ORDER[1]]) * (width - half_w)
    bottom = b"\x00" + bytes(COLOURS[ORDER[2]]) * half_w + bytes(COLOURS[ORDER[3]]) * (width - half_w)
    raw = top * half_h + bottom * (height - half_h)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")


def gpu_used() -> list[str]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    return (r.stdout.split() + ["", ""])[:2]


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--shapes", nargs="+", default=SHAPES)
    ap.add_argument("--csv", type=pathlib.Path, default=here.parent / "bench-results" / "images.csv")
    args = ap.parse_args()

    question = (
        "The image is split into four equal quadrants, each a single colour. Reply with exactly four words: the "
        "colours of the top-left, top-right, bottom-left and bottom-right quadrants, in that order."
    )
    new_file = not args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for shape in args.shapes:
            width, height = (int(x) for x in shape.lower().split("x"))
            body = {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(width, height)).decode()}},
                ]}],
                "max_tokens": 1024, "temperature": 0.0,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            }
            req = urllib.request.Request(
                f"{args.url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            t0, ok, correct, prompt_n, gen_n, error, answer = time.time(), 0, 0, "", "", "", ""
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    d = json.load(r)
                ok = 1
                answer = (d["choices"][0]["message"].get("content") or "").lower()
                words = [x.strip(".,;:") for x in answer.replace(",", " ").split()]
                correct = int([x for x in words if x in COLOURS] == list(ORDER))
                usage = d.get("usage") or {}
                prompt_n, gen_n = usage.get("prompt_tokens", ""), usage.get("completion_tokens", "")
            except urllib.error.HTTPError as e:
                error = f"HTTP {e.code}: {e.read()[:160].decode(errors='replace')}"
            except OSError as e:
                error = f"{type(e).__name__}: {e}"[:200]
            row = [args.label, width, height, ok, correct, prompt_n, gen_n, f"{time.time() - t0:.1f}", *gpu_used(), error]
            w.writerow(row)
            f.flush()
            print(" | ".join(str(x) for x in row), "| answer:", repr(answer[:60]), flush=True)


if __name__ == "__main__":
    main()
