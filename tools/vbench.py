#!/usr/bin/env python3
"""Decode / prefill / long-context bench for a vLLM server.

Four short workloads, then optional depth prompts. vLLM reports no
per-request timings, so the request is streamed: prefill tok/s is
prompt_tokens / time-to-first-token, decode tok/s is (completion_tokens - 1)
over the time after the first token. Prefix caching is defeated with a unique
first line per request.

A depth prompt is N tokens of real text (--corpus-dir: any directory of source
and docs) with five planted facts at fixed fractions of the context, so each
depth row is also a retrieval check: a timing-only depth bench cannot tell a
healthy server from one emitting a repeated token.

Usage: vbench.py <label> <url> <csv> [--skip-short] [--max-tokens 512]
                 [--depth-tokens N ... --corpus-dir DIR]
"""
import argparse
import csv
import json
import pathlib
import random
import sys
import time
import urllib.request
import uuid

WORKLOADS = [
    # (name, temperature, prompt)
    ("code greedy", 0.0, "Write a complete Python implementation of a rope data structure supporting insert, delete, index, split, and concat, followed by a property-based test harness comparing it against a plain string reference. Code only, no commentary."),
    ("code temp1", 1.0, "Write a complete Python implementation of a rope data structure supporting insert, delete, index, split, and concat, followed by a property-based test harness comparing it against a plain string reference. Code only, no commentary."),
    ("prose greedy", 0.0, "Write an essay on how the shift from single-core to many-core CPUs changed software architecture between 2005 and 2020, with concrete historical examples."),
    ("json tools greedy", 0.0, 'You have tools: read_file(path), write_file(path, content), run_command(cmd). Respond ONLY with a JSON array of tool calls (objects with "tool" and "args" keys) that would add token-bucket rate limiting to a Flask app at app/server.py with tests. No prose.'),
]

# Planted facts: one per bin, at these fractions of the context.
FRACTIONS = (0.05, 0.30, 0.55, 0.80, 0.97)
BINS = ("KESTREL", "ORIOLE", "HERON", "WREN", "PIPIT")

CORPUS_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".svelte", ".rs", ".go", ".c", ".h", ".cpp", ".md", ".txt", ".toml", ".yml", ".yaml"}
CORPUS_SKIP = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}

FIELDS = ["label", "workload", "temperature", "prompt_n", "ttft_s", "pp_tok_s", "gen_n", "tg_tok_s", "needles", "found", "finish_reason", "wall_s"]


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def stream(url, body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = {}; finish = ""; content = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or delta.get("tool_calls")
                if piece and t_first is None:
                    t_first = time.time()
                if delta.get("content"):
                    content.append(delta["content"])
                finish = c.get("finish_reason") or finish
    t1 = time.time()
    n_p, n_c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    ttft = (t_first or t1) - t0
    return {"prompt_n": n_p, "ttft": ttft, "pp": n_p / ttft if ttft else 0, "gen_n": n_c,
            "tg": (n_c - 1) / (t1 - t_first) if t_first and t1 > t_first and n_c > 1 else 0,
            "finish": finish, "wall": t1 - t0, "content": "".join(content)}


def corpus(root: pathlib.Path) -> str:
    """Varied real text (source and docs under `root`): repetitive filler
    would let the drafter coast and inflate acceptance. Hidden files and
    directories are skipped: files are read in path order, and a dot
    directory of tool output sorts first and would become the whole prompt."""
    parts = []
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root).parts
        if f.suffix not in CORPUS_SUFFIXES or CORPUS_SKIP & set(rel) or any(p.startswith(".") for p in rel) or not f.is_file():
            continue
        try:
            parts.append(f"===== {f.relative_to(root)} =====\n{f.read_text(errors='ignore')}")
        except OSError:
            continue
    return "\n\n".join(parts)


def depth_prompt(url, text, target, rng):
    """`target` tokens of the corpus (repeated with a counter when it is too
    short), five facts planted, the question at the end."""
    chunks, n = [], 0
    while len("".join(chunks)) < target * 6:
        chunks.append(f"\n\n##### corpus pass {n} #####\n\n{text}")
        n += 1
    toks = post(url, "/tokenize", {"prompt": "".join(chunks), "add_special_tokens": False})["tokens"][:target]
    lines = post(url, "/detokenize", {"tokens": toks})["prompt"].split("\n")
    facts = {name: str(rng.randrange(10000, 99999)) for name in BINS}
    # insert from the back so earlier indices stay valid
    for frac, name in sorted(zip(FRACTIONS, BINS), reverse=True):
        lines.insert(int(len(lines) * frac), f"INVENTORY NOTE: bin {name} holds exactly {facts[name]} units.")
    q = ("Using only the INVENTORY NOTE lines in the documents above, how many units does each of these bins hold: "
         + ", ".join(BINS) + "? Reply with one line per bin in the form NAME=NUMBER and nothing else.")
    return f"run {uuid.uuid4().hex}\n" + "\n".join(lines) + "\n\n" + q, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("url"); ap.add_argument("csv", type=pathlib.Path)
    ap.add_argument("--depth-tokens", type=int, nargs="*", default=[])
    ap.add_argument("--corpus-dir", type=pathlib.Path, help="directory of source and docs to build depth prompts from")
    ap.add_argument("--skip-short", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    text = ""
    if a.depth_tokens:
        if not a.corpus_dir or not a.corpus_dir.is_dir():
            sys.exit("--depth-tokens needs --corpus-dir <directory of source and docs>")
        text = corpus(a.corpus_dir)
        if len(text) < 20000:
            sys.exit(f"{a.corpus_dir}: only {len(text)} characters of text found; point --corpus-dir at a larger tree")
    new = not a.csv.exists(); a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)

        def rec(name, temp, r, needles="", found=""):
            row = [a.label, name, temp, r["prompt_n"], f"{r['ttft']:.2f}", f"{r['pp']:.1f}", r["gen_n"], f"{r['tg']:.2f}", needles, found, r["finish"], f"{r['wall']:.1f}"]
            w.writerow(row); f.flush(); print(" | ".join(str(x) for x in row), flush=True)

        if not a.skip_short:
            for name, temp, prompt in WORKLOADS:
                salted = f"run {uuid.uuid4().hex}\n{prompt}"
                stream(a.url, {"messages": [{"role": "user", "content": salted}], "max_tokens": 32, "temperature": temp})  # warmup, discard
                rec(name, temp, stream(a.url, {"messages": [{"role": "user", "content": f"run {uuid.uuid4().hex}\n{prompt}"}], "max_tokens": a.max_tokens, "temperature": temp}))
        if a.depth_tokens:
            rng = random.Random(20261121)
            for target in a.depth_tokens:
                prompt, facts = depth_prompt(a.url, text, target, rng)
                r = stream(a.url, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.0,
                                   "chat_template_kwargs": {"reasoning_effort": "low"}})
                found = sum(f"{k}={v}" in r["content"].replace(" ", "") for k, v in facts.items())
                rec(f"deep {target}", 0.0, r, len(facts), found)
                print("   answer:", repr(r["content"][:150]), flush=True)


if __name__ == "__main__":
    main()
