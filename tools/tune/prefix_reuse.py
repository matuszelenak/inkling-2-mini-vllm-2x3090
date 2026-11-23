#!/usr/bin/env python3
"""Prefix-cache reuse on a running server, for the conv-state cache of patches/sm86-tune.diff.

  shared  requests that share a 10,000-token prefix and then differ: how many prompt tokens each takes from the cache.
  exact   one cold 30,000-token prompt, then prompts that end a few tokens past a point inside it, so each resumes
          from cached state and the first generated token shows that state directly.

--save FILE writes each request's first-token top-5 logprobs; --compare FILE prints the gap to a saved run. Two
servers that prefill in the same chunks (same --max-num-batched-tokens, --no-async-scheduling) and resume at the same
point agree exactly, which is how the checkpoints were checked against the stock conv-state manager
(INKLING_SM86_TUNE without convtail). Against a server without prefix caching the gap is the usual 0.01-0.2 of
prefilling the same text in different chunks. Nothing else should be using the server.

usage: prefix_reuse.py {shared|exact} --text FILE [--url URL] [--save FILE | --compare FILE]
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["shared", "exact"])
ap.add_argument("--text", required=True, help="a plain-text file of at least 1 MB")
ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--save"); ap.add_argument("--compare")
args = ap.parse_args()


def post(path, body, timeout=900):
    req = urllib.request.Request(args.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def hits():
    with urllib.request.urlopen(args.url + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_hits_total"):
                return int(float(line.split()[-1]))
    return 0


def tok(text):
    return post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False})["tokens"]


ref = json.load(open(args.compare)) if args.compare else {}
out = {}


def run(key, label, ids):
    before = hits()
    r = post("/v1/completions", {"model": args.model, "prompt": ids, "max_tokens": 4, "temperature": 0, "logprobs": 5,
                                 "add_special_tokens": False})
    lp = r["choices"][0]["logprobs"]
    out[key] = {"token": lp["tokens"][0], "top": lp["top_logprobs"][0]}
    line = f"{label:<46} prompt {r['usage']['prompt_tokens']:>6} | from cache {hits() - before:>6}"
    if key in ref:
        gaps = [abs(v - ref[key]["top"][k]) for k, v in out[key]["top"].items() if k in ref[key]["top"]]
        same = "same" if out[key]["token"] == ref[key]["token"] else "DIFFERENT"
        line += f" | first token {same}, top-5 |dlogprob| max {max(gaps) if gaps else float('nan'):.4f}"
    print(line, flush=True)


text = open(args.text, encoding="utf-8", errors="ignore").read()
if args.mode == "shared":
    shared = tok(text[600000:680000])[:10000]
    own = [tok(text[700000 + i * 20000: 712000 + i * 20000]) for i in range(5)]
    first = shared + own[0][:1500]
    run("A", "A  = shared 10,000 + 1,500 own (cold)", first)
    run("B", "B  = shared 10,000 + 1,500 other", shared + own[1][:1500])
    run("C", "C  = shared 10,000 + 1,500 other", shared + own[2][:1500])
    run("D", "D  = shared 10,000 + 200 other", shared + own[3][:200])
    run("E", "E  = shared 10,000 + 1,500 other", shared + own[4][:1500])
    run("A2", "A2 = A + 300 more (a follow-up turn)", first + own[0][1500:1800])
    run("F", "F  = first 6,000 of the shared part + 1,500", shared[:6000] + own[1][2000:3500])
    run("G", "G  = first 6,000 of the shared part + 1,500", shared[:6000] + own[2][2000:3500])
else:
    ids = tok(text[200000:400000])[:31000]
    run("P", "cold prompt", ids[:30000])
    run("branch", "its first 20,000 tokens + other text",
        ids[:20000] + tok("\n\nSummarize the passage above in one sentence.\n"))
    for n in (29990, 23050, 19460, 12810, 10000, 8200, 4100):
        run(str(n), f"its first {n:,} tokens", ids[:n])
if args.save:
    json.dump(out, open(args.save, "w"))
