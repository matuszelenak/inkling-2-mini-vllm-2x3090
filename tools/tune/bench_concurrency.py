#!/usr/bin/env python3
"""Aggregate output throughput at several concurrency levels.

Each user sends a different prompt (seeded, so every server sees the same set); 512 tokens out. Aggregate
tok/s is all completion tokens over the wall time from the first streamed token to the last.

Usage: bench_concurrency.py <label> <out.jsonl> [--url U] [--levels 1 2 4 8 16] [--max-tokens 512] [--temp 0.6] [--effort low]
"""
import argparse, concurrent.futures, json, statistics, time, urllib.request

TOPICS = ["a token-bucket rate limiter", "an LRU cache with TTL", "a trie with prefix search", "a retrying HTTP client",
          "a thread-safe bounded queue", "an interval tree", "a Bloom filter", "a consistent-hash ring", "a tiny regex matcher",
          "a JSON pointer resolver", "a priority scheduler", "a write-ahead log", "a union-find with path compression",
          "a skip list", "a circular buffer", "a debounce and throttle pair", "a matrix class with LU decomposition",
          "a CSV parser that handles quotes", "an event emitter", "a rope for text editing", "a segment tree",
          "an LFU cache", "a binary heap with decrease-key", "a suffix array builder", "a topological sorter",
          "a rate-limited worker pool", "a calendar with recurring events", "a diff algorithm", "a URL router",
          "a bank ledger with double entry", "a memoizing fibonacci service", "a sparse vector type"]
KINDS = [("code", "Write a complete, tested Python implementation of {t}. Code only."),
         ("prose", "Write a detailed design note about {t}: when to use it, trade-offs and failure modes."),
         ("json", "Describe {t} as a JSON object with keys name, purpose, operations (array of objects with name, complexity, description) and pitfalls (array). JSON only.")]


def stream(url, body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if t_first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    t_first = time.time()
    return t_first or time.time(), time.time(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--kind", default="mixed", choices=["mixed", "code", "prose", "json"])
    ap.add_argument("--ignore-eos", action="store_true", help="every request runs to max_tokens (for step-cost measurements)")
    a = ap.parse_args()
    print(a.label)
    for c in a.levels:
        def job(i):
            kind, tmpl = KINDS[i % 3] if a.kind == "mixed" else next(k for k in KINDS if k[0] == a.kind)
            prompt = f"request {c}-{i}\n" + tmpl.format(t=TOPICS[(i * 7 + c) % len(TOPICS)])
            return stream(a.url, {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": a.temp,
                                  "seed": 1000 + i, "chat_template_kwargs": {"reasoning_effort": a.effort}, **({"ignore_eos": True} if a.ignore_eos else {})})
        with concurrent.futures.ThreadPoolExecutor(c) as ex:
            res = list(ex.map(job, range(c)))
        first = min(r[0] for r in res); last = max(r[1] for r in res); toks = sum(r[2] for r in res)
        per = [(r[2] - 1) / max(r[1] - r[0], 1e-6) for r in res]
        row = {"label": a.label, "concurrency": c, "kind": a.kind, "tokens": toks, "agg_tps": toks / (last - first), "per_stream_tps": statistics.mean(per)}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  c={c:2d}: aggregate {row['agg_tps']:7.1f} tok/s | per stream {row['per_stream_tps']:6.1f} | {toks} tokens", flush=True)


if __name__ == "__main__":
    main()
