#!/usr/bin/env python3
"""Re-implementation of the published benchmark protocols of two more projects, to read this server against
their tables. Their prompts and request parameters are used as published in their READMEs and scripts; none of
their code is used.

  lulakien/inkling2-mini-3090-recipe   tests/prefill_ttft.py, tests/spec_bench.py
  skyne98/inkling-ingine-3090          native/apps/bench/main.cpp

usage: more_protocols.py <label> <out.json> [--base URL] [--model NAME] [--skip-prefill]
"""
import argparse, json, random, sys, time, threading, urllib.request

LULAKIEN_PROMPTS = {
    "code": "Write a complete Python implementation of an LRU cache class with get/put in O(1), with docstrings and a few unit tests.",
    "explanation": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery) in detail.",
    "story": "Write a long story about a lighthouse keeper who discovers a message in a bottle.",
}
LULAKIEN_PUBLISHED = {"code": 313, "explanation": 166, "story": 126, "streams": {2: 260, 4: 339, 8: 445},
                      "prefill": {2000: 7480, 16000: 7140, 64000: 6280}}
INGINE_PROMPTS = {
    "chat": "Explain how a hash map handles collisions, with a short Python example.",
    "code": "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.",
}
INGINE_PUBLISHED = {"chat": "235-239", "code": "219-224"}


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def prefill_random_ids(base, model, n, rng):
    """Their method: non-streamed /v1/completions, unique random token ids, max_tokens 1, tok/s = n / wall."""
    ids = [rng.randrange(1000, 100000) for _ in range(n)]
    t0 = time.perf_counter()
    out = post(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0})
    dt = time.perf_counter() - t0
    return {"n": n, "prompt_tokens": out["usage"]["prompt_tokens"], "wall_s": dt, "tok_s": n / dt}


def chat_nonstream(base, model, prompt, max_tokens, effort):
    """lulakien spec_bench: non-streamed chat, temperature 0, tok/s = completion tokens / wall."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    t0 = time.perf_counter()
    out = post(base, "/v1/chat/completions", body)
    dt = time.perf_counter() - t0
    n = out["usage"]["completion_tokens"]
    return {"completion_tokens": n, "wall_s": dt, "tok_s": n / dt, "finish": out["choices"][0]["finish_reason"]}


def chat_stream(base, model, prompt, max_tokens, effort):
    """ingine bench: greedy, rate = (generated - 1) / seconds after the first token."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t_first = None; n = 0; finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if t_first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls")):
                    t_first = time.perf_counter()
                finish = ch.get("finish_reason") or finish
    dt = time.perf_counter() - t_first
    return {"completion_tokens": n, "decode_s": dt, "tok_s": (n - 1) / dt, "finish": finish}


def streams(base, model, k, max_tokens, effort):
    names = list(LULAKIEN_PROMPTS)
    res = [None] * k
    def run(i):
        res[i] = chat_nonstream(base, model, LULAKIEN_PROMPTS[names[i % len(names)]], max_tokens, effort)
    th = [threading.Thread(target=run, args=(i,)) for i in range(k)]
    t0 = time.perf_counter()
    for t in th: t.start()
    for t in th: t.join()
    dt = time.perf_counter() - t0
    tot = sum(r["completion_tokens"] for r in res)
    return {"streams": k, "completion_tokens": tot, "wall_s": dt, "agg_tok_s": tot / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--base", default="http://localhost:8000"); ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--skip-prefill", action="store_true")
    a = ap.parse_args()
    rng = random.Random(20261121)
    out = {"label": a.label, "python": sys.version.split()[0], "lulakien": {"published": LULAKIEN_PUBLISHED}, "ingine": {"published": INGINE_PUBLISHED}}

    if not a.skip_prefill:
        print("lulakien prefill protocol (random token ids, /v1/completions, max_tokens 1)")
        out["lulakien"]["prefill"] = {}
        prefill_random_ids(a.base, a.model, 512, rng)  # warm
        for n in (2000, 16000, 64000):
            runs = [prefill_random_ids(a.base, a.model, n, rng) for _ in range(3)]
            out["lulakien"]["prefill"][n] = runs
            v = sorted(r["tok_s"] for r in runs)
            print(f"  {n:6d} tokens | {v[0]:7.0f} {v[1]:7.0f} {v[2]:7.0f} tok/s | theirs {LULAKIEN_PUBLISHED['prefill'][n]}")

    print("lulakien spec bench (non-streamed, 1500 tokens, temperature 0, low effort)")
    out["lulakien"]["spec"] = {}
    for name, p in LULAKIEN_PROMPTS.items():
        runs = [chat_nonstream(a.base, a.model, p, 1500, "low") for _ in range(3)]
        out["lulakien"]["spec"][name] = runs
        print(f"  {name:12s} | " + " ".join(f"{r['tok_s']:6.1f}" for r in runs) + f" tok/s | tokens {[r['completion_tokens'] for r in runs]} | theirs {LULAKIEN_PUBLISHED[name]}")
    out["lulakien"]["streams"] = []
    for k in (2, 4, 8):
        r = streams(a.base, a.model, k, 1500, "low"); out["lulakien"]["streams"].append(r)
        print(f"  {k} streams | aggregate {r['agg_tok_s']:6.1f} tok/s | theirs {LULAKIEN_PUBLISHED['streams'][k]}")

    print("ingine bench (streamed, 512 tokens, greedy, 3 runs; rate after the first token)")
    out["ingine"]["runs"] = {}
    for effort in (None, "low"):
        for name, p in INGINE_PROMPTS.items():
            runs = [chat_stream(a.base, a.model, p, 512, effort) for _ in range(3)]
            out["ingine"]["runs"][f"{name}, effort {effort or 'default'}"] = runs
            print(f"  {name:5s} effort {effort or 'default':7s} | " + " ".join(f"{r['tok_s']:6.1f}" for r in runs) + f" tok/s | theirs {INGINE_PUBLISHED[name]}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
