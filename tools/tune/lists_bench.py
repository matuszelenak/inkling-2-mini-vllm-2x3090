#!/usr/bin/env python3
"""Very predictable output: numbered lists, one request and eight at once. Not another project's protocol (none
publishes its list prompts); a category run of this repo's own.
usage: lists_bench.py <label> <out.json> [--url URL] [--effort none]"""
import argparse, json, threading, time, urllib.request


def ask(url, model, prompt, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if t_first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    t_first = time.perf_counter()
    t1 = time.perf_counter()
    return {"completion_tokens": n, "decode_s": t1 - (t_first or t0), "wall_s": t1 - t0}


def prompt(i):
    step = (3, 7, 11, 13, 17, 19, 23, 29)[i % 8]
    return (f"List the first 150 multiples of {step}, one per line, in the form `n x {step} = result`. "
            "Output only the list, no commentary.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--effort", default="none"); ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args(); out = {"label": a.label, "effort": a.effort, "runs": []}
    for users in (1, 1, 1, 8, 8, 8):
        res = [None] * users
        def run(i): res[i] = ask(a.url, a.model, prompt(i), a.max_tokens, a.effort)
        th = [threading.Thread(target=run, args=(i,)) for i in range(users)]
        t0 = time.perf_counter()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.perf_counter() - t0
        tot = sum(r["completion_tokens"] for r in res)
        per = sum((r["completion_tokens"] - 1) / r["decode_s"] for r in res) / users
        row = {"users": users, "completion_tokens": tot, "wall_s": wall, "aggregate_tok_s": tot / wall, "per_stream_decode_tok_s": per}
        out["runs"].append(row)
        print(f"  {users} user(s): aggregate {row['aggregate_tok_s']:7.1f} tok/s | per stream after first token {per:6.1f} | {tot} tokens", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
