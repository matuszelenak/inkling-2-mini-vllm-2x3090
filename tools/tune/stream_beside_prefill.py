#!/usr/bin/env python3
"""One request streams a long answer while another prefills a cold prompt: the prefill rate, the stream's rate
before, during and after, and the longest pause in the stream. Nothing else should be using the server.

usage: stream_beside_prefill.py --text FILE [--depth 60000] [--url URL]
"""
import argparse, json, os, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True, help="a plain-text file, about 6 bytes per token of --depth")
ap.add_argument("--depth", type=int, default=60000)
ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--model", default="inkling-2-mini")
args = ap.parse_args()


def post(path, body, timeout=900):
    req = urllib.request.Request(args.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def tok(text):
    return json.load(post("/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False}))["tokens"]


text = open(args.text, encoding="utf-8", errors="ignore").read()
prompt = tok(f"Document {uuid.uuid4()}\n") + tok(text[300000:300000 + args.depth * 6])[:args.depth]
stamps = []  # (time, completion tokens so far)


def stream():
    body = {"model": args.model, "max_tokens": 4000, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content": "Write a 2000-word essay on the history of the printing press."}],
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    with post("/v1/chat/completions", body) as r:
        for line in r:
            if line.startswith(b"data: {"):
                usage = json.loads(line[6:]).get("usage")
                if usage:
                    stamps.append((time.time(), usage["completion_tokens"]))


threading.Thread(target=stream, daemon=True).start()
time.sleep(4.0)
t0 = time.time()
json.load(post("/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                                   "add_special_tokens": False}))
t1 = time.time()
time.sleep(4.0)
t2 = time.time()


def rate(a, b):
    s = [(t, n) for t, n in stamps if a <= t <= b]
    return (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]) if len(s) > 1 else 0.0


gaps = [b[0] - a[0] for a, b in zip(stamps, stamps[1:]) if t0 <= a[0] <= t1]
print(f"cold {len(prompt)}-token prefill beside one stream: {t1 - t0:.1f} s ({len(prompt) / (t1 - t0):.0f} tok/s) | "
      f"stream before {rate(t0 - 3.5, t0):.0f} tok/s, during {rate(t0, t1):.1f}, after {rate(t1 + 0.5, t2):.0f} | "
      f"longest pause during {max(gaps) if gaps else 0:.2f} s", flush=True)
os._exit(0)
