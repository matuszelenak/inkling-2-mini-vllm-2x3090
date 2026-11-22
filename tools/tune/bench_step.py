#!/usr/bin/env python3
"""Seeded decode bench with per-step accounting from the server's spec-decode counters.

Every arm sees the same prompts (salted per workload and repeat, so nothing is
served from the prefix cache on a fresh server). Decode tok/s is
(completion_tokens - 1) over the time after the first streamed token; steps
are the drafts counter (or one per token without a drafter).

Usage: bench_step.py <label> <out.jsonl> [--url U] [--reps 3] [--max-tokens 512] [--temp T] [--effort E]
"""
import argparse, json, re, statistics, sys, time, urllib.request

WORKLOADS = [
    ("code", 0.0, "Write a complete Python implementation of a rope data structure supporting insert, delete, index, split, and concat, followed by a property-based test harness comparing it against a plain string reference. Code only, no commentary."),
    ("prose", 0.0, "Write an essay on how the shift from single-core to many-core CPUs changed software architecture between 2005 and 2020, with concrete historical examples."),
    ("json", 0.0, 'You have tools: read_file(path), write_file(path, content), run_command(cmd). Respond ONLY with a JSON array of tool calls (objects with "tool" and "args" keys) that would add token-bucket rate limiting to a Flask app at app/server.py with tests. No prose.'),
    ("chat", 0.0, "My sourdough starter doubles in about nine hours at 20 C. I want to bake on Saturday morning. Walk me through a schedule from Thursday night, and explain what to change if the kitchen is colder."),
    ("code t=1", 1.0, "Write a complete Python implementation of a rope data structure supporting insert, delete, index, split, and concat, followed by a property-based test harness comparing it against a plain string reference. Code only, no commentary."),
]


def counters(url):
    out = {}
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            m = re.match(r'vllm:spec_decode_num_(drafts|accepted_tokens|draft_tokens)_total\{[^}]*\} ([0-9.e+]+)', line)
            if m:
                out[m.group(1)] = float(m.group(2))
            m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\} ([0-9.e+]+)', line)
            if m:
                out[f"pos{m.group(1)}"] = float(m.group(2))
    return out


def stream(url, body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = {}; finish = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if t_first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    t_first = time.time()
                finish = c.get("finish_reason") or finish
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    return {"prompt_n": usage.get("prompt_tokens", 0), "gen_n": n, "ttft": (t_first or t1) - t0,
            "decode_s": t1 - (t_first or t1), "tps": (n - 1) / (t1 - t_first) if t_first and t1 > t_first and n > 1 else 0.0, "finish": finish}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--temp", type=float, help="override every workload's temperature")
    ap.add_argument("--effort", help="reasoning_effort for every request")
    a = ap.parse_args()
    rows = []
    for name, temp, prompt in WORKLOADS:
        if a.only and name not in a.only:
            continue
        if a.temp is not None:
            temp = a.temp
        base = {"max_tokens": a.max_tokens, "temperature": temp, "seed": 1234}
        if a.effort:
            base["chat_template_kwargs"] = {"reasoning_effort": a.effort}
        stream(a.url, {**base, "max_tokens": 48, "messages": [{"role": "user", "content": f"warmup {name}\n{prompt}"}]})
        for r in range(a.reps):
            c0 = counters(a.url)
            res = stream(a.url, {**base, "messages": [{"role": "user", "content": f"run 20261121-{name}-{r}\n{prompt}"}]})
            c1 = counters(a.url)
            d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
            steps = d.get("drafts") or res["gen_n"]
            res.update(label=a.label, workload=name, rep=r, steps=steps,
                       tok_per_step=res["gen_n"] / steps if steps else 0,
                       ms_per_step=1000 * res["decode_s"] / steps if steps else 0,
                       pos=[round(d.get(f"pos{i}", 0) / d["drafts"], 3) for i in range(7) if d.get("drafts") and f"pos{i}" in d])
            rows.append(res)
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")
    print(f"{a.label}")
    print(f"  {'workload':9} {'tok/s':>7} {'(min-max)':>13} {'tok/step':>8} {'ms/step':>8}  accept by position")
    for name, _, _ in WORKLOADS:
        rs = [r for r in rows if r["workload"] == name]
        if not rs:
            continue
        tps = [r["tps"] for r in rs]
        pos = [statistics.mean(r["pos"][i] for r in rs) for i in range(len(rs[0]["pos"]))] if rs[0]["pos"] else []
        print(f"  {name:9} {statistics.median(tps):7.1f} {min(tps):6.1f}-{max(tps):<6.1f} {statistics.mean(r['tok_per_step'] for r in rs):8.2f} "
              f"{statistics.mean(r['ms_per_step'] for r in rs):8.2f}  {' '.join(f'{p:.2f}' for p in pos)}")


if __name__ == "__main__":
    main()
