#!/usr/bin/env python3
"""Draft acceptance, prefill and decode by context depth on a running server (drafter on).
A depth prompt is N tokens of real text from --corpus-dir, a unique first line (no prefix-cache reuse) and one
fixed writing task, so the kind of output is the same at every depth. Draft counters are read from /metrics
before and after each request; nothing else should be using the server.

usage: depth_accept.py <label> <out.jsonl> --corpus-dir DIR [--depths N ...] [--reps 2] [--temp 0.6] [--effort low]
"""
import argparse, importlib.util, json, pathlib, sys, uuid

HERE = pathlib.Path(__file__).resolve().parent
TASK = ("Write about 400 words explaining what the material above covers and how its parts relate to each other. "
        "Plain prose, no lists.")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--corpus-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tools", type=pathlib.Path, required=True, help="the repo's tools/ directory (vbench.py, tune/bench_step.py)")
    ap.add_argument("--depths", type=int, nargs="+", default=[2000, 60000, 110000, 190000])
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--effort", default="low"); ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    vb = load(a.tools / "vbench.py", "vbench"); bs = load(a.tools / "tune" / "bench_step.py", "bench_step")
    text = vb.corpus(a.corpus_dir)
    chunks, n = [], 0
    while len("".join(chunks)) < max(a.depths) * 6:
        chunks.append(f"\n\n##### corpus pass {n} #####\n\n{text}"); n += 1
    toks = vb.post(a.url, "/tokenize", {"prompt": "".join(chunks), "add_special_tokens": False})["tokens"]
    print(f"{a.label}\n  depth     prompt  ttft_s  prefill  gen  tok/s  steps  tok/step  accept by position")
    for depth in a.depths:
        body_text = vb.post(a.url, "/detokenize", {"tokens": toks[:depth]})["prompt"]
        for rep in range(a.reps):
            prompt = f"run {uuid.uuid4().hex}\n{body_text}\n\n{TASK}"
            before = bs.counters(a.url)
            r = bs.stream(a.url, {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": a.max_tokens, "temperature": a.temp,
                                  "chat_template_kwargs": {"reasoning_effort": a.effort}})
            after = bs.counters(a.url)
            steps = after.get("drafts", 0) - before.get("drafts", 0)
            acc = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
            pos = [round((after.get(f"pos{i}", 0) - before.get(f"pos{i}", 0)) / steps, 3) if steps else 0 for i in range(7)]
            row = {"label": a.label, "depth": depth, "rep": rep, "temperature": a.temp, "effort": a.effort, **r,
                   "prefill_tok_s": r["prompt_n"] / r["ttft"] if r["ttft"] else 0, "steps": steps,
                   "tok_per_step": 1 + acc / steps if steps else 0, "pos": pos}
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {depth:7d} {r['prompt_n']:8d} {r['ttft']:7.2f} {row['prefill_tok_s']:8.0f} {r['gen_n']:4d} {r['tps']:6.1f} "
                  f"{int(steps):5d} {row['tok_per_step']:8.2f}  {' '.join(f'{p:.2f}' for p in pos)}  {r['finish']}", flush=True)


if __name__ == "__main__":
    main()
