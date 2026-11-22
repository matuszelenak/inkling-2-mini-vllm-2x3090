#!/usr/bin/env python3
"""Correctness gate for a tuned server.

collect-long <url> <out.json> <corpus-dir> [tokens]
    teacher-forced reference on a long token sequence from a known-good server.
check <url> <short_ref.json> <long_ref.json>
    1. short prefill vs the one-card reference (tf_reference_tp1.json)
    2. decode vs prefill on this server: greedy-generate, then teacher-force the same ids
    3. long prefill vs the long reference (full-attention layers past the relative-bias extent)
    4. long decode vs prefill: continue the long sequence greedily, then teacher-force
"""
import json, pathlib, statistics, sys, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
SUFFIXES = {".py", ".ts", ".svelte", ".md"}
SKIP = {".git", ".venv", "node_modules", "__pycache__"}


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def score(url, ids):
    d = post(url, "/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1, "add_special_tokens": False})
    out = []
    for tok, entry in zip(ids[1:], d["choices"][0]["prompt_logprobs"][1:]):
        best = min(entry.items(), key=lambda kv: kv[1]["rank"])
        actual = entry.get(str(tok))
        out.append((actual["logprob"] if actual else None, actual["rank"] if actual else None, int(best[0])))
    return out


def corpus_ids(url, root, n):
    parts = []
    for f in sorted(pathlib.Path(root).rglob("*")):
        if f.suffix in SUFFIXES and not (SKIP & set(f.parts)) and f.is_file():
            parts.append(f"===== {f.name} =====\n{f.read_text(errors='ignore')}")
    text = "\n\n".join(parts)[: n * 8]
    return post(url, "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"][:n]


def collect_long(url, out, root, n=16000):
    ids = corpus_ids(url, root, int(n))
    s = score(url, ids)
    json.dump({"ids": ids, "lp": [x[0] for x in s], "top1": [x[2] for x in s]}, open(out, "w"))
    lp = [x[0] for x in s if x[0] is not None]
    print(f"long reference: {len(ids)} tokens, mean NLL {-sum(lp) / len(lp):.4f}, actual token is rank 1 at {sum(x[1] == 1 for x in s) / len(s):.4f}")


def gaps(ref_lp, cand):
    g = sorted(abs(a - b[0]) for a, b in zip(ref_lp, cand) if a is not None and b[0] is not None)
    return statistics.median(g), g[int(len(g) * 0.95)], g[int(len(g) * 0.999)], g[-1]


def check(url, short_ref, long_ref):
    ok = True
    ref = json.load(open(short_ref))
    n = r1 = 0; allg = []; self_n = self_r1 = 0
    for item in ref:
        cand = score(url, item["ids"])
        gen = list(zip(item["scores"], cand))[item["n_prompt"] - 1:]
        for a, b in gen:
            n += 1; r1 += b[1] == 1
            if a["lp"] is not None and b[0] is not None:
                allg.append(abs(a["lp"] - b[0]))
        d = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": item["prompt"]}], "max_tokens": 192, "temperature": 0.0, "return_token_ids": True})
        ids = d["prompt_token_ids"] + d["choices"][0]["token_ids"]
        s = score(url, ids)[len(d["prompt_token_ids"]) - 1:]
        self_n += len(s); self_r1 += sum(x[1] == 1 for x in s)
    allg.sort()
    print(f"1 short prefill vs TP1 reference: reference token rank 1 {r1 / n:.4f} | |dlogprob| median {statistics.median(allg):.5f} p95 {allg[int(len(allg) * .95)]:.4f} max {allg[-1]:.3f}   [stock: 0.9876, 0.00124, 0.0704, 0.454]")
    ok &= r1 / n >= 0.98
    print(f"2 decode vs prefill, this server:  generated token rank 1 {self_r1 / self_n:.4f} over {self_n} tokens   [stock one card, no drafter: 0.9844]")
    ok &= self_r1 / self_n >= 0.97
    lr = json.load(open(long_ref))
    cand = score(url, lr["ids"])
    same = sum(a == b[2] for a, b in zip(lr["top1"], cand)) / len(cand)
    med, p95, p999, mx = gaps(lr["lp"], cand)
    lp = [b[0] for b in cand if b[0] is not None]
    far = list(zip(lr["lp"], cand))[4096:]
    medf, p95f, _, mxf = gaps([a for a, _ in far], [b for _, b in far])
    print(f"3 long prefill ({len(lr['ids'])} tok) vs reference: same top-1 {same:.4f} | |dlogprob| median {med:.5f} p95 {p95:.4f} p99.9 {p999:.3f} max {mx:.3f} | beyond 4K: median {medf:.5f} p95 {p95f:.4f} max {mxf:.3f} | NLL {-sum(lp) / len(lp):.4f}")
    ok &= same >= 0.97 and p95 < 0.25
    # 4. decode vs prefill at depth. One short continuation is too small a sample (a handful of near-ties decide
    #    it), so three windows of the reference are continued for 256 tokens each.
    tot = hit = 0
    for start in (0, 2000, 4000):
        prompt = lr["ids"][start:start + 12000]
        d = post(url, "/v1/completions", {"prompt": prompt, "max_tokens": 256, "temperature": 0.0, "add_special_tokens": False, "return_token_ids": True})
        gen = d["choices"][0].get("token_ids") or []
        s = score(url, prompt + gen)[len(prompt) - 1:]
        tot += len(s); hit += sum(x[1] == 1 for x in s)
    if tot:
        print(f"4 long decode vs prefill, this server: generated token rank 1 {hit / tot:.4f} over {tot} tokens   [stock port: 0.981 over 2,304]")
        ok &= hit / tot >= 0.965
    else:
        print("4 long decode: no token ids returned"); ok = False
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def longdecode(url, long_ref, n="768", out=None):
    """Decode vs prefill at depth with a larger sample: continue the long reference greedily for n tokens,
    then teacher-force the same ids on this server. Several windows of the reference give several samples."""
    lr = json.load(open(long_ref)); n = int(n); tot = hit = 0; gens = []
    for start in (0, 2000, 4000):
        prompt = lr["ids"][start:start + 12000]
        d = post(url, "/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0.0, "add_special_tokens": False, "return_token_ids": True})
        gen = d["choices"][0].get("token_ids") or []
        s = score(url, prompt + gen)[len(prompt) - 1:]
        tot += len(s); hit += sum(x[1] == 1 for x in s); gens.append(gen)
        print(f"  window {start}: generated {len(gen)}, rank-1 under teacher forcing {sum(x[1] == 1 for x in s) / len(s):.4f}", flush=True)
    print(f"long decode vs prefill (12K context, {tot} tokens): generated token rank 1 {hit / tot:.4f} ({tot - hit} mismatches)")
    if out:
        json.dump(gens, open(out, "w"))


if __name__ == "__main__":
    {"collect-long": collect_long, "check": check, "longdecode": longdecode}[sys.argv[1]](*sys.argv[2:])
