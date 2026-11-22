#!/usr/bin/env python3
"""Re-run two other projects' published benchmark protocols against this server.

Nothing from their repositories is executed: this is a re-implementation that sends the same prompts with the
same request parameters and computes the same metrics, so the numbers can be read against their tables.

A. agent2x0r/inkling2-sm86-kernels, benchmarks/suite_functional_and_speed.py (2x RTX 3090, NVLink):
   context scaling (1K..60K, 3 trials), output categories (3 trials), concurrency 1/2/4/8.
   512 tokens out, temperature 0.6, reasoning_effort low, prompts built from Python stdlib sources.
B. Pento95/HyperInkling, bench/spec_bench.py (1x RTX 3090 Ti): chat / code / struct / prose, one request at a
   time, 768 tokens out, temperature 0.2 (struct) or 0.5, default reasoning; then six at once.
   Its "doc" category needs a document that is not in the repository and is left out.

Usage: published_protocols.py <label> <out.json> [--url U] [--only a b]
"""
import argparse, concurrent.futures, json, pathlib, platform, statistics, sysconfig, threading, time, urllib.request

MODEL = "inkling-2-mini"


def metrics(url):
    out = {}
    for line in urllib.request.urlopen(url + "/metrics", timeout=30).read().decode().splitlines():
        for k in ("drafts", "accepted_tokens"):
            if line.startswith(f"vllm:spec_decode_num_{k}_total{{"):
                out[k] = out.get(k, 0.0) + float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_hits_total"):
            out["hits"] = out.get("hits", 0.0) + float(line.split()[-1])
    return out


def chat(url, content, max_tokens, temperature, effort=None, seed=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                t_last = time.perf_counter()
                d = ch.get("delta") or {}
                piece = (d.get("reasoning") or d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece and t_first is None:
                    t_first = time.perf_counter()
                text.append(piece)
    t1 = time.perf_counter()
    p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    ttft = (t_first - t0) if t_first else (t1 - t0)
    return {"prompt_tokens": p, "completion_tokens": c, "ttft": ttft, "prefill_tps": p / max(ttft, 1e-6),
            "decode_tps": (c - 1) / max(t1 - (t_first or t0), 1e-6) if c > 1 else 0.0,           # protocol A
            "decode_tps_b": (c - 1) / max((t_last or t1) - (t_first or t0), 1e-6) if c > 1 else 0.0,  # protocol B
            "text": "".join(text)}


def ms(v):
    return f"{statistics.mean(v):.1f} ± {statistics.pstdev(v) if len(v) > 1 else 0:.1f}"


def acc(m0, m1):
    d = m1.get("drafts", 0) - m0.get("drafts", 0)
    return 1 + (m1.get("accepted_tokens", 0) - m0.get("accepted_tokens", 0)) / d if d else 1.0


# ---------------------------------------------------------------- protocol A
A_CATEGORIES = [
    ("Code (Python/CUDA)", "Implement a complete LRU cache with TTL expiration in Python including type hints, unit tests, and docstring comments."),
    ("Math Reasoning", "Prove that there are infinitely many primes congruent to 3 mod 4, then solve 7x ≡ 3 (mod 29) and verify the result."),
    ("Technical Prose", "Write an in-depth architecture note explaining the trade-offs between tensor parallelism, expert parallelism, and pipeline parallelism for sparse MoE models on NVLink dual-GPU nodes."),
    ("Structured JSON", "Generate a valid JSON array of 12 detailed synthetic GPU benchmark entries with fields: gpu_name, arch, vram_gb, bandwidth_gbs, fp16_tflops, tdp_w, release_year, notes."),
]
A_PUBLISHED = {"scaling": {"1K": (4918, 114.7, 172.2), "4K": (5575, 113.6, 166.0), "16K": (5709, 110.0, 178.3), "32K": (5439, 105.6, 177.9), "60K": (4954, 99.4, 170.2)},
               "categories": {"Code (Python/CUDA)": (115.6, 202.4), "Math Reasoning": (115.8, 177.6), "Technical Prose": (115.5, 135.5), "Structured JSON": (115.7, 209.9)},
               "concurrency": {1: (113.5, 190.6), 2: (173.9, 277.9), 4: (249.0, 352.6), 8: (328.1, 383.4)}}


def stdlib_corpus():
    stdlib = pathlib.Path(sysconfig.get_path("stdlib"))
    files = sorted([p for p in stdlib.glob("*.py") if p.stat().st_size > 8000], key=lambda p: p.name)
    return "\n".join(f"# === FILE: {p.name} ===\n" + p.read_text(errors="ignore") for p in files[:50])


def protocol_a(url, drafter):
    corpus = stdlib_corpus(); out = {"scaling": [], "categories": [], "concurrency": []}
    col = 2 if drafter else 1
    print("A. context scaling (theirs: prefill, decode)")
    for label, n_chars in [("1K", 3_100), ("4K", 12_800), ("16K", 52_000), ("32K", 104_000), ("60K", 192_000)]:
        pre, dec, ac, hits, needles, ptoks = [], [], [], 0, 0, []
        for t in range(3):
            nonce = int(time.time() * 100) % 100000 + t * 997 + n_chars % 97
            needle = f"CHK-{1000 + t * 313 + n_chars % 700}"
            text = corpus[:n_chars]; mid = len(text) // 2
            doc = (f"# Benchmark session {nonce}\n" + text[:mid] + f"\n# BENCHMARK_CALIBRATION_TAG = \"{needle}\"\n" + text[mid:]
                   + "\n\n---\n1. State the exact BENCHMARK_CALIBRATION_TAG value from the code above.\n"
                   "2. Write a 6-sentence technical summary of Python standard library patterns visible in this excerpt.")
            m0 = metrics(url); r = chat(url, doc, 512, 0.6, "low"); m1 = metrics(url)
            pre.append(r["prefill_tps"]); dec.append(r["decode_tps"]); ac.append(acc(m0, m1)); ptoks.append(r["prompt_tokens"])
            hits += m1.get("hits", 0) - m0.get("hits", 0); needles += needle in r["text"]
        pub = A_PUBLISHED["scaling"][label]
        out["scaling"].append({"bucket": label, "prompt_tokens": round(statistics.mean(ptoks)), "prefill": pre, "decode": dec, "acc": ac, "needles": needles, "prefix_hits": hits})
        print(f"  {label:3s} ~{round(statistics.mean(ptoks)):6d} tok | prefill {ms(pre):>16} (theirs {pub[0]}) | decode {ms(dec):>13} (theirs {pub[col]}) | acc {statistics.mean(ac):.2f} | needle {needles}/3 | prefix hits {int(hits)}", flush=True)
    print("A. output categories")
    for cat, prompt in A_CATEGORIES:
        dec, ac = [], []
        for t in range(3):
            nonce = int(time.time() * 100) % 100000 + t * 503
            m0 = metrics(url); r = chat(url, f"[trial={nonce}] {prompt}", 512, 0.6, "low"); m1 = metrics(url)
            dec.append(r["decode_tps"]); ac.append(acc(m0, m1))
        out["categories"].append({"category": cat, "decode": dec, "acc": ac})
        print(f"  {cat:20s} | decode {ms(dec):>13} (theirs {A_PUBLISHED['categories'][cat][col - 1]}) | acc {statistics.mean(ac):.2f}", flush=True)
    print("A. concurrency (aggregate decode tok/s)")
    for c in (1, 2, 4, 8):
        m0 = metrics(url)

        def worker(i):
            nonce = int(time.time() * 100) % 100000 + c * 1000 + i * 73
            off = (i * 1700) % 25000
            doc = (f"# User {i} session {nonce}\n" + corpus[off: off + 12_800]
                   + "\n\n---\nSummarize the key classes and error-handling design patterns in this module in 6 technical sentences.")
            return chat(url, doc, 512, 0.6, "low")
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            res = [f.result() for f in [ex.submit(worker, i) for i in range(c)]]
        wall = time.perf_counter() - t0; m1 = metrics(url)
        comp = sum(r["completion_tokens"] for r in res); ttfts = sorted(r["ttft"] for r in res)
        agg = comp / max(wall - ttfts[0], 1e-6)
        out["concurrency"].append({"c": c, "agg_decode_tps": agg, "per_user": statistics.mean(r["decode_tps"] for r in res), "acc": acc(m0, m1)})
        print(f"  c={c} | aggregate {agg:6.1f} (theirs {A_PUBLISHED['concurrency'][c][col - 1]}) | per user {statistics.mean(r['decode_tps'] for r in res):5.1f} | acc {acc(m0, m1):.2f}", flush=True)
    return out


# ---------------------------------------------------------------- protocol B
B_PROMPTS = {
    "chat": ["Ciao! Mi spieghi in modo semplice cos'è l'inflazione e perché le banche centrali alzano i tassi?",
             "What are some good habits for staying productive when working from home?"],
    "code": ["Write a Python class implementing an LRU cache with get/put in O(1), with docstrings and a few unit tests.",
             "Scrivi una funzione TypeScript che fa il parsing di una data in formato DD/MM/YYYY e la valida, con test."],
    "struct": ["Estrai i dati in JSON con chiavi nome, cognome, data_nascita, citta, email, telefono da questo testo: "
               "'Mario Rossi, nato il 12/03/1985 a Torino, scrivere a mario.rossi@example.com o chiamare il 333 1234567. "
               "Laura Bianchi, nata il 01/09/1990 a Bari, laura.b@example.org, 340 7654321.' Rispondi solo con il JSON.",
               "Convert this CSV to a JSON array of objects, nothing else:\nid,name,price,stock\n1,Widget,9.99,120\n"
               "2,Gadget,24.50,15\n3,Doohickey,3.75,900\n4,Thingamajig,49.00,7\n5,Whatsit,12.30,64"],
    "prose": ["Scrivi un racconto breve su un vecchio guardiano del faro in Sardegna che riceve una lettera inattesa.",
              "Write a short essay on how the printing press changed European society."],
}
B_PUBLISHED = {"chat": (136, 145), "code": (135, 230), "struct": (135, 310), "prose": (135, 145)}


def protocol_b(url, drafter):
    out = {"c1": [], "c6": None}; col = 1 if drafter else 0
    temp = lambda cat: 0.2 if cat == "struct" else 0.5
    chat(url, B_PROMPTS["chat"][1], 768, 0.0, seed=0)   # warm-up, as theirs
    print("B. one request at a time (tok/s per stream)")
    for cat, prompts in B_PROMPTS.items():
        tps, ac, toks = [], [], 0
        for p in prompts:
            m0 = metrics(url); r = chat(url, p, 768, temp(cat), seed=0); m1 = metrics(url)
            tps.append(r["decode_tps_b"]); ac.append(acc(m0, m1)); toks += r["completion_tokens"]
        out["c1"].append({"category": cat, "tps": tps, "acc": ac, "tokens": toks})
        print(f"  {cat:7s} {statistics.mean(tps):6.1f} (theirs ~{B_PUBLISHED[cat][col]}) | {', '.join(f'{a:.2f}' for a in ac)} tok/step | {toks} tokens", flush=True)
    pairs = [(p, temp(c)) for c, ps in B_PROMPTS.items() for p in ps][:6]
    res = []; m0 = metrics(url); t0 = time.time()
    ts = [threading.Thread(target=lambda p=p, t=t: res.append(chat(url, p, 768, t, seed=0))) for p, t in pairs]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0; m1 = metrics(url)
    out["c6"] = {"per_stream": statistics.mean(r["decode_tps_b"] for r in res), "aggregate": sum(r["completion_tokens"] for r in res) / wall, "acc": acc(m0, m1)}
    print(f"B. six at once: {out['c6']['per_stream']:.1f} tok/s per stream, {out['c6']['aggregate']:.1f} aggregate ({out['c6']['acc']:.2f} tok/step)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--only", nargs="*", default=["a", "b"])
    a = ap.parse_args()
    drafter = metrics(a.url).get("drafts") is not None
    print(f"{a.label} | drafter {'on' if drafter else 'off'}")
    res = {"label": a.label, "drafter": drafter, "python": platform.python_version()}
    if "a" in a.only:
        res["agent2x0r_protocol"] = protocol_a(a.url, drafter)
    if "b" in a.only:
        res["hyperinkling_protocol"] = protocol_b(a.url, drafter)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
