#!/usr/bin/env python3
"""Collect varied model output (reasoning + answer + tool calls) to rank tokens for a reduced draft vocabulary.
usage: collect_outputs.py <out.jsonl> [--url URL] [--concurrency 8] [--max-tokens 700]"""
import argparse, itertools, json, random, threading, urllib.request

LANGS = ["Python", "JavaScript", "TypeScript", "C++", "Rust", "Go", "Java", "C", "SQL", "Bash", "C#", "Kotlin"]
CODE = ["a function that merges overlapping intervals", "an LRU cache", "a tokenizer for arithmetic expressions", "a retry wrapper with exponential backoff",
        "a binary search over a sorted array with tests", "a small HTTP client call with error handling", "a trie with insert and prefix search",
        "a function that parses a CSV line with quoted fields", "a debounce helper", "a matrix multiplication routine", "a rate limiter",
        "a function that validates an email address", "a depth-first search on a graph", "a queue backed by two stacks"]
CHAT = ["how vaccines train the immune system", "the difference between TCP and UDP", "why the sky is blue", "how to negotiate a salary",
        "what causes inflation", "how a refrigerator works", "tips for learning a new language", "how compilers differ from interpreters",
        "what a mortgage escrow account is", "how to brine a turkey", "the causes of the First World War", "how GPS finds your position",
        "what makes sourdough rise", "how to read a balance sheet", "why cats purr", "how noise-cancelling headphones work"]
PROSE = ["a short story about a lighthouse keeper", "a product description for a mechanical keyboard", "a polite email declining a meeting",
         "a blog intro about home networking", "a eulogy for a beloved teacher", "a travel diary entry from Lisbon", "a cover letter for a data analyst role",
         "a bedtime story about a fox", "release notes for a small software update", "a persuasive paragraph on public libraries"]
JSONT = ["a user profile with nested address and three orders", "a weather forecast for five days", "a list of four books with authors and ISBNs",
         "an OpenAPI path item for creating a user", "a package.json for a small web app", "a Kubernetes deployment manifest as JSON",
         "an invoice with line items and totals", "a tool call to search_flights with realistic arguments", "a config file for a logging system"]
MATH = ["A train leaves at 9:40 travelling 84 km/h; another leaves 25 minutes later at 96 km/h. When does it catch up?",
        "If 3 pumps fill a tank in 14 hours, how long do 7 pumps take? Show the reasoning.", "Compute the compound interest on 12,500 at 4.2% for 6 years.",
        "A recipe for 6 uses 450 g of flour. How much for 14 people, and what if flour comes in 1 kg bags?", "Solve 3x^2 - 7x - 6 = 0 and check the roots.",
        "What is the probability of at least two heads in five fair coin flips?"]
DOCS = ["a README for a command-line tool that renames files", "a markdown table comparing three databases", "a step-by-step guide to setting up SSH keys",
        "a changelog in Keep a Changelog format", "API documentation for a /users endpoint", "a troubleshooting section for a failing Docker build"]


def prompts(rng):
    out = []
    for lang, task in itertools.product(LANGS, CODE):
        out.append(("code", f"Write {task} in {lang}. Include comments and a short usage example."))
    rng.shuffle(out); out = out[:70]
    out += [("chat", f"Explain {t} to a curious adult.") for t in CHAT]
    out += [("chat", f"Give me practical advice: {t}.") for t in CHAT[:8]]
    out += [("prose", f"Write {t}.") for t in PROSE]
    out += [("json", f"Return only JSON: {t}.") for t in JSONT] + [("json", f"Produce {t} as pretty-printed JSON, then the same data as YAML.") for t in JSONT[:5]]
    out += [("math", t) for t in MATH]
    out += [("docs", f"Write {t}.") for t in DOCS]
    rng.shuffle(out)
    return out


def ask(url, model, prompt, max_tokens, effort, temp):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temp}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    text = "\n".join(x for x in (m.get("reasoning") or m.get("reasoning_content") or "", m.get("content") or "") if x)
    for t in m.get("tool_calls") or []:
        text += "\n" + json.dumps(t.get("function", {}))
    return text


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="inkling-2-mini"); ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=700)
    a = ap.parse_args()
    rng = random.Random(11); work = prompts(rng); lock = threading.Lock(); it = iter(enumerate(work)); done = [0]
    def worker():
        while True:
            with lock:
                nxt = next(it, None)
            if nxt is None:
                return
            i, (kind, p) = nxt
            try:
                text = ask(a.url, a.model, p, a.max_tokens, "low" if i % 2 else None, 0.7)
            except Exception as e:  # keep going; a failed request is just a missing sample
                text = ""
            with lock:
                with open(a.out, "a") as f:
                    f.write(json.dumps({"i": i, "kind": kind, "text": text}) + "\n")
                done[0] += 1
    th = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in th: t.start()
    for t in th: t.join()
    print(f"{done[0]} outputs written to {a.out}")


if __name__ == "__main__":
    main()
