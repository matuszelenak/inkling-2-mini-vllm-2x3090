#!/usr/bin/env python3
"""GSM8K accuracy gate for an OpenAI-compatible chat endpoint.

Mirrors the chat protocol of vLLM's tests/evals/gsm8k/gsm8k_eval.py so the
score lines up with what vLLM publishes for this model (96-97%): a 5-shot
prompt built from the first five train examples, sent as one user message,
temperature 0, and the prediction is the last integer in the answer content
(reasoning the server parsed out is not scored).

Writes one row per question so two engines or two configs can be compared
item by item, not just by their totals.

Usage: tools/gsm8k_eval.py <label> [--url http://localhost:8000] [-n 200]
                           [--concurrency 1] [--max-tokens 4096] [--csv FILE]
Appends to bench-results/gsm8k.csv unless --csv is given.
"""

import argparse
import csv
import json
import pathlib
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
INVALID = -9999999
FIELDS = ["label", "idx", "gold", "pred", "correct", "finish_reason", "completion_tokens", "wall_s"]


def load(split: str, data_dir: pathlib.Path) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=split), path)
    return [json.loads(line) for line in path.read_text().splitlines() if line and not line.startswith("#")]


def answer_value(text: str) -> int:
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else INVALID


def ask(url: str, prompt: str, max_tokens: int) -> tuple[str, str, int, float]:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    choice = d["choices"][0]
    return (
        choice["message"].get("content") or "",
        choice.get("finish_reason") or "",
        d.get("usage", {}).get("completion_tokens", 0),
        time.time() - t0,
    )


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", "--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path.home() / ".cache" / "gsm8k")
    ap.add_argument("--csv", type=pathlib.Path, default=here.parent / "bench-results" / "gsm8k.csv")
    args = ap.parse_args()

    train, test = load("train", args.data_dir), load("test", args.data_dir)
    shots = "".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n" for ex in train[: args.num_shots])
    items = test[: args.num_questions]
    prompts = [f"{shots}Question: {ex['question']}\nAnswer:" for ex in items]
    golds = [answer_value(ex["answer"]) for ex in items]
    assert INVALID not in golds

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda p: ask(args.url, p, args.max_tokens), prompts))
    wall = time.time() - t0

    new_file = not args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    correct = invalid = length = tokens = 0
    with args.csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for i, (gold, (content, finish, n_tok, secs)) in enumerate(zip(golds, results)):
            pred = answer_value(content)
            correct += pred == gold
            invalid += pred == INVALID
            length += finish == "length"
            tokens += n_tok
            w.writerow([args.label, i, gold, "" if pred == INVALID else pred, int(pred == gold), finish, n_tok, f"{secs:.2f}"])

    n = len(items)
    print(f"{args.label}: accuracy {correct}/{n} = {correct / n:.3f}; invalid {invalid}; hit max_tokens {length}")
    print(f"  {tokens} output tokens in {wall:.0f} s = {tokens / wall:.1f} tok/s aggregate at concurrency {args.concurrency}")


if __name__ == "__main__":
    main()
