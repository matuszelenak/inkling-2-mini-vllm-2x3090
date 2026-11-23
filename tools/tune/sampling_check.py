#!/usr/bin/env python3
"""Does sampling still follow the same distribution? Sample short continuations of fixed prompts at temperature 1
with many seeds and record the token ids. Run it against two servers and compare the files with `compare`.
Positions after the first are produced through draft verification, so a drafter that biased sampling would shift
their distribution.

usage: sampling_check.py collect <out.json> [--url URL] [--n 4000] [--concurrency 8]
       sampling_check.py compare <a.json> <b.json> [<ranking.json> <K per shard>]"""
import argparse, collections, json, math, sys, threading, urllib.request

PROMPTS = [
    "Here is a list of unrelated English words, one per line:\nriver\nlantern\nvelvet\n",
    "def main():\n    # pick something to do next\n    ",
    "Names of imaginary towns: Eldermoor, Quillhaven, Brackenfold,",
    '{"items": ["anvil", "marble", "thistle", ',
]


def collect(a):
    work = [(p, s) for s in range(a.n // len(PROMPTS)) for p in range(len(PROMPTS))]
    out = [None] * len(work); it = iter(enumerate(work)); lock = threading.Lock()
    def run():
        while True:
            with lock:
                nxt = next(it, None)
            if nxt is None:
                return
            i, (p, seed) = nxt
            body = {"model": a.model, "prompt": PROMPTS[p], "max_tokens": 5, "temperature": 1.0, "top_p": 1.0, "seed": seed,
                    "logprobs": 1, "return_tokens_as_token_ids": True}
            req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                toks = json.loads(r.read())["choices"][0]["logprobs"]["tokens"]
            out[i] = [p] + [int(t.split(":")[1]) for t in toks]
    th = [threading.Thread(target=run) for _ in range(a.concurrency)]
    for t in th: t.start()
    for t in th: t.join()
    json.dump(out, open(a.out, "w")); print(len(out), "samples ->", a.out)


def tv(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    return 0.5 * sum(abs(c1[k] / n1 - c2[k] / n2) for k in set(c1) | set(c2))


def compare(a):
    A = json.load(open(a.a)); B = json.load(open(a.b))
    keep = None
    if a.ranking:
        rank = json.load(open(a.ranking)); K = int(a.k); keep = set()
        for s in (0, 1):
            keep.update([t for t in rank if s * 100512 <= t < (s + 1) * 100512][:K])
    print("position | prompt | TV(A,B) | TV(A half 1, A half 2)  [sampling noise] | distinct ids A/B" + (" | mass outside the kept set A/B" if keep else ""))
    worst = 0.0
    for pos in range(1, 6):
        for p in range(len(PROMPTS)):
            a_ = [r[pos] for r in A if r and r[0] == p and len(r) > pos]; b_ = [r[pos] for r in B if r and r[0] == p and len(r) > pos]
            ca, cb = collections.Counter(a_), collections.Counter(b_)
            noise = tv(collections.Counter(a_[0::2]), collections.Counter(a_[1::2]))
            d = tv(ca, cb); extra = ""
            if keep is not None:
                extra = f" | {sum(1 for t in a_ if t not in keep)/len(a_):.3f} / {sum(1 for t in b_ if t not in keep)/len(b_):.3f}"
            # the two-half noise uses n/2 samples each, so it overstates the n-vs-n noise by about sqrt(2)
            print(f"   {pos}     |   {p}    |  {d:.3f}  |  {noise:.3f}  | {len(ca)}/{len(cb)}{extra}")
            worst = max(worst, d - noise)
    print("largest excess of TV(A,B) over the within-A noise:", round(worst, 3))


ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("collect"); c.add_argument("out"); c.add_argument("--url", default="http://localhost:8000"); c.add_argument("--model", default="inkling-2-mini")
c.add_argument("--n", type=int, default=4000); c.add_argument("--concurrency", type=int, default=8)
m = sub.add_parser("compare"); m.add_argument("a"); m.add_argument("b"); m.add_argument("ranking", nargs="?"); m.add_argument("k", nargs="?")
a = ap.parse_args()
collect(a) if a.cmd == "collect" else compare(a)
