#!/usr/bin/env python3
"""Build the token ranking for the reduced draft vocabulary (patches/draft_vocab_rank.json).

The drafter scores only the K highest-ranked token ids of each vocabulary shard (INKLING_SM86_DVOCAB_K, default
24576), so the ranking should put first the tokens the model actually writes. This script counts tokens in the
text you give it, weights each source by 1 / its token count times `weight`, and writes the ids in rank order.
The shipped ranking was built from model output (collect_outputs.py, weight 6), wikitext-2, HellaSwag, a novel,
Python and C source and a tree of configs and docs. It is English and code: for other languages build your own
and point INKLING_SM86_DVOCAB at it.

usage: build_draft_vocab.py <out.json> [--outputs outputs.jsonl[:weight]] [--text PATH[:weight] ...]
                            [--always LO-HI ...] [--held-out NAME=PATH ...] [--url URL]
PATH is a text file or a directory of text (read like vbench.py's --corpus-dir). --always puts id ranges first
whatever their counts (the tokenizer's added tokens: 199998-200057 for this model). --held-out reports how much
of a text the top K ids of each shard cover.
"""
import argparse, collections, importlib.util, json, pathlib

SHARD = 100512   # TP-local rows of the LM head at tensor parallel 2 (padded vocabulary 201,024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--outputs"); ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--always", action="append", default=[]); ap.add_argument("--held-out", action="append", default=[])
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("vbench", here.parent / "vbench.py")
    vb = importlib.util.module_from_spec(spec); spec.loader.exec_module(vb)

    def tok(text):
        ids = []
        for i in range(0, len(text), 600_000):
            ids += vb.post(a.url, "/tokenize", {"prompt": text[i:i + 600_000], "add_special_tokens": False})["tokens"]
        return ids

    def read(path):
        p = pathlib.Path(path)
        return vb.corpus(p) if p.is_dir() else p.read_text(errors="ignore")

    def split(arg, default):
        path, _, w = arg.rpartition(":")
        try:
            return (path, float(w)) if path else (arg, default)
        except ValueError:
            return arg, default

    score = collections.Counter()
    sources = []
    if a.outputs:
        path, w = split(a.outputs, 6.0)
        text = "\n\n".join(json.loads(l)["text"] for l in open(path) if l.strip())
        sources.append((f"model output ({path})", tok(text), w))
    for arg in a.text:
        path, w = split(arg, 1.0)
        sources.append((path, tok(read(path)), w))
    for name, ids, w in sources:
        print(f"{len(ids):9d} tokens, weight {w:g}: {name}")
        for t in ids:
            score[t] += w / len(ids)
    always = []
    for r in a.always:
        lo, hi = map(int, r.split("-")); always += list(range(lo, hi + 1))
    first = set(always)
    rank = always + [t for t, _ in sorted(score.items(), key=lambda kv: -kv[1]) if t not in first]
    json.dump(rank, open(a.out, "w"))
    per = [sum(1 for t in rank if s * SHARD <= t < (s + 1) * SHARD) for s in (0, 1)]
    print(f"{len(rank)} ids ranked ({per[0]} in the first shard, {per[1]} in the second) -> {a.out}")
    ks = (8192, 16384, 24576, 32768)
    keep = {k: {t for s in (0, 1) for t in [x for x in rank if s * SHARD <= x < (s + 1) * SHARD][:k]} for k in ks}
    for arg in a.held_out:
        name, _, path = arg.partition("=")
        ids = tok(read(path))
        print(f"coverage of {name} ({len(ids)} tokens): " + ", ".join(f"K={k} {100 * sum(1 for t in ids if t in keep[k]) / len(ids):.2f}%" for k in ks))


if __name__ == "__main__":
    main()
