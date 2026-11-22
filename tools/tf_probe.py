#!/usr/bin/env python3
"""Token-level agreement between two vLLM servers of the same model.

collect <url> <out.json>   greedy-generate from a fixed prompt set on a
                           reference server, keep the exact token ids, then
                           score them teacher-forced (prompt_logprobs).
compare <url> <ref.json>   score the same token ids on another server and
                           compare, position by position: is the reference
                           token still rank 1, and how far did its logprob move?
nll <url>                  mean NLL of a fixed English passage (sanity: a
                           broken forward pass sits far above 3 nats/token).

A boot proves nothing (a port can boot and still produce word salad), and
free-running greedy outputs diverge at near-ties, so the comparison is
teacher-forced on identical ids.
"""
import json
import statistics
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Summarize the causes of the 2008 financial crisis in five sentences.",
    "A train leaves at 14:10 travelling 80 km/h; another leaves the same station at 14:40 at 100 km/h on the same track. When does the second catch the first? Show the work.",
    "Translate into French, German and Spanish: 'The warehouse closes at six, but deliveries are accepted until eight.'",
    "Explain the difference between a mutex and a semaphore to a new systems programmer.",
    "Give a JSON object with keys name, year and three_facts describing the Hubble Space Telescope.",
    "What are the trade-offs between B-trees and LSM-trees for a write-heavy workload?",
    "Write a limerick about a GPU that ran out of memory.",
]
PASSAGE = (
    "The history of computing is often told as a sequence of machines, but it is better understood as a sequence of "
    "ideas about what can be delegated. Early mechanical calculators delegated arithmetic; stored-program computers "
    "delegated the sequencing of arithmetic; compilers delegated the translation of intent into sequence. Each step "
    "moved a human skill into a mechanism, and each time the people closest to the old skill doubted that the new "
    "mechanism could be trusted. Time-sharing systems in the 1960s let many people use one machine at once, and the "
    "habits formed there, files, processes, permissions and shells, still shape the systems in use today. Networking "
    "then turned isolated machines into a shared medium, and the protocols designed for a few research sites proved "
    "flexible enough to carry commerce, conversation and entertainment for billions of people."
)


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def score(url: str, ids: list[int]) -> list[dict]:
    """Teacher-forced: for each position >= 1, the logprob of the actual token and the top-1 token id."""
    d = post(url, "/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1,
                                      "add_special_tokens": False})
    out = []
    for tok, entry in zip(ids[1:], d["choices"][0]["prompt_logprobs"][1:]):
        best = min(entry.items(), key=lambda kv: kv[1]["rank"])
        actual = entry.get(str(tok))
        out.append({"tok": tok, "lp": actual["logprob"] if actual else None, "rank": actual["rank"] if actual else None,
                    "top1": int(best[0]), "top1_lp": best[1]["logprob"]})
    return out


def collect(url: str, path: str) -> None:
    ref = []
    for p in PROMPTS:
        d = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 192,
                                                 "temperature": 0.0, "return_token_ids": True})
        ids = d["prompt_token_ids"] + d["choices"][0]["token_ids"]
        ref.append({"prompt": p, "n_prompt": len(d["prompt_token_ids"]), "ids": ids, "scores": score(url, ids)})
        gen = ref[-1]["scores"][len(d["prompt_token_ids"]) - 1:]
        print(f"collected {len(ids)} ids ({len(gen)} generated); generated-token rank-1 rate under teacher forcing: "
              f"{sum(s['rank'] == 1 for s in gen) / len(gen):.3f}", flush=True)
    json.dump(ref, open(path, "w"))


def compare(url: str, path: str) -> None:
    ref = json.load(open(path))
    n = same_top1 = ref_rank1 = 0
    gaps = []
    for item in ref:
        cand = score(url, item["ids"])
        for a, b in list(zip(item["scores"], cand))[item["n_prompt"] - 1:]:  # generated region only
            n += 1
            same_top1 += a["top1"] == b["top1"]
            ref_rank1 += b["rank"] == 1
            if a["lp"] is not None and b["lp"] is not None:
                gaps.append(abs(a["lp"] - b["lp"]))
    gaps.sort()
    print(f"positions {n} | same top-1 token {same_top1 / n:.4f} | reference token is rank 1 on candidate {ref_rank1 / n:.4f}")
    print(f"|logprob gap| median {statistics.median(gaps):.5f}  p95 {gaps[int(len(gaps) * 0.95)]:.5f}  max {gaps[-1]:.4f}")


def nll(url: str) -> None:
    ids = post(url, "/tokenize", {"prompt": PASSAGE, "add_special_tokens": False})["tokens"]
    s = [x["lp"] for x in score(url, ids) if x["lp"] is not None]
    print(f"passage tokens {len(ids)} | mean NLL {-sum(s) / len(s):.3f} nats/token | top-1 hit rate "
          f"{sum(x['rank'] == 1 for x in score(url, ids)) / (len(ids) - 1):.3f}")


if __name__ == "__main__":
    {"collect": collect, "compare": compare, "nll": nll}[sys.argv[1]](*sys.argv[2:])
