#!/usr/bin/env python3
"""Sum GPU kernel time by kernel name from torch-profiler traces of the vLLM workers.
usage: ksum.py <steps> <trace> [<trace> ...]   (steps = decode steps inside the profiled span)"""
import collections, gzip, json, re, sys

steps = int(sys.argv[1])
for path in sys.argv[2:]:
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not k:
        print(path.split("/")[-1], ": no GPU kernel events"); continue
    tot = collections.defaultdict(float); n = collections.Counter()
    for e in k:
        s = re.sub(r"\(.*", "", re.sub(r"<.*", "", e["name"]))[:70]
        tot[s] += e["dur"]; n[s] += 1
    busy = sum(tot.values()); t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
    print(f"== {path.split('/')[-1][:60]}: {len(k)} GPU events, {len(k) / steps:.0f} per step; busy {busy / 1000 / steps:.2f} ms/step of {(t1 - t0) / 1000 / steps:.2f} ms/step span ({busy / (t1 - t0):.0%} busy)")
    print(f"{'kernel':70s} {'ms/step':>8s} {'calls/step':>10s} {'mean us':>8s} {'share':>6s}")
    for s, v in sorted(tot.items(), key=lambda x: -x[1])[:int(__import__('os').environ.get('TOP', 28))]:
        print(f"{s:70s} {v / 1000 / steps:8.3f} {n[s] / steps:10.1f} {v / n[s]:8.1f} {v / busy:6.1%}")
