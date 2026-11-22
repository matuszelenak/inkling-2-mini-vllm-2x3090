"""Microbench the low-row GEMV tiles on this card, inside CUDA graphs with cold weights.

Run inside the image on one free GPU (see tools/tune/README.md):  mb_gemv.py ['[[m,n,k],...]' [out.jsonl]]

Each config is timed as one graph of R launches, each on a different copy of the weight (so the weights are
read from DRAM-side VRAM, not L2, as in the model where every layer has its own), replayed many times.
Baseline: cuBLAS through torch.nn.functional.linear, captured the same way.
"""
import itertools, json, sys, time
import torch, triton
from vllm.models.inkling2.nvidia.ops.gemv import _gemv_kernel

dev = "cuda"
SMEM = 96 * 1024
SHAPES = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
    (8, 512, 2304), (8, 2304, 1536), (8, 3280, 2304), (8, 4304, 2304),
    (8, 9216, 2304), (8, 2560, 2304), (8, 2304, 4608), (8, 1024, 2304), (8, 100512, 2304), (7, 100512, 256),
]
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def copies(n, k):
    return max(2, min(48, int(600e6 // (n * k * 2))))


def graph_time(fn, ws, x, reps):
    outs = [None] * len(ws)
    fn(x, ws[0]); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i, w in enumerate(ws):
            outs[i] = fn(x, w)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(ws) * 1e6, outs[0]


def gemv(bn, bk, warps, stages):
    def fn(x, w):
        m, k = x.shape; n = w.shape[0]
        out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
        _gemv_kernel[(triton.cdiv(n, bn),)](x, w, None, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                                             HAS_BIAS=False, BN=bn, BK=bk, MP=16, num_warps=warps, num_stages=stages)
        return out
    return fn


configs = [c for c in itertools.product((16, 32, 64, 128), (64, 128, 256, 512), (2, 4, 8), (2, 3, 4))
           if (c[3] - 1) * (c[0] + 16) * c[1] * 2 <= SMEM]
print(f"{len(configs)} configs, {torch.cuda.get_device_name(0)}", flush=True)
for m, n, k in SHAPES:
    r = copies(n, k)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(r)]
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    reps = 60 if n < 50000 else 12
    base, ref = graph_time(lambda a, w: torch.nn.functional.linear(a, w), ws, x, reps)
    rows = []
    for c in configs:
        if c[1] > k:
            continue
        try:
            us, out = graph_time(gemv(*c), ws, x, reps)
        except Exception as e:  # a config Triton cannot place on this card
            rows.append((float("inf"), c, type(e).__name__)); continue
        err = (out.float() - ref.float()).abs().max().item()
        rows.append((us, c, err))
    rows.sort(key=lambda t: t[0])
    best = rows[0]
    print(f"m={m} n={n} k={k} copies={r}: cuBLAS {base:.1f} us | best {best[0]:.1f} us {best[1]} ({base / best[0]:.2f}x) | "
          f"next {[(round(u, 1), c) for u, c, _ in rows[1:4]]} | max abs err vs cuBLAS {best[2]:.3g}", flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps({"m": m, "n": n, "k": k, "cublas_us": base, "rows": [(u, c) for u, c, e in rows if u != float("inf")][:40],
                                "failed": [(c, e) for u, c, e in rows if u == float("inf")]}) + "\n")
    del ws
    torch.cuda.empty_cache()
