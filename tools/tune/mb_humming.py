"""Microbench the expert GEMM kernel configs on this card: one rank of the real layer shape
(64 local experts of 128, hidden 2304, intermediate 1152, NVFP4), timed inside CUDA graphs.
usage: mb_humming.py [tokens ...]     env: MODE=dump | sweep (default), FMT=nvfp4|mxfp8, MS / NS / KS / STS = candidate
block rows / block N / block K / stages (comma lists), OUT = jsonl to append to. Run with INKLING_SM86_TUNE=none so the
baseline is Humming's own heuristic (see tools/tune/README.md)."""
import copy, itertools, json, os, sys, time
import torch
from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts

torch.manual_seed(0)
FMT = os.environ.get("FMT", "nvfp4"); nvfp4 = FMT == "nvfp4"
H, I, E, G, TOPK = 2304, 1152, 64, 128, 8


def random_nvfp4(*shape):
    values = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device="cuda")
    scales = torch.rand(*shape[:-1], shape[-1] // 16, device="cuda") + 0.5
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn)


def random_mxfp8(*shape):
    values = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(120, 128, (*shape[:-1], shape[-1] // 32), device="cuda")
    return values, scales.to(torch.uint8)


rq = random_nvfp4 if nvfp4 else random_mxfp8
gs = torch.full((E,), 1 / 64, device="cuda") if nvfp4 else None
experts = HummingExperts({"w13": (*rq(E, 2 * I, H), gs), "w2": (*rq(E, H, I), gs)}, ((2 * I, H), (H, I)), nvfp4)
BASE = copy.deepcopy(experts.tunings)
if os.environ.get("MODE") == "dump":
    for name in ("w13", "w2"):
        print(name, json.dumps(BASE[name]))
    sys.exit()


def inputs(tokens, n=6):
    sets = []
    for _ in range(n):
        ids = torch.stack([torch.randperm(G, device="cuda")[:TOPK] for _ in range(tokens)]).to(torch.int32)
        ids = torch.where(ids < E, ids, -1)
        rows = torch.arange(tokens * TOPK, device="cuda", dtype=torch.int32)
        src2dst = torch.where(ids.flatten() >= 0, rows, -1)
        sets.append((torch.randn(tokens, H, device="cuda").bfloat16(), ids, torch.rand(tokens * TOPK, device="cuda"), src2dst))
    return sets


def run(s):
    return experts(s[0], s[1], s[2], s[3], G, 4.0, 25.0)


def timed(sets, reps=40):
    outs = [run(s) for s in sets]; torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [run(s) for s in sets]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / reps / len(sets) * 1e6
    local = sets[0][3] >= 0
    return us, outs[0][local].float().clone()


def apply(cfg, rows):
    for name in ("w13", "w2"):
        t = copy.deepcopy(BASE[name])
        for lower, upper, kc in t:
            if lower < rows <= upper:
                kc.update(cfg)
        experts.tunings[name] = t; experts.tuning_strs[name] = json.dumps(t)


def reset():
    for name in ("w13", "w2"):
        experts.tunings[name] = copy.deepcopy(BASE[name]); experts.tuning_strs[name] = json.dumps(BASE[name])


cands = []
for m, n, k, st, ctas in itertools.product([int(v) for v in os.environ.get('MS', '8').split(',')], [int(v) for v in os.environ.get('NS', '64,128,256').split(',')], [int(v) for v in os.environ.get('KS', '32,64,128,256').split(',')], [int(v) for v in os.environ.get('STS', '2,3,4').split(',')], (1, 2)):
    for wn in (32, 64):
        if wn > n:
            continue
        for wm in ((m,) if m < 32 else (m, m // 2)):
            cands.append({"block_shape": [m, n, k], "warp_shape": [wm, wn, min(k, 64)], "num_stages": st, "num_ctas_per_sm": ctas})
print(f"{len(cands)} candidates, {torch.cuda.get_device_name(0)}, {FMT}", flush=True)
for tokens in [int(a) for a in sys.argv[1:]] or [8]:
    rows = tokens * TOPK; valid = -(-rows * E // G)
    sets = inputs(tokens)
    reset(); base_us, ref = timed(sets)
    cur = {n_: [kc for lo, up, kc in BASE[n_] if lo < valid <= up] for n_ in ("w13", "w2")}
    print(f"tokens={tokens} rows={rows} local~{valid}: heuristic {base_us:.1f} us/call | w13 {json.dumps(cur['w13'])[:230]}", flush=True)
    res = []
    t_start = time.time()
    for i, c in enumerate(cands):
        if i == 10:
            print(f'  first 10 candidates took {time.time() - t_start:.0f} s', flush=True)
        apply(c, valid)
        try:
            us, out = timed(sets, reps=25)
            err = (out - ref).abs().max().item()
            res.append((us, c, err))
        except Exception as e:
            res.append((float("inf"), c, f"{type(e).__name__}: {str(e)[:60]}"))
    ok = sorted([r for r in res if r[0] != float("inf")], key=lambda r: r[0])
    print(f"  {len(ok)} of {len(res)} candidates run on this card; failures e.g. {next((r[2] for r in res if r[0] == float('inf')), None)}")
    for us, c, err in ok[:8]:
        print(f"  {us:7.1f} us ({base_us / us:.2f}x)  block {c['block_shape']} warp {c['warp_shape']} stages {c['num_stages']} ctas {c['num_ctas_per_sm']}  max|d| vs heuristic {err:.3g}", flush=True)
    if os.environ.get("OUT"):
        with open(os.environ["OUT"], "a") as f:
            f.write(json.dumps({"tokens": tokens, "heuristic_us": base_us, "heuristic": cur, "top": [(u, c, e) for u, c, e in ok[:40]]}) + "\n")
