"""Compare the compiled moe_align_block_size op with the one-kernel version used for single-stream decode,
inside CUDA graphs, and check where their outputs are identical. Run inside the image on one free GPU."""
import time

import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.models.inkling2.nvidia.ops.moe_align_small import MAX_ROWS, moe_align_small

E, G, TOPK, BS = 64, 128, 8, 8
torch.manual_seed(0)


def ids_for(tokens):
    ids = torch.stack([torch.randperm(G, device="cuda")[:TOPK] for _ in range(tokens)]).to(torch.int32)
    return torch.where(ids < E, ids, -1)


def timed(fn, sets, reps=200):
    outs = [fn(s) for s in sets]; torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(s) for s in sets]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(sets) * 1e6, outs


for tokens in range(1, MAX_ROWS // TOPK + 1):
    sets = [ids_for(tokens) for _ in range(32)]
    a_us, a = timed(lambda s: moe_align_block_size(s, BS, E, ignore_invalid_experts=True)[:3], sets)
    b_us, b = timed(lambda s: moe_align_small(s, BS, E), sets)
    same = True
    for (sa, ea, na), (sb, eb, nb) in zip(a, b):
        n = int(na.item()); same &= int(nb.item()) == n
        same &= torch.equal(sa[:n].to(torch.int32), sb[:n].to(torch.int32))
        same &= torch.equal(ea[: n // BS].to(torch.int32), eb[: n // BS].to(torch.int32))
    print(f"tokens={tokens}: compiled {a_us:5.2f} us | one-kernel {b_us:5.2f} us | {a_us / b_us:.2f}x | identical output: {same}", flush=True)
