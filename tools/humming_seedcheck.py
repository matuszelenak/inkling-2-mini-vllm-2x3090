"""Re-run the [1-nvfp4] determinism case over several seeds: the upstream test
draws its expert ids at random and asserts at least one is local."""
import pathlib
import sys

import torch

# run from a directory that holds the widened upstream tests/ tree (see widen_test_gates.py)
sys.path.insert(0, str(pathlib.Path.cwd()))
from tests.models.inkling2.test_moe import HummingExperts, random_nvfp4
for seed in range(6):
    torch.manual_seed(seed)
    hidden, inter, num_experts, num_global = 4608, 2304, 96, 384
    gs = torch.full((num_experts,), 1 / 64, device="cuda")
    experts = HummingExperts({"w13": (*random_nvfp4(num_experts, 2 * inter, hidden), gs), "w2": (*random_nvfp4(num_experts, hidden, inter), gs)},
                             ((2 * inter, hidden), (hidden, inter)), True)
    ids = torch.randint(0, num_global, (1, 8), device="cuda", dtype=torch.int32)
    ids = torch.where(ids < num_experts, ids, -1)
    rows = torch.arange(8, device="cuda", dtype=torch.int32)
    src2dst = torch.where(ids.flatten() >= 0, rows, -1)
    weights = torch.rand(8, device="cuda"); x = torch.randn(1, hidden, device="cuda").bfloat16()
    local = src2dst >= 0
    if not local.any():
        print(f"seed {seed}: no local expert drawn (ids {ids.flatten().tolist()}) -> the upstream precondition assert fires"); continue
    outs = [experts(x, ids, weights, src2dst, num_global, 4.0, 25.0)[local] for _ in range(5)]
    same = all(torch.equal(o, outs[0]) for o in outs[1:])
    print(f"seed {seed}: local rows {int(local.sum())}, bitwise identical over 5 runs: {same}, finite: {bool(torch.isfinite(outs[0].float()).all())}, absmax {outs[0].float().abs().max().item():.3f}")
    del experts
