"""Stage 0: can this image's kernels run on an sm_86 card at all?

Run inside the image on one 3090, e.g.
  docker run --rm --gpus '"device=1"' -v $PWD/tools:/work --entrypoint python3 <image> /work/preflight.py
"""
import torch, sys
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), cap, "| arch list", torch.cuda.get_arch_list())
x = torch.randn(64, 2304, device="cuda", dtype=torch.bfloat16)
print("bf16 gemm:", (x @ x.T).float().abs().mean().item() > 0)

# 1. generic vLLM C extension kernel (is _C built for this arch?)
import vllm._C_stable_libtorch  # noqa
ops = [o for o in dir(torch.ops._C) if not o.startswith("_")]
print("torch.ops._C ops:", len(ops), "| inkling2:", [o for o in ops if "inkling" in o])
try:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    ids = torch.randint(0, 16, (64, 8), device="cuda", dtype=torch.int32)
    out = moe_align_block_size(ids, 64, 16)
    print("moe_align_block_size OK:", [tuple(t.shape) for t in out])
except Exception as e:  # noqa
    print("moe_align_block_size FAILED:", type(e).__name__, str(e)[:300])

# 2. Triton: gdc_wait + launch_pdl below sm_90 (patched: must compile and run)
import triton, triton.language as tl
@triton.jit
def k(X, N, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs, offs < N) * 2.0, offs < N)
    tl.extra.cuda.gdc_launch_dependents()
a = torch.ones(1000, device="cuda")
try:
    k[(triton.cdiv(1000, 256),)](a, 1000, 256, launch_pdl=True)
    torch.cuda.synchronize()
    print("triton gdc + launch_pdl kernel OK:", a.sum().item() == 2000.0)
except Exception as e:  # noqa
    print("triton gdc kernel FAILED:", type(e).__name__, str(e)[:400])

# 3. platform helpers used by the patches
from vllm.platforms import current_platform
print("family(80):", current_platform.is_device_capability_family(80), "| family(120):", current_platform.is_device_capability_family(120))
from vllm.models.inkling2.nvidia.ops import use_sm100_kernels
print("use_sm100_kernels:", use_sm100_kernels())
from vllm.models.inkling2.quant_config import Inkling2QuantConfig
print("min capability:", Inkling2QuantConfig.get_min_capability())
