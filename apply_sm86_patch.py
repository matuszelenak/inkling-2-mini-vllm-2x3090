#!/usr/bin/env python3
"""Steer vLLM's Inkling 2 code onto paths an Ampere (sm_86) GPU can run.

Targets the Python files of vllm/vllm-openai:inkling-2-mini (vLLM
0.1.dev25855+g7cb5dae22, Triton 3.8.0). Every edit is anchored on the exact
upstream text and the script fails if an anchor is missing or ambiguous, so
it cannot silently patch a different build. Idempotent: an already-patched
file is left alone. A pristine copy of each file is kept as <file>.orig.

Usage: apply_sm86_patch.py <site-packages dir> [--revert]
"""

import pathlib
import shutil
import sys

MARK = "# sm86-port"

PATCHES = [
    # 1. Capability floor. vllm/config/vllm.py refuses the "inkling2" quant
    #    method below get_min_capability() before any weights load.
    ("vllm/models/inkling2/quant_config.py",
     "    def get_min_capability(cls) -> int:\n        return 90\n",
     f"    def get_min_capability(cls) -> int:\n        return 80  {MARK}: was 90\n"),

    # 2. conv + mix. The fused csrc/inkling2 kernel is built for sm_90/100/103
    #    only (mbarrier, bulk async copies). SM12x already falls back to the
    #    Triton short_conv + attn_res pair; send sm_8x the same way.
    ("vllm/models/inkling2/nvidia/ops/conv_mix.py",
     "        if current_platform.is_device_capability_family(120):\n",
     "        if current_platform.is_device_capability_family(120) or "
     f"current_platform.is_device_capability_family(80):  {MARK}\n"),

    # 3. Router GEMM. The TMA kernel needs Hopper tensor descriptors and
    #    ~197 KB of shared memory; SM12x skips it and caps num_stages for its
    #    101 KB. sm_86 has 99 KB, so treat it like SM12x here.
    ("vllm/models/inkling2/nvidia/ops/linear.py",
     "    sm120 = current_platform.is_device_capability_family(120)\n",
     "    sm120 = current_platform.is_device_capability_family(120) or "
     f"current_platform.is_device_capability_family(80)  {MARK}\n"),

    # 5. Vision tower. Off SM100 it uses FlashAttention 3, which is Hopper-only;
    #    SM12x already takes FlashAttention 2. Same for sm_8x.
    ("vllm/models/inkling2/nvidia/vision.py",
     "            version = 2 if current_platform.is_device_capability_family(120) else 3\n",
     "            version = 2 if (current_platform.is_device_capability_family(120) or "
     f"current_platform.is_device_capability_family(80)) else 3  {MARK}\n"),

    # 4a. Programmatic dependent launch. griddepcontrol is sm_90+ PTX and
    #     Triton emits it unconditionally; below sm_90 emit nothing. Without
    #     PDL a kernel only starts after the previous one on the stream has
    #     finished, which is exactly what gdc_wait would have waited for.
    ("triton/language/extra/cuda/gdc.py",
     "from triton.language import core\n",
     "from triton.language import core\n"
     "\n\n"
     f"def _pre_sm90():  {MARK}\n"
     "    import torch\n"
     "    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9\n"),
    ("triton/language/extra/cuda/gdc.py",
     '    core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0",',
     f"    if _pre_sm90():  {MARK}\n        return\n"
     '    core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0",'),
    ("triton/language/extra/cuda/gdc.py",
     '    core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0",',
     f"    if _pre_sm90():  {MARK}\n        return\n"
     '    core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0",'),

    # 4b. ...and never ask the driver for a PDL launch below sm_90.
    ("triton/backends/nvidia/compiler.py",
     '        capability = int(self._parse_arch(args["arch"]))\n\n        if args.get("num_ctas", 1) > 1 and capability < 90:\n',
     '        capability = int(self._parse_arch(args["arch"]))\n'
     f"        if capability < 90:  {MARK}\n"
     '            args["launch_pdl"] = False\n'
     '\n        if args.get("num_ctas", 1) > 1 and capability < 90:\n'),
]


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    revert = "--revert" in sys.argv
    files = sorted({rel for rel, _, _ in PATCHES})
    if revert:
        for rel in files:
            orig = root / (rel + ".orig")
            if orig.exists():
                shutil.copy2(orig, root / rel)
                print("reverted", rel)
        return
    for rel in files:
        path = root / rel
        text = path.read_text()
        if MARK in text:
            print("already patched:", rel)
            continue
        new = text
        for prel, old, repl in PATCHES:
            if prel != rel:
                continue
            if new.count(old) != 1:
                sys.exit(f"ANCHOR MISMATCH in {rel}: expected exactly one occurrence of {old!r}, found {new.count(old)}")
            new = new.replace(old, repl)
        compile(new, str(path), "exec")
        orig = root / (rel + ".orig")
        if not orig.exists():
            shutil.copy2(path, orig)
        path.write_text(new)
        print("patched:", rel)


if __name__ == "__main__":
    main()
