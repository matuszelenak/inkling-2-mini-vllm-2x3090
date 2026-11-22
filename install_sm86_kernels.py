#!/usr/bin/env python3
"""Replace the image's Blackwell-only compiled kernels with general-arch ones.

vllm/vllm-openai:inkling-2-mini was built with TORCH_CUDA_ARCH_LIST="10.0+PTX
10.3+PTX": its _C and _moe_C extensions hold no code an sm_86 card can run
("no kernel image is available for execution on the device" on the first
call). The per-commit wheel of the main commit this branch sits on is built
for every architecture and registers the same operators; the only schema
differences are the two fused inkling2 kernels the Ampere path never calls and
one optional trailing argument.

Usage: install_sm86_kernels.py <site-packages dir>
"""

import hashlib
import importlib.metadata
import pathlib
import shutil
import sys
import urllib.request
import zipfile

EXPECTED_VLLM = "0.1.dev25855+g7cb5dae22.cu130"
WHEEL_URL = (
    "https://wheels.vllm.ai/bf4a317ee65dba1198e420be5253334f23f2fbdc/"
    "vllm-0.34.0rc1.dev1104%2Bgbf4a317ee-cp38-abi3-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "ae243260269979c5fe5582d8f49a74a778d1c13d699812d3c5d666aee7c663a6"
LIBS = ("vllm/_C_stable_libtorch.abi3.so", "vllm/_moe_C_stable_libtorch.abi3.so")


def main() -> None:
    site = pathlib.Path(sys.argv[1])
    installed = importlib.metadata.version("vllm")
    if installed != EXPECTED_VLLM:
        sys.exit(f"vllm {installed} is not the build these kernels were matched against ({EXPECTED_VLLM})")
    wheel = pathlib.Path("/tmp/vllm-general-arch.whl")
    urllib.request.urlretrieve(WHEEL_URL, wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        sys.exit(f"wheel sha256 {digest} != {WHEEL_SHA256}")
    with zipfile.ZipFile(wheel) as z:
        for lib in LIBS:
            target = site / lib
            if not target.exists():
                sys.exit(f"missing {target}")
            shutil.copy2(target, target.with_name(target.name + ".orig"))
            with z.open(lib) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"replaced {lib}: {target.stat().st_size} bytes")
    wheel.unlink()


if __name__ == "__main__":
    main()
