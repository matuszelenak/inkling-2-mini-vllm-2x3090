#!/usr/bin/env bash
# Run a command inside the unpacked vllm/vllm-openai:inkling-2-mini userland.
# The image does not fit in this box's Docker root, so its filesystem lives in
# ./rootfs and is bind-mounted over a small CUDA base image that Docker has.
# Usage: vrun.sh <gpus e.g. '"device=1"'> [docker run flags ... --] <command...>
# Expects ./rootfs next to this script (tools/fetch_rootfs.sh, then
# install_sm86_kernels.py-equivalent swap and apply_sm86_patch.py on
# rootfs/usr/local/lib/python3.14/dist-packages). ./work is mounted at /work.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); R=$HERE/rootfs
gpus=$1; shift
# Docker's CDI injection mounts the driver's user-space files under /usr before
# the /usr bind lands on top and hides them, so bind the compute ones again
# explicitly (deeper destinations are mounted after /usr) and keep the SONAME
# symlinks in the rootfs (see setup_driver_links below).
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nv=(-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro)
for l in libcuda libnvidia-ml libnvidia-ptxjitcompiler libnvidia-nvvm libnvidia-gpucomp libnvidia-allocator libnvidia-cfg; do
  nv+=(-v "/usr/lib/$l.so.$DRV:/usr/lib/$l.so.$DRV:ro")
  for link in "$l.so.1" "$l.so"; do [ -e "$R/usr/lib/$link" ] || [ -L "$R/usr/lib/$link" ] || ln -s "$l.so.$DRV" "$R/usr/lib/$link"; done
done
nv+=(-v /usr/lib/libnvidia-nvvm70.so.4:/usr/lib/libnvidia-nvvm70.so.4:ro)
[ -L "$R/usr/lib/libnvidia-nvvm.so.4" ] || ln -s "libnvidia-nvvm.so.$DRV" "$R/usr/lib/libnvidia-nvvm.so.4"
flags=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do flags+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
# VRUN_KEEP=1 keeps the stopped container so its logs survive a crash.
exec docker run ${VRUN_KEEP:+--log-opt max-size=50m} $([ -n "${VRUN_KEEP:-}" ] || echo --rm) --gpus "$gpus" --ipc host \
  -v "$R/usr":/usr -v "$R/etc":/etc -v "$R/opt":/opt -v "$R/var":/var \
  -v "$R/root":/root -v "$R/home":/home -v "$R/vllm-workspace":/vllm-workspace \
  -v "$HERE/cache/triton":/root/.triton -v "$HERE/cache/humming":/root/.cache/humming \
  -v "$HERE/cache/vllm":/root/.cache/vllm -v "$HERE/cache/flashinfer":/root/.cache/flashinfer \
  -v "$HERE/work":/work \
  -v "$(readlink -f "$HOME/.cache/huggingface")":/root/.cache/huggingface:ro \
  "${nv[@]}" \
  -e PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
  -e CUDA_HOME=/usr/local/cuda -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_USAGE_SOURCE=production-docker-image -e DO_NOT_TRACK=1 -e VLLM_NO_USAGE_STATS=1 \
  -w /vllm-workspace --entrypoint /usr/bin/env "${flags[@]}" \
  nvidia/cuda:12.8.1-runtime-ubuntu24.04 "$@"
