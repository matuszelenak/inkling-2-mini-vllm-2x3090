#!/usr/bin/env bash
# Start `vllm serve` for Inkling 2 Mini from the unpacked image and wait for
# health or exit. Usage: serve.sh <gpus> <host-port> <logfile> [vllm serve args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
gpus=$1; port=$2; log=$3; shift 3
C=vllm-inkling-exp
docker rm -f $C >/dev/null 2>&1
VRUN_KEEP=1 "$HERE/vrun.sh" "$gpus" -d --name $C -p "$port:8000" \
  -e NCCL_CUMEM_ENABLE=0 -e OMP_NUM_THREADS=1 -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  -- vllm serve thinkingmachines/Inkling-2-Mini --served-model-name inkling-2-mini "$@" >/dev/null
t0=$(date +%s); ok=1
until curl -sf -m 3 "http://localhost:$port/health" >/dev/null 2>&1; do
  [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" = "true" ] || { echo "CONTAINER EXITED after $(( $(date +%s) - t0 )) s"; ok=0; break; }
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo "TIMEOUT"; ok=0; break; }
  docker logs $C > "$log" 2>&1
  sleep 5
done
docker logs $C > "$log" 2>&1
echo "elapsed $(( $(date +%s) - t0 )) s | healthy=$ok | args: $*"
grep -E 'ERROR|Error|error:|Traceback|raise |OutOfResources|out of memory|OOM|GiB|KV cache|Maximum concurrency|all-reduce|All Reduce|backend|Humming|humming|took|Loading weights|capability|Starting vLLM|Application startup' "$log" | grep -vE 'DeprecationWarning|UserWarning' | sed -E 's/^\([A-Za-z_]+ pid=[0-9]+\) //' | cut -c1-230 | tail -45
echo "vram: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -2 | tr '\n' ';')"
[ $ok = 1 ]
