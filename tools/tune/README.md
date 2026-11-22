# tools/tune

What the tuning in `patches/sm86-tune.diff` was measured with. Results: [BENCH.md](../../BENCH.md#tuning).

| File | Runs | Measures |
|---|---|---|
| `bench_step.py` | against a server | decode tok/s per workload, tokens per step, **ms per step** (from the server's draft counters), acceptance by draft position. Seeded prompts: every server sees the same ones |
| `gate.py` | against a server | correctness: short prefill against the one-card reference, decode against prefill, long prefill against a reference collected on a known-good server, decode against prefill at depth |
| `bench_concurrency.py` | against a server | aggregate output tok/s at 1 to N concurrent requests |
| `published_protocols.py` | against a server | the benchmark protocols two other projects publish, re-implemented (same prompts, parameters and metrics), to read this server against their tables |
| `more_protocols.py` | against a server | the same for two more projects (lulakien's prefill and decode protocol, ingine's prompts) |
| `ksum.py` | on profiler traces | GPU kernel time per decode step or prefill chunk, by kernel |
| `mb_gemv.py` | inside the image, one free GPU | every small-matmul tile config that fits the card, per weight shape, against cuBLAS |
| `mb_humming.py` | inside the image, one free GPU | expert-GEMM kernel configs at the real layer shape, against the library's heuristic |
| `mb_align.py` | inside the image, one free GPU | the compiled expert-alignment op against the one-kernel version |

Step time is the number to tune on. It repeats to about 0.05 ms, while tok/s on one prompt swings by 10–30% between
repeats because it depends on how many drafted tokens that particular text lets through.

## Server-side

```bash
tools/tune/bench_step.py "label" /tmp/steps.jsonl                      # 5 workloads x 3 repeats, 512 tokens
tools/tune/gate.py collect-long http://localhost:8000 /tmp/long_ref.json <a source tree>   # on a known-good server
tools/tune/gate.py check http://localhost:8000 bench-results/tf_reference_tp1.json /tmp/long_ref.json
tools/tune/gate.py longdecode http://localhost:8000 /tmp/long_ref.json 768
tools/tune/bench_concurrency.py "label" /tmp/conc.jsonl --levels 1 2 4 8 16
```

To A/B one tuning, restart the server with `INKLING_SM86_TUNE` set to `none`, `all`, or a comma list of
`gemv, fold, seg3, ctrl, humming, humming_big, align, far, router, tiles, adaptive`.

## Microbenchmarks

They time kernels inside CUDA graphs, cycling through copies of the weights so every launch reads cold memory, as
in the model. Stop the server first: they need one card to themselves.

```bash
IMG=inkling-2-mini-vllm-2x3090:g7cb5dae22-tune1
run() { docker run --rm --gpus '"device=0"' --ipc host -v $PWD/tools/tune:/t "$@"; }
run --entrypoint python3 $IMG /t/mb_gemv.py
run -e INKLING_SM86_TUNE=none --entrypoint python3 $IMG /t/mb_humming.py 1 8 16      # tokens per step
run -e INKLING_SM86_TUNE=none -e MS=64,96,128 -e NS=128,256 -e KS=64,128 -e STS=2,3 --entrypoint python3 $IMG /t/mb_humming.py 2048
run --entrypoint python3 $IMG /t/mb_align.py
```

## Profiling

```bash
# add to the server command:  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/prof","torch_profiler_with_stack":false}'
curl -X POST localhost:8000/start_profile; <one request>; curl -X POST localhost:8000/stop_profile
tools/tune/ksum.py <decode steps or prefill chunks in the request> /prof/*rank0*.pt.trace.json.gz
```
