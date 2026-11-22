# Inkling 2 Mini on vLLM, 2× RTX 3090 — measurements

2026-11-21. 2× RTX 3090 at 300 W, NVLink, TP2 + EP2, bf16 KV, `--max-num-batched-tokens 2048`, `--gpu-memory-utilization 0.92`. vLLM `0.1.dev25855+g7cb5dae22` (the `inkling-2-mini` image) with the kernel swap and patches of this repo. Rows labelled `packaged image …` ran on the image built from the Dockerfile; all others ran on the same files from an unpacked copy of the image (`tools/vrun.sh`), and the built image reproduces their token-agreement figures exactly. Data and column definitions: [bench-results/](bench-results/README.md).

llama.cpp columns: the same cards, prompts and day. llama.cpp at `690b87d` (ggml-org/llama.cpp#32694 + #32704) plus one local patch for loop-slot placement, Unsloth's `UD-Q8_K_XL` GGUF, layer split across both cards, f16 KV, 131,072 window, no drafter. That config and its CSVs are not part of this repo: `bench-results/` holds the vLLM rows only.

## Tuning

2026-11-21, evening. Same cards and settings as below, with `patches/sm86-tune.diff` (the image `inkling-2-mini-vllm-2x3090:g7cb5dae22-tune1`, booted from `docker-compose.yml` as shipped unless a row says otherwise: drafter on, vision loaded, `--max-num-seqs 8`). "Untuned" is the port as measured earlier the same day; `INKLING_SM86_TUNE=none` reproduces it. "Tuned" columns are the final image. The change-by-change tables were measured as each change went in, so their absolute figures predate the last three rows of the step table where noted. Data: `bench-results/tuning/`; the same suite on the build before those three rows is in `tuning/before-router-and-tiles/`. Tools: `tools/tune/`.

### Decode, one request

`tuning/runs.csv` (tuned) and `runs.csv` (untuned, packaged image). 512 tokens generated.

| Workload | Untuned, no drafter | Tuned, no drafter | Untuned, drafter | Tuned, drafter |
|---|---|---|---|---|
| code, greedy | 122.6 | 143.5 | 175.1 | 208.0 |
| code, temperature 1 | 122.7 | 143.4 | 162.8 | 201.4 |
| prose, greedy | 122.6 | 143.2 | 160.7 | 175.1 |
| JSON tool calls, greedy | 123.0 | 145.0 | 225.7 | 228.7 |

One run per cell. With the drafter on a single run moves with the text: the JSON request stopped after 344 tokens here and gave 280.8 on the previous build. Medians of three 512-token runs on the final image (`tuning/steps_final.jsonl`):

| Workload | Greedy tok/s | Tokens per step | ms per step | Temperature 0.6 tok/s |
|---|---|---|---|---|
| code | 223 | 2.68 | 12.2 | 243 |
| prose | 170 | 2.00 | 11.6 | 179 |
| JSON | 306 | 3.85 | 12.7 | 294 |
| chat | 200 | 2.40 | 11.9 | 200 |
| code, temperature 1 | 224 | 2.74 | 12.3 | |

### Depth

Five planted facts per prompt, all found at every depth in both builds. Decode is without the drafter.

| Prompt tokens | Prefill tok/s, untuned | Tuned | Decode tok/s, untuned | Tuned | Time to first token, tuned |
|---|---|---|---|---|---|
| 8,173 | 6,864 | 7,028 | 120.1 | 139.0 | 1.2 s |
| 24,173 | 6,467 | 6,695 | 115.3 | 133.8 | 3.6 s |
| 32,176 | 6,303 | 6,546 | 113.2 | 130.6 | 4.9 s |
| 48,177 | 5,949 | 6,240 | 109.5 | 125.9 | 7.7 s |
| 64,176 | 5,653 | 6,001 | 105.6 | 121.4 | 10.7 s |
| 120,176 | 4,753 | 5,170 | 94.3 | 106.9 | 23.3 s |

Tuned prefill is as shipped (drafter loaded, `--max-num-seqs 8`). With the drafter off the same image gave 7,078 / 6,747 / 6,621 / 6,310 / 6,047 / 5,212. Prefill repeats to about ±0.5% between runs and boots; both cards draw their 300 W cap throughout.

### Long windows

`tuning/long_window.csv`. Text only (`--limit-mm-per-prompt '{"image":0}'`), bf16 KV, `--max-num-seqs 4`, one boot and one fill per row.

| Mode | Utilization | KV pool | Window | Prompt tokens | Facts found | Prefill tok/s | To first token | Decode tok/s |
|---|---|---|---|---|---|---|---|---|
| Drafter | 0.95 | 286,611 | 262,144 | 250,173 | 5 / 5 | 3,924 | 63.8 s | 257.9 (162 tokens of a predictable answer) |
| No drafter | 0.92 | 370,051 | 294,912 | 200,172 | 5 / 5 | 4,341 | 46.1 s | 92.3 |
| No drafter | 0.92 | 370,051 | 294,912 | 280,173 | 5 / 5 | 3,709 | 75.5 s | 81.0 |
| No drafter | 0.92 | 404,949 | 368,640 | 360,174 | 5 / 5 | 3,305 | 109.0 s | 72.1 |

No server errors in any of them. Memory in use read 20.2 GiB per card after each fill. At utilization 0.97 without the drafter, windows of 458,752 and 491,520 do not boot: 10.41 GiB of KV is needed for 458,752 against 9.92 GiB available, and the server estimates a maximum of 426,880. That window was not booted or filled.

### Step time, change by change

`tuning/steps_by_arm.jsonl`, `tools/tune/bench_step.py`. A step is one target forward; with the drafter it verifies 7 drafted tokens. Step time repeats to about 0.05 ms across prompts and repeats, so it is the column to compare; tok/s on a single prompt swings 10–30% with how many drafts that text lets through.

| Build | ms per step, 7 drafts | ms per step, no drafter |
|---|---|---|
| Untuned port | 14.9 | 8.1 |
| + small-matmul kernel (tiles fitted to sm_86) | 14.35 | |
| + drafter fold on it | 14.25 | |
| fused q/k/v launch alone | 14.5 | |
| loop-controller kernels alone | 14.3 | |
| the four together | 13.45 | 7.8 |
| + expert-GEMM decode config | 13.2 | 7.4 |
| + one-kernel expert alignment | 12.95 | 7.1 |
| + router matmul tile 16 wide, 2 warps, 3 stages | | 6.99 |
| + attention decode tile of 64 keys | | 6.95 |

The last two rows were measured without the drafter only (`tuning/steps_router_and_decode_tiles.jsonl`). With the draft-length policy on, the final image's mean step is 11.6–11.7 ms on prose and 11.9–12.2 ms on chat.

### Where a decode step goes

Torch profiler, rank 0, 47 steps with 7 drafts (`tools/tune/ksum.py`). ms per step.

| Kernels | Untuned | Tuned (before the alignment kernel) | Tuned, no drafter |
|---|---|---|---|
| Expert GEMMs (Humming) | 4.9 | 4.4 | 1.6 |
| Dense matmuls (cuBLAS, then the small-matmul kernel) | 6.0 | 4.0 | 2.2 |
| All-reduce over NVLink (vLLM's kernel) | 0.9 | 0.9 | 0.6 |
| Expert alignment (compiled op) | 0.5 | 0.5 | 0.4 |
| Logits all-gather (NCCL) | 0.4 | 0.4 | 0.04 |
| About 700 small Triton kernels (after tuning, including the loop-controller kernels) | 2.5 | 3.0 | 2.7 |

Expert GEMMs and dense matmuls run near the cards' memory bandwidth after tuning. Prefill, per 2,048-token chunk on the untuned port: expert GEMMs 41%, cuBLAS 15%, NCCL all-reduce 11.5%, attention 11%.

### Kernel microbenchmarks on one RTX 3090

Timed inside CUDA graphs, each launch on a different copy of the weights.

Small-matmul kernel, 8 rows, 114 tile configs that fit 96 KB of shared memory (`tuning/mb_gemv.jsonl`):

| Weight (rows × columns) | cuBLAS µs | Best tile µs | Speed-up |
|---|---|---|---|
| 512 × 2304 (k, v projections) | 8.1 | 5.6 | 1.46× |
| 2304 × 1536 (output projection) | 12.8 | 9.8 | 1.31× |
| 3280 × 2304 (q side, window layers) | 23.9 | 19.2 | 1.24× |
| 4304 × 2304 (q side, full layers) | 27.8 | 24.6 | 1.13× |
| 9216 × 2304 (drafter) | 57.2 | 50.3 | 1.14× |
| 2304 × 4608 (drafter) | 35.1 | 26.0 | 1.35× |
| 100512 × 2304 (vocabulary head, per card) | 559.3 | 528.1 | 1.06× |

The tiles the other repo chose on its sm_80 card are within 1–2% of these, except the vocabulary-head tile, which needs 120 KB of shared memory and is cut to 3 stages here.

Expert layer (64 local experts of 128, real shapes), 144 kernel configs (`tuning/mb_humming_*.jsonl`):

| Tokens per step | Format | Heuristic µs | Best µs | Speed-up | Config adopted |
|---|---|---|---|---|---|
| 1 | NVFP4 | 54.9 | 45.5 | 1.21× | block 8×128×64, warp 8×32×64, 4 stages, 2 instances per SM |
| 8 | NVFP4 | 167.7 | 156.7 | 1.07× | same |
| 16 | NVFP4 | 238.8 | 222.6 | 1.07× | same |
| 8 | MXFP8 | 275.9 | 265.9 | 1.04× | same |
| 2,048 (prefill) | NVFP4 | 2,634.6 | 2,463.4 | 1.07× | heuristic block 64×256×64 with warp 64×64×64 and 2 instances per SM; identical output |

The library's own sm_86 heuristic already picks the block shape the other repo found by sweeping on sm_80; that repo's override (4 stages, 2 instances) does not launch on sm_86.

Expert alignment: compiled op 9.9–10.0 µs; one-kernel version 2.7 µs at 1–2 tokens and 4.0 µs at 5–8, with identical output up to 8 tokens (64 routed rows). Above that it orders rows differently inside an expert's block, so the compiled op is kept there.

### Draft length

`tuning/steps_by_arm.jsonl`, measured before the router and attention-tile rows (the no-drafter step is 6.95 ms and 143 tok/s on the final image). Temperature 0.6, default reasoning effort, tok/s medians over 3–4 repeats of 512 tokens. "Drafted" changes how many tokens the drafter produces; the last two columns keep 7 drafted and change how many are verified.

| Draft tokens | ms per step, drafted = verified | Code | Prose | JSON | Chat | ms per step, 7 drafted, this many verified |
|---|---|---|---|---|---|---|
| none | 7.1 | 140 | 140 | 140 | 140 | |
| 1 | 9.3 | 186 | 166 | 197 | 180 | 10.05 |
| 2 | 10.1 | 228 | 177 | 245 | 221 | 10.75 |
| 3 | 10.8 | 244 | 184 | 274 | 224 | 11.25 |
| 4 | 11.35 | 239 | 182 | 330 | 216 | 11.75 |
| 5 | 12.0 | 282 | 173 | 329 | 206 | 12.2 |
| 6 | 12.5 | 281 | 176 | 286 | 201 | |
| 7 | 13.0 | 241 | 162 | 311 | 190 | 13.0 |

- Acceptance by draft position at 7 drafts: code 0.72 / 0.49 / 0.33 / 0.23 / 0.18 / 0.12 / 0.07; prose 0.53 / 0.27 / 0.14 / 0.08 / 0.05 / 0.03 / 0.02; JSON 0.84 / 0.67 / 0.47 / 0.38 / 0.29 / 0.22 / 0.17. Over a day of mixed use on the untuned server (300K steps): 0.82 / 0.65 / 0.54 / 0.45 / 0.39 / 0.34 / 0.29.
- No fixed length is best on all of them, so the length is chosen per step (`vllm/v1/spec_decode/dynamic/adaptive_k.py` in the patch): two running estimates, how often the first draft survives and how often each later one does given the one before, and the verified count that maximises expected tokens per millisecond for the last column's costs.
- Policy on against a fixed 7, same prompts: prose 182 vs 161, chat 202 vs 190, code 239 vs 241, JSON 304 vs 311.
- Block verification (`rejection_sample_method: block`) at temperature 0.6: 2.80 / 2.29 / 3.91 / 2.95 tokens per step on code / prose / JSON / chat against 3.14 / 2.13 / 4.03 / 2.49 with standard verification. No gain; not used.

### Several requests

`tuning/concurrency_by_draft_length.jsonl`, `tools/tune/bench_concurrency.py`: every request a different prompt (code, prose and JSON mixed), 512 tokens out, temperature 0.6, low reasoning effort, `--max-num-seqs 16`. Aggregate tok/s, 7 tokens drafted.

| Verified drafts | 1 request | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 7 | 280 | 292 | 453 | 645 | 924 |
| 5 | 273 | 302 | 476 | 727 | 1,040 |
| 4 | 261 | 318 | 457 | 752 | 1,050 |
| 3 | 253 | 338 | 507 | 738 | 1,120 |
| 2 | 217 | 298 | 496 | 715 | 1,106 |
| 1 | 177 | 280 | 464 | 688 | 1,012 |

The compose file sets 7 for one request and 3 for two or more. Final image as shipped (`--max-num-seqs 8`, `tuning/concurrency_final.jsonl`): 253 / 336 / 509 / 747 tok/s at 1 / 2 / 4 / 8. The build before the router and tile rows gave 232 / 323 / 508 / 740, and 231 / 323 / 508 / 725 after a restart; by kind at 8 requests on that build: JSON 856, code 762, prose 726.

### The benchmark protocols of two other projects

`tools/tune/published_protocols.py` re-implements both (their prompts, request parameters and metrics; none of their code). `tuning/published_protocols-*.json`. "Theirs" is what each project publishes for its own hardware.

agent2x0r/inkling2-sm86-kernels: 2× RTX 3090, NVLink, 225 W, original weights. 512 tokens out, temperature 0.6, low reasoning effort, 3 trials.

| | Theirs | Here |
|---|---|---|
| Prefill tok/s at 1K / 4K / 16K / 32K / 60K | 4,918 / 5,575 / 5,709 / 5,439 / 4,954 | 5,896 / 6,572 / 6,775 / 6,526 / 5,999 |
| Decode without the drafter, same depths | 114.7 / 113.6 / 110.0 / 105.6 / 99.4 | 139.8 / 138.5 / 133.8 / 128.4 / 120.3 |
| Decode with the drafter, same depths | 172.2 / 166.0 / 178.3 / 177.9 / 170.2 | 216.4 / 227.7 / 216.0 / 210.7 / 199.6 |
| Code / math / prose / JSON, drafter on | 202.4 / 177.6 / 135.5 / 209.9 | 228.9 / 224.8 / 180.2 / 250.6 |
| Code / math / prose / JSON, drafter off | 115.6 / 115.8 / 115.5 / 115.7 | 142.3 / 142.5 / 142.1 / 142.3 |
| Aggregate at 1 / 2 / 4 / 8 users, drafter on | 190.6 / 277.9 / 352.6 / 383.4 | 260.0 / 322.4 / 422.2 / 482.1 |
| Aggregate at 1 / 2 / 4 / 8 users, drafter off | 113.5 / 173.9 / 249.0 / 328.1 | 138.6 / 233.0 / 316.3 / 406.4 |

Planted tag found in 3 of 3 trials at every depth, prefix-cache hits 0.

Pento95/HyperInkling: 1× RTX 3090 Ti, weights re-quantized to INT4. 768 tokens out, temperature 0.2 (structured) or 0.5, default reasoning effort, two prompts per category, one run. Its `doc` category is left out (the document is not in its repo).

| | Theirs | Here |
|---|---|---|
| Chat, drafter off / on | 136 / about 145 | 143.1 / 146.2 |
| Code | 135 / about 230 | 141.5 / 268.1 |
| Structured output | 135 / about 310 | 142.0 / 326.8 |
| Prose | 135 / about 145 | 141.7 / 144.4 |
| Six requests at once, aggregate, drafter off / on | | 388.5 / 518.2 |

- "Theirs" is what the project published on 2026-11-21. On 2026-11-22 it reports about 150 / 220 / 295 / 155 with the drafter on; its benchmark script has not changed since 2026-11-21, so those are on the same prompts.
- One run per cell, as in their protocol. The build before the router and tile rows gave 151.4 / 225.9 / 340.8 / 143.0 with the drafter on: the code cell moved by 19% between two runs of near-identical builds, which is the spread of a single drafter-on run.
- With a fixed 7 verified drafts chat gave 132 and prose 131, below the drafter-off figure. The draft-length policy brings them back level with it.

### Two more protocols

`tools/tune/more_protocols.py`, `tuning/more_protocols-drafter.json`. Re-implemented from the prompts and parameters the two projects publish, run against the final image as shipped (drafter on, vision loaded) on 2026-11-21 after their figures changed. "Theirs" is as published on 2026-11-22 02:50 UTC.

lulakien/inkling2-mini-3090-recipe: 2× RTX 3090 over PCIe at 280 W, original experts with INT8 attention, head and drafter weights, text-only server, the `gpu-unlock` kernel module loaded. Decode: 1,500 tokens out, not streamed, temperature 0, low reasoning effort, tok/s = completion tokens over wall time, three runs. Prefill: one-token `/v1/completions` over unique random token ids, tok/s = prompt tokens over wall time, three runs.

| | Theirs | Here (median) |
|---|---|---|
| Code | 313 | 339.7 |
| Explanation | 166 | 189.9 |
| Story | 126 | 153.4 |
| Aggregate at 2 / 4 / 8 streams, the three prompts in rotation | 260 / 339 / 445 | 324.1 / 397.4 / 673.7 |
| Prefill at 2,000 / 16,000 / 64,000 tokens | 7,480 / 7,140 / 6,280 | 7,405 / 7,031 / 6,094 |

Greedy and identical across runs: every repeat generated the same number of tokens. How it combines prompts for its stream rows is not published; the rotation is this repo's choice.

skyne98/inkling-ingine-3090: 1× RTX 3090 at 350 W, own engine, dense weights and KV in fp8. 512 tokens out, greedy, three runs, rate = (tokens − 1) over the time after the first token. Its reasoning-effort setting is not published.

| | Theirs | Here, default effort | Here, low effort |
|---|---|---|---|
| Chat prompt | 235–239 | 231.5–235.4 | 269.4–269.7 |
| Code prompt | 219–224 | 261.1–262.8 | 293.2–293.9 |

### Checks on the tuned image

| Check | Untuned port | Tuned image |
|---|---|---|
| Teacher-forced, 8 prompts × 192 tokens, against the one-card reference | 98.76% rank 1, median \|Δ logprob\| 0.00124, p95 0.0704 | 98.37% rank 1, median 0.00145, p95 0.0666 |
| Decode against prefill on the same server, 1,536 tokens | 98.63% | 99.02% |
| Decode against prefill at 12K depth, 2,304 tokens | 98.13% | 98.09% |
| 16K-token prompt, teacher-forced, against the untuned port | | 97.7% same top token, median \|Δ logprob\| 0.011, p95 0.20; NLL 1.4828 against 1.4831 |
| GSM8K, first 200, drafter on, concurrency 4 | 193 / 200, 591 tok/s | 196 / 200, 609 tok/s. The build before the router and tile rows: 196 / 200 twice |
| Retrieval of 5 planted facts, 8K to 120K | 5 / 5 at every depth | 5 / 5 at every depth |
| Images, 64×64 to 4096×1024 | correct | correct; 4096×4096 rejected with HTTP 400 as before |
| Tool call through the `inkling` parser | | name and JSON arguments correct |

- The first row and the 16K row move because two changes reorder floating-point work in prefill attention: the long-prompt loop of the full-attention layers and the 32-key prefill tile. The build with the loop but without the tile change returned the untuned port's first-row figures exactly, and 97.9% / 0.008 / 0.19 on the 16K row. The changes before those two leave both rows at exactly zero difference, and the router and decode tiles do not run on prompts of this size.
- GSM8K by question, tuned against untuned with the drafter off: both right 195, only tuned 1, only untuned 1.
- Memory: 4.2 GiB free per card after GSM8K and the concurrency sweep on a fresh boot; 1.3 GiB after the whole run on the final image (22.7 of 24.0 GiB used), most of the difference being the gate's 16K-token scoring requests. KV pool 225,883 tokens at `--max-num-seqs 8` (233,016 at 4).
- Upstream's unit tests were not re-run on the tuned tree.

### Tried, not adopted

| Change | Result |
|---|---|
| The other repo's expert-GEMM override (4 stages, 2 instances per SM) | does not launch on sm_86 |
| Its prefill attention tiles | 24K / 64K / 120K prefill 6,534 / 5,838 / 4,987 against 6,605 / 5,890 / 5,033 without them |
| 1,792-token prefill chunks, so every all-reduce fits vLLM's NVLink kernel (8 MiB limit) | prefill 10% slower at every depth. Not a clean test of the all-reduce: the smaller chunk also puts the expert GEMMs in a different kernel bucket |
| That kernel's limit raised to 16 MiB, chunks kept at 2,048 (`tuning/prefill_by_allreduce_limit.csv`) | 8K / 24K / 32K / 64K prefill 7,101–7,164 / 6,778 / 6,648 / 6,038 against 7,054–7,117 / 6,769 / 6,647 / 6,042: no change |
| `NCCL_BUFFSIZE` 16 MiB, `NCCL_P2P_NVL_CHUNKSIZE` 2 or 4 MiB, `NCCL_MIN_NCHANNELS` 8, alone and combined (`tuning/prefill_by_nccl_setting.csv`) | every arm within 0.7% of baseline at 24K–64K, none consistently ahead |
| 4,096-token prefill chunks with a 131K window | does not fit: 9.9 GiB of KV needed |
| Prefill chunks above 2,048 tokens at a 65K window | does not boot either |
| Router matmul tiles 32×4×3, 16×4×3, 16×4×2, 32×2×2, 64×4×2, 32×8×2 (`tuning/steps_router_and_decode_tiles.jsonl`) | 7.02–7.18 ms per step without the drafter, against 6.99 for the adopted 16×2×3 and 7.09 before |
| Attention decode tiles 64×4×3, 64×8×2, 32×4×3, 128×8×2 | 6.97–7.04 ms against 6.95 for the adopted 64×4×2. 256 keys needs 151 KB of shared memory |
| Window-attention prefill tiles 64×4×2, 32×8×2, 64×8×3 (`tuning/prefill_by_attention_tile.csv`) | 24K–64K prefill −3.8%, −0.6%, −0.4%. 128×8×2 does not boot, for the window or the full layers |
| Software fp8 KV cache, the approach of vllm#69165, no drafter (`tuning/fp8_software_kv.csv`) | KV pool 374,234 tokens, retrieval 5 / 5 to 120K, top token against the one-card bf16 reference 98.5% same. Prefill 2,306 / 2,095 / 1,681 tok/s at 8K / 32K / 120K against 7,078 / 6,621 / 5,212; decode 122.6 tok/s against 143.5. Not in the patch |
| A fixed draft of 4 or 5 | ahead of 7 on low-acceptance text, behind on predictable text; the per-step policy does both |
| Block verification | no gain |
| The draft-length policy at every batch size (step costs per batch in `tuning/step_cost_by_batch.jsonl`) | no gain over a fixed 3 for two or more requests: at 8 requests JSON 856 against 856, code 766 against 762, prose 702 against 726, mixed 702 against 740. GSM8K at concurrency 4: 192 / 200 with three questions at the token cap. Kept for one request only |

### Reasoning loops and draft length

Not a tuning result, but the runs above line up. GSM8K, first 200, concurrency 4, 4,096-token cap:

| Build | Drafts verified per step | Correct | Questions that ran to the cap |
|---|---|---|---|
| Untuned port, no drafter | none | 196 | 1 |
| Untuned port, drafter | 7 | 193 | 3 |
| Tuned, before the router and tile rows | 3 | 196 | 1 |
| The same after a restart | 3 | 196 | 1 |
| Tuned, final image | 3 | 196 | 1 |
| Tuned, policy at every batch size | up to 7 | 192 | 3 |

Question 119 is the one that loops in every run with 3 drafts or none. The two long-draft runs looped on 40, 87 and 119, and on 87, 93 and 139. Six runs of 200 do not establish that long verified drafts cause loops, and nothing here explains why they would at temperature 0. A full-set run at fixed lengths would settle it.

## Before tuning

Everything from here on was measured on the port before the tuning patch.

## Decode

`bench-results/runs.csv`. 512 tokens generated; decode rate is measured after the first streamed token.

| Workload | vLLM, drafter off | vLLM, drafter on | llama.cpp |
|---|---|---|---|
| code, greedy | 122.6 | 191.0 | 50.2 |
| code, temperature 1 | 122.7 | 185.1 | 48.0 |
| prose, greedy | 122.6 | 162.2 | 50.0 |
| JSON tool calls, greedy | 123.0 | 236.6 | 50.2 |

One 3090 alone (TP1, 8K window): 95 tok/s without the drafter.

## Depth: prefill, decode, retrieval

Five planted facts per prompt at 5 / 30 / 55 / 80 / 97% of the context, `reasoning_effort: low`, unique first line per request (prefix cache hit rate 0%).

Drafter off, `--max-model-len 196608`:

| Prompt tokens | Facts | Time to first token | Prefill tok/s | Decode tok/s | llama.cpp prefill | llama.cpp decode |
|---|---|---|---|---|---|---|
| 8,175 | 5 / 5 | 1.2 s | 6,864 | 120.1 | 697 | 47.9 |
| 24,175 | 5 / 5 | 3.7 s | 6,467 | 115.3 | 564 | 45.2 |
| 32,174 | 5 / 5 | 5.1 s | 6,303 | 113.2 | 511 | 44.3 |
| 48,182 | 5 / 5 | 8.1 s | 5,949 | 109.5 | 436 | 42.1 |
| 64,174 | 5 / 5 | 11.4 s | 5,653 | 105.6 | 376 | 39.8 |
| 120,173 | 5 / 5 | 25.3 s | 4,753 | 94.3 | 255 | 33.4 |

Drafter on:

| Prompt tokens | Facts | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 24,178 | 5 / 5 | 6,442 | 368.4 |
| 120,176 | 5 / 5 | 4,758 | 307.9 |

The retrieval answer is a short, highly predictable list (mean acceptance 5.8 tokens), so the drafter-on decode rates at depth are a best case, not a typical one.

Packaged image booted from `docker-compose.yml` as shipped (drafter on, vision loaded), `runs.csv` label `packaged image …`:

| Workload | Decode tok/s | Prefill tok/s |
|---|---|---|
| code, greedy | 175.1 | — |
| code, temperature 1 | 162.8 | — |
| prose, greedy | 160.7 | — |
| JSON tool calls, greedy | 225.7 | — |
| 24K context, first large prompt after a cold boot | 372.4 | 4,115 (includes kernel compilation) |
| 24K context, repeated | 371.3 | 6,518 |
| 64K context | 366.2 | 5,702 |

Each request carries a different random first line, so drafter acceptance and the decode rate vary a few percent between runs of the same workload.

## GSM8K

`bench-results/gsm8k.csv`. First 200 test questions, vLLM's chat protocol (5-shot in one user message, temperature 0, 4,096-token cap), concurrency 4. Same script for all three arms (`tools/gsm8k_eval.py`).

| Arm | Correct | Hit token cap | Wall | Aggregate tok/s |
|---|---|---|---|---|
| vLLM, drafter off | 196 / 200 = 98.0% | 1 | 370 s | 310.6 |
| vLLM, drafter on | 193 / 200 = 96.5% | 3 | 197 s | 590.8 |
| llama.cpp (concurrency 1) | 194 / 200 = 97.0% | 1 | 2,435 s | 47.6 |

Paired by question:

| Pair | Both correct | Only first | Only second |
|---|---|---|---|
| vLLM off vs llama.cpp | 193 | 3 | 1 |
| vLLM off vs vLLM on | 193 | 3 | 0 |

- Upstream's numbers on a GB200: 96.8% on the full 1,319 (96.7% without the drafter), 97.5% on the first 200.
- Drafter on vs off: 3 discordant questions, all against the drafter (exact McNemar p = 0.25). Two of the three are reasoning loops that ran to the token cap. Not significant at n = 200; the full 1,319 has not been run. The drafter is on in the shipped config.
- Mean acceptance length with the drafter: 3.6 on GSM8K, 2.6–2.9 on code and prose.

## Token-level agreement

`bench-results/token_agreement.csv`, reference ids in `tf_reference_tp1.json`. Eight prompts × 192 greedy tokens generated on the one-card server, then scored teacher-forced on both.

| Candidate | Reference token is rank 1 | Median \|Δ logprob\| | p95 | Max | Passage NLL |
|---|---|---|---|---|---|
| TP1, re-scored on itself | 98.4% | 0 | 0 | 0 | 1.771 |
| TP2 against TP1 | 98.8% | 0.0012 | 0.070 | 0.454 | 1.776 |

- The 98.4% on TP1 itself is decode against prefill: the two paths use different kernels and disagree at near-ties. TP2 is no further from TP1 than TP1's own decode path is from its prefill path.
- Re-scoring is bitwise repeatable. `tools/tf_probe.py compare` against the shipped server, run again from this repo's copy of the script, returns the TP2 row unchanged.

## Images

`bench-results/images.csv`. One four-quadrant PNG per shape, drafter on.

| Shape | Result | Wall s |
|---|---|---|
| 64×64 | correct | 0.4 |
| 512×512 | correct | 0.3 |
| 1024×768, 768×1024 | correct | 0.4 |
| 2048×2048 | correct (2,732 prompt tokens) | 1.0 |
| 4096×1024 | correct | 1.0 |
| 2560×2560 and larger | HTTP 400: 4,096+ image tokens exceed the 2,704-token encoder cache | — |

## Memory

| Config | Weights per card | KV pool | Free per card at boot |
|---|---|---|---|
| TP1, one card, 8K window | 17.64 GiB | 8,639 tokens | 3.1 GiB |
| TP2, text only, no drafter, 65K window | 8.88 GiB | 242,694 tokens | 3.7 GiB |
| TP2, text only, no drafter, 196K window | 8.88 GiB | 303,976 tokens | 4.0 GiB |
| TP2, text only, drafter, 196K window | 9.39 GiB | 249,185 tokens | 4.0 GiB |
| TP2, vision + drafter, 196K window (shipped) | 9.98 GiB | 233,016 tokens | 4.0 GiB |
| TP2, `--max-num-batched-tokens 8192` | — | refuses to start: one 196K request would need 18.5 GiB of KV | — |

After the 120K fill and the image sweep: 3.3 GiB free per card.

## Unit tests

Upstream `tests/models/inkling2` at `fc144315` (the head of vllm-project/vllm#69028 at the time), gates widened with `tools/widen_test_gates.py`, on one RTX 3090. Recipe: [README](README.md#upstream-unit-tests).

| File | Passed | Failed | Skipped | Failures on the Ampere path |
|---|---|---|---|---|
| `test_core_ops.py` | 64 | 1 | 4 | 0 |
| `test_attention.py` | 106 | 11 | 30 | 0 |
| `test_moe.py` | 31 | 33 | 41 | 0 |
| `test_model.py` | 21 | 0 | 0 | 0 |
| `test_dspark.py` | 9 | 0 | 2 | 0 |

- 44 of the 45 failures are kernels that use Triton's `fp8e4nv` type, which needs sm_89: the fp8 KV cache variants and the Blackwell route / activation-quantization path. The Ampere path runs bf16 KV and BF16 activations and calls none of them.
- The 45th is `test_humming_experts_are_deterministic[1-nvfp4]`: at seed 0 its random draw contains no local expert and its own precondition asserts. At seeds 1, 2, 3 and 5 the output is bitwise identical over five runs (`tools/humming_seedcheck.py`).
- Passing and on the path: the expert layer against the torch reference for NVFP4 and MXFP8, expert-parallel against single rank, expert selection, the router GEMM fallback, Triton attention with bf16 KV, tiny looped model decode against prefill, and DSpark.
- `test_allreduce.py` and `test_vision.py` were not run: the first covers the fused NVLink-multicast collective this path does not use, the second skips itself off capability 9/10/12 (vision was checked end to end instead).

## Reproduce

```bash
docker compose up -d

tools/vbench.py "<label>" http://localhost:8000 bench-results/runs.csv \
    --depth-tokens 8000 24000 32000 48000 64000 120000 --corpus-dir <a source tree>
tools/gsm8k_eval.py "<label>" -n 200 --concurrency 4
tools/image_sweep.py "<label>"
tools/tf_probe.py compare http://localhost:8000 bench-results/tf_reference_tp1.json
```

- **Depth corpus:** the published depth rows were built from the source and docs of a private config repo. `--corpus-dir` takes any tree of source and docs; the planted facts and the question are the same, the filler text is yours.
- **Unpacked image:** the rows not labelled `packaged image` came from `tools/serve.sh '"device=0,1"' 8000 /tmp/serve.log --tensor-parallel-size 2 --max-model-len 196608 --max-num-batched-tokens 2048 --max-num-seqs 4 --gpu-memory-utilization 0.92 --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` on a copy of the image unpacked with `tools/fetch_rootfs.sh`, with the same kernel swap and patches applied.
