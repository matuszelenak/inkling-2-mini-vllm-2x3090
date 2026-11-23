# Inkling 2 Mini on vLLM, 2× RTX 3090 — measurements

2026-11-21. 2× RTX 3090 at 300 W, NVLink, TP2 + EP2, bf16 KV, `--max-num-batched-tokens 2048`, `--gpu-memory-utilization 0.92`. vLLM `0.1.dev25855+g7cb5dae22` (the `inkling-2-mini` image) with the kernel swap and patches of this repo. Rows labelled `packaged image …` ran on the image built from the Dockerfile; all others ran on the same files from an unpacked copy of the image (`tools/vrun.sh`), and the built image reproduces their token-agreement figures exactly. Data and column definitions: [bench-results/](bench-results/README.md).

llama.cpp columns: the same cards, prompts and day. llama.cpp at `690b87d` (ggml-org/llama.cpp#32694 + #32704) plus one local patch for loop-slot placement, Unsloth's `UD-Q8_K_XL` GGUF, layer split across both cards, f16 KV, 131,072 window, no drafter. That config and its CSVs are not part of this repo: `bench-results/` holds the vLLM rows only.

## Conv-state cache and 8,192-token prefill chunks

2026-11-23. The image this repo builds now (`inkling-2-mini-vllm-2x3090:g7cb5dae22-tune3`), booted from `docker-compose.yml` as shipped, against the previous build (`-tune2`, next section). Same weights, same bf16 KV cache. Data: `bench-results/tuning/conv-state-tail/`. The tables down to [Checks](#checks) were measured before the last change of the day, [expert-GEMM blocks](#expert-gemm-blocks-for-prefill-batches), which has the final image's own rows.

Where the memory was: each of the 48 layer visits keeps the inputs of its 4-tap convolutions in a paged sliding-window cache, 16 KB per token per card. vLLM backs that cache for every token in flight, which with async scheduling is two prefill chunks. By the planner's formula that is 3,084 of the pool's 7,504 blocks for one request at 2,048-token chunks (3.4 GiB per card, more than a 196K request's attention KV), and it grows with the chunk: at 4,096 the largest window that boots is 22,016 tokens. A step only ever reads the conv state of the 3 tokens before it. This build backs the tail of each step and leaves the rest of a chunk unbacked (the manager is adapted from HyperInkling, on vLLM's own tail-only manager for drafter caches), plus one block every 512 tokens so the prefix cache still has somewhere to resume ([Prefix reuse](#prefix-reuse)).

### Prefill by chunk size

`tools/vbench.py` depth prompts on War and Peace, this build's code at the 196,608 window with the drafter on, one boot per chunk size, the same morning. No 512-token checkpoints in these boots. The 8K column is the two warm runs.

| `--max-num-batched-tokens` | 8K | 24K | 64K | 120K | KV pool, tokens |
|---|---|---|---|---|---|
| 2,048 | 7,173–7,189 | 6,782 | 6,076 | 5,251 | 391,858 |
| 4,096 | 7,561–7,570 | 7,213 | 6,455 | 5,632 | 333,439 |
| 8,192 | 8,119–8,140 | 7,750 | 6,859 | 5,920 | 257,650 |
| 16,384, at a 98,304 window | 8,156 | 7,910 | 6,999 | not run | 116,495 |

- **8,192 is shipped:** +13% at every depth over 2,048. 16,384 adds 2% and leaves a pool smaller than the shipped window.
- **A prompt under 2,048 tokens is one chunk either way** and does not change.
- **With vLLM's conv-state layout** the same comparison only boots at a 20,000-token window: 7,155 → 7,503 tok/s at 8K and 6,893 → 7,339 at 16K for 2,048 → 4,096 (`prefill_by_chunk_stock_manager_20k_window.csv`).

### This build against the previous one

| | Previous build | This build |
|---|---|---|
| Prefill on War and Peace at 8K / 24K / 64K / 120K, cards warm from the suite (`prefill_depth.csv`) | 7,180 / 6,782 / 6,076 / 5,251 (the 2,048 row above, cooler cards) | 7,977 / 7,617 / 6,756 / 5,843 |
| To first token at 120K | 22.9 s | 20.6 s |
| War and Peace with the drafter at 2K / 60K / 110K / 190K, prefill tok/s (`depth_accept_warpeace.jsonl`) | 6,567 / 6,016 / 5,264 / 4,384 | 6,914 / 6,795 / 5,965 / 4,997 |
| The same, decode tok/s, medians of three | 201.6 / 160.2 / 143.2 / 130.4 | 194.8 / 159.5 / 144.6 / 130.1 |
| The same, tokens per step | 2.29 / 1.92 / 1.88 / 1.96 | 2.13 / 1.91 / 1.89 / 1.93 |
| Greedy code / prose / JSON / chat, tok/s (`steps.jsonl`) | 239.7 / 182.1 / 327.6 / 214.0 | 241.2 / 183.5 / 327.8 / 215.1 |
| The same, ms per step | 11.35 / 10.71 / 11.90 / 11.10 | 11.33 / 10.73 / 11.89 / 11.10 |
| Aggregate at 1 / 2 / 4 / 8 requests (`concurrency.jsonl`) | 270 / 357 / 534 / 778 | 270 / 357 / 535 / 770 |
| GSM8K-200 at concurrency 4 | 194 / 200, 2 at the token cap, 649 tok/s | 194 / 200, 3 at the token cap, 650 tok/s |
| KV pool at the shipped flags | 223,030 tokens | 246,986 |
| Largest image (`images_large.csv`) | about 2080×2080 px, 2,704 tokens | 3584×3584 px, 8,168 tokens; 3840×3840 gets the HTTP 400 |
| Memory in use per card after the whole suite | 22.9 GiB | 22.6 GiB |

Decode is unchanged, as it should be: nothing in a decode step changed.

The other projects' protocols on this build (`published_protocols-drafter.json`, `more_protocols-drafter.json`). Decode rows with the drafter at temperature 0.6 move by 10–30% between runs with how many drafts the sampled text lets through; the per-step time behind them is the same on both builds.

| Protocol | Row | Theirs | Previous build | This build |
|---|---|---|---|---|
| agent2x0r | Prefill at 4K / 16K / 32K / 60K | 5,575 / 5,709 / 5,439 / 4,954 | 6,592 / 6,772 / 6,528 / 5,996 | 7,280 / 7,606 / 7,329 / 6,775 |
| agent2x0r | Decode with the drafter at 1K / 4K / 16K / 32K / 60K | 172 / 166 / 178 / 178 / 170 | 240 / 234 / 248 / 244 / 215 | 223 / 235 / 227 / 205 / 208 |
| agent2x0r | Code / math / prose / JSON | 202 / 178 / 136 / 210 | 262 / 231 / 195 / 262 | 238 / 243 / 197 / 269 |
| agent2x0r | Aggregate at 1 / 2 / 4 / 8 users | 191 / 278 / 353 / 383 | 268 / 369 / 423 / 494 | 230 / 335 / 421 / 505 |
| HyperInkling, as of 2026-11-22 | Chat / code / structured / prose, one run each | about 150 / 220 / 295 / 155 | 159 / 270 / 340 / 171 | 158 / 257 / 340 / 162 |
| lulakien | Prefill at 2,000 / 16,000 / 64,000 tokens | 7,480 / 7,140 / 6,280 at 280 W; 7,840 / 7,480 / 6,620 at 350 W | 7,190 / 6,844 / 5,950 | 7,170–7,243 / 7,624–7,636 / 6,596–6,600 |
| lulakien | Code / explanation / story | 313 / 166 / 126; 323 / 170 / 129 at 350 W | 355 / 197 / 165 | 356 / 196 / 165 |
| lulakien | Aggregate at 2 / 4 / 8 streams | 260 / 339 / 445 | 336 / 420 / 687 | 336 / 420 / 687 |
| ingine | Chat / code | 235–239 / 219–224 | 249–255 / 278–280 | 249–255 / 279–280 |

lulakien's prefill figures are with INT8 attention weights and the `gpu-unlock` kernel module; these are on stock cards at 300 W with the original weights. Its 2,000-token row is still ahead of this build.

### Long windows

One boot and one fill each, `tools/vbench.py` retrieval prompts on War and Peace (`long_window.csv`). All on the bf16 KV cache. Flags other than those listed are the shipped ones, with `--max-num-seqs 4`.

| Mode | Chunk | `--gpu-memory-utilization` | KV pool | Window | Prompt | Facts found | Prefill tok/s | To first token | Decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| As shipped (images, drafter) | 8,192 | 0.92 | 246,986 | 196,608 | 190,178 | 5 / 5 | 5,061 | 37.6 s | 294 on a short answer |
| Images, drafter | 8,192 | 0.92 | 281,123 | 262,144 | 250,180 | 5 / 5 | 4,496 | 55.7 s | 268 on a short answer |
| Images, drafter | 2,048 | 0.92 | 436,045 | 425,984 | 415,174 | 5 / 5 | 3,003 | 138 s | 199 on a short answer |
| Text only, drafter | 2,048 | 0.95 | 471,527 | 466,944 | 456,176 | 5 / 5 in two fills; a third ran to the 2,048-token cap without answering | 2,842 | 161 s | 127–239 |
| Text only, no drafter | 2,048 | 0.97 | 607,708 | 589,824 | 578,175 | 5 / 5 | 2,433 | 238 s | 56 |

- **Text only, drafter:** 491,520 does not boot at 0.95 (the server estimates a maximum of 471,936). A 430,175-token prompt on the 466,944 boot: 5 / 5, 2,920 tok/s.
- **Not tried:** windows above 589,824, and a text-only boot at 8,192-token chunks.

### Prefix reuse

vLLM's default (`--prefix-cache-retention-interval 0`) keeps sliding-window state only where a prompt ended and at a junction it has seen two requests share, so a prefix shared by different conversations is reused from the third request on. A tail-only conv state has nothing at that junction, and on its own loses that reuse. This build keeps one conv-state block every 512 tokens and ships `--prefix-cache-retention-interval 512`. Prompt tokens taken from the cache, each row on a fresh boot with the drafter on (`tools/tune/prefix_reuse.py shared`):

| Request | Previous build (vLLM's layout, 2,048 chunks) | Tail only, no checkpoints | This build |
|---|---|---|---|
| A: 10,000 shared tokens + 1,500 of its own, cold | 0 | 0 | 0 |
| B: the same 10,000 + 1,500 different | 0 | 0 | 9,728 |
| C: the same 10,000 + 1,500 different | 9,984 | 0 | 9,728 |
| D: the same 10,000 + 200 different | 9,984 | 0 | 8,704 |
| E: the same 10,000 + 1,500 different | 9,984 | 0 | 9,728 |
| A2: A + 300 more (a follow-up turn) | 11,456 | 11,456 | 11,456 |
| F: the first 6,000 of the shared part + 1,500 | 0 | 0 | 5,632 |
| G: the first 6,000 of the shared part + 1,500 | 5,952 | 0 | 5,632 |

- **D** resumes lower than C because the drafter's own cache needs a request to end at least 1,023 tokens past the point it resumes from.
- **Cost:** the pool is 246,986 tokens with the checkpoints and 257,650 without.

Whether state resumed from a checkpoint is right (`tools/tune/prefix_reuse.py exact`, no drafter, 8,192-token chunks, `--no-async-scheduling`, 32,768 window so vLLM's own conv-state layout also boots): a cold 30,000-token prompt, then seven prompts that end 4 to 272 tokens past a point inside it. The first generated token's top-5 logprobs against the same run on vLLM's layout:

| Build | Resumed at the same point as vLLM's layout | Largest logprob gap on those |
|---|---|---|
| This build | 7 of 8 probes (the eighth resumed at 4,096 where vLLM's layout started cold) | 0.0000 on all 7 |
| Control: the checkpoint blocks withheld from the worker | the same 7 | 0.107 to 0.735 on the 5 that resume from a checkpoint in a later chunk; 0.0000 on the 2 that resume from a step's tail |

Against a cold computation of the same prompts the gap is 0.01 to 0.2, which is what prefilling the same text in different chunks gives, so that comparison cannot tell good state from missing state. The one above can.

### A stream beside a cold prefill

`tools/tune/stream_beside_prefill.py`: one request streams an essay while another prefills a cold 60,000-token prompt. Two runs each.

| | Previous build (2,048 chunks) | This build (8,192) |
|---|---|---|
| The prefill | 9.9 s | 8.8 s |
| The stream before / during / after, tok/s | 173 / 5.3 / 161–167 | 172–173 / 1.8–2.0 / 169–172 |
| Longest pause in the stream | 0.39 s | 1.35 s |

### Checks

| Check | Result |
|---|---|
| Built image against the tested tree | 43 patched files identical |
| Correctness gate (`tools/tune/gate.py check`) | pass: reference token rank 1 at 98.4% of positions, median \|Δ logprob\| 0.0015; decode agrees with prefill at 99.0%; 16K prefill top-1 97.4% against the reference server; decode at depth 97.8% |
| Decode against prefill at 12K, 6,912 tokens | 99.2% |
| Block manager unit test (`tools/tune/test_conv_tail.py`, no GPU) | 12 pass: a chunk holds a tail and its checkpoints, the admission count matches the pool exactly, state survives every draft in flight being rejected, hits land on the last aligned position and on checkpoints |
| GSM8K-200 | 194 / 200 |
| Tool call, images to 3584×3584 px | pass |
| Memory, worst order tried: logprob requests, a 190K fill, the largest images during 8 concurrent requests, another 190K fill (`memory_stress_*.csv`) | 23.0 GiB in use per card of 24.0, no error |
| Server log over the suite | no traceback, no CUDA error |

### Expert-GEMM blocks for prefill batches

The expert-GEMM library picks its row block by batch size and alternates between 64, 96 and 128 rows from range to range. The earlier tuning replaced the warp layout only where it picked 64. On a 3090 (`tools/tune/mb_humming.py`, one rank of the real layer shape, `mb_humming_prefill_*.jsonl`) 64-row blocks are faster at every size from 1,600 tokens up, with identical output:

| Tokens in the batch | Library's choice, µs per call | 64-row blocks, 64×64×64 warps, 2 CTAs per SM | Gain |
|---|---|---|---|
| 1,280 (NVFP4) | 1,672 (96 rows) | not faster; left alone | |
| 1,600 | 2,157 (128 rows) | 2,059 | 5% |
| 3,072 | 4,064 (128 rows) | 3,460 | 17% |
| 4,096 | 4,659 (96 rows) | 4,422 | 5% |
| 5,120 | 6,034 (128 rows) | 5,448 | 11% |
| 8,192 | 8,860 (64 rows, its own warps) | 8,356 | 6%, already applied |
| 9,000 | 9,882 (128 rows) | 9,188 | 8% |
| 3,072 / 5,120 / 8,192 on the MXFP8 layers | 4,136 / 6,124 / 9,356 | 3,629 / 5,679 / 8,832 with 64×32×64 warps, 3 stages, 1 CTA per SM | 14% / 8% / 6% |

64-row blocks were applied to every batch above 5,568 local rows (about 1,400 tokens) first. On the server, two alternating boots of each build, `tools/vbench.py` on War and Peace (`prefill_by_expert_block.csv`):

| Prompt tokens | Before | After | Change |
|---|---|---|---|
| 1,677 | 5,927–5,954 | 6,690–6,714 | +13% |
| 2,975 | 7,002–7,041 | 7,524–7,535 | +7% |
| 3,175 | 7,129–7,214 | 7,622–7,687 | +7% |
| 4,175 | 7,773–7,808 | 7,782–7,815 | none |
| 5,175 | 7,732–7,748 | 8,076–8,095 | +4% |
| 6,675 | 7,865–7,942 | 8,127–8,136 | +3% |
| 12,175 | 7,755–7,812 | 6,375 (first run after boot), 7,888 | +1.7% on the warm run |
| 8K / 20K / 24K / 64K / 120K | 8,117–8,192 / 7,719–7,767 / 7,736–7,794 / 6,854–6,909 / 5,911–5,981 | 8,127–8,147 / 7,764–7,790 / 7,732–7,770 / 6,867–6,890 / 5,924–5,940 | none: full 8,192-token chunks were already covered |

Batches under 1,400 tokens are left with the library's choice: the faster configs found there change the K block, which is not guaranteed to give identical output, for a few milliseconds.

A second sweep with smaller blocks (`mb_humming_prefill_nvfp4_small_blocks.jsonl`): below about 3,600 tokens an expert sees so few rows that 64-row blocks are mostly padding, and 48-row blocks are 1% to 4% faster again (2,367 against 2,463 µs at 2,048 tokens; 3,415 against 3,460 at 3,072; level at 4,096). Applied to the NVFP4 layers below 14,784 local rows. On the server, one-token completions over random token ids, two alternating boots of each build:

| Prompt tokens | 64-row blocks | 48-row blocks below 3,600 tokens | Change |
|---|---|---|---|
| 1,500 | 7,133–7,150 | 7,250–7,267 | +1.7% |
| 2,000 | 7,407–7,429 | 7,508–7,530 | +1.3% |
| 2,500 | 7,532–7,539 | 7,571–7,608 | +0.7% |
| 3,000 | 7,716–7,720 | 7,754–7,764 | +0.5% |
| 4,000 | 7,828–7,852 | 7,866–7,868 | none |

The final image (`final-image/`; the same suite one change earlier is in `before-48-row-blocks/`), booted from `docker-compose.yml`:

| Row | Final image |
|---|---|
| Built image against the tested tree | 43 patched files identical |
| Correctness gate | pass, every figure identical to the build before these two changes |
| GSM8K-200 at concurrency 4 | 195 / 200, 1 at the token cap, 649 tok/s. One change earlier: 193 / 200 with 4 at the cap, and 194 / 200 with 2 on a repeat. The six runs before today: 194 to 196 |
| Prefill on War and Peace at 8K / 24K / 64K / 120K, warm cards | 7,994 / 7,632 / 6,782 / 5,859 |
| Prefill at 1.7K / 2.2K / 3K / 4.2K / 5.2K / 6.7K, warm cards | 6,706 / 7,177 / 7,610 / 7,709 / 7,979 / 8,025 |
| War and Peace with the drafter at 2K / 60K / 110K / 190K: prefill; decode, medians of three | 7,017 / 6,820 / 5,989 / 5,009; 198.1 / 167.4 / 151.0 / 130.1 |
| Greedy code / prose / JSON / chat | 240.5 / 182.8 / 328.4 / 214.6 tok/s; 11.31 / 10.68 / 11.87 / 11.07 ms per step |
| Aggregate at 1 / 2 / 4 / 8 requests | 270 / 357 / 535 / 770 |
| agent2x0r protocol, prefill at 4K / 16K / 32K / 60K | 7,547 / 7,635 / 7,349 / 6,803 |
| lulakien protocol, prefill at 2,000 / 16,000 / 64,000, cards warm from the suite | 7,275–7,345 / 7,643–7,653 / 6,616–6,618 |
| lulakien protocol, code / explanation / story; 2 / 4 / 8 streams | 356 / 196 / 165; 336 / 419 / 711 |
| Server log | no traceback, no CUDA error |

### Many requests at once

`tools/tune/bench_concurrency.py` (mixed prompts, 512 tokens out, temperature 0.6, low reasoning effort) on the final build's code with `--max-num-seqs 64` and the draft schedule extended to 64 requests (`[[1,1,7],[2,64,3]]`), everything else as shipped (`concurrency_64_seqs.jsonl`). Aggregate output tok/s:

| Requests | Drafter, 3 verified drafts | No drafter |
|---|---|---|
| 8 | 778 | 628 |
| 16 | 1,167 | 966 |
| 32 | 1,634 | 1,392 |
| 64 | 2,058 | 2,114 |

The KV pool at 64 sequences is 200,397 tokens with the drafter and 317,770 without. vllm#69165 reports, on two 3090s without a bridge at a 4K window on its own prompts, 663 at 16 and 781 at 64 with the drafter and 1,144 at 64 without.

### Tried on the way, not adopted

| Change | Result |
|---|---|
| Tail-only conv state without checkpoints (HyperInkling's form) | 10,664 more pool tokens, and no reuse of a prefix shared across conversations (table above) |
| 4,096- and 16,384-token chunks | between 2,048 and 8,192; 2% over 8,192 for most of the pool |
| More splits of the decode attention over a full layer, 2 and 4 CTAs per SM (HyperInkling reports +3% at 32K and +10% at 128K on one card) | no change on two cards, where each rank already runs one CTA per SM: 130.2 / 109.2 / 93.6 tok/s at 32K / 110K / 190K without the drafter, against 129.7 / 109.1 / 93.5 and 129.5 / 108.7 / 93.3 (`decode_depth_by_attention_splits.jsonl`) |

Decode at depth without the drafter is at the cards' memory bandwidth: at 190K a step reads 3.1 GB of bf16 keys and values per card and takes 3.75 ms longer than at no depth, about 830 GB/s.

## Reduced draft vocabulary

2026-11-22. The previous image (`inkling-2-mini-vllm-2x3090:g7cb5dae22-tune2`), booted from `docker-compose.yml` as shipped, against the previous build (`-tune1`, everything in [Tuning](#tuning) below) measured the same morning. Data: `bench-results/tuning/reduced-draft-vocab/`. Only the drafter path changes: decode without the drafter and prefill are the previous build's.

Where the time was (torch profiler, rank 0, a step with 7 drafts): two passes over the TP-local output head per step at 537 µs each, one for the model and one for the drafter; 7 projections of the drafter's Markov latent onto every vocabulary row at 64 µs each; 8 all-gathers of full-vocabulary logits at 32 µs each, 7 of them the drafter's. About 1.2 ms of every step with the drafter was vocabulary-sized work for the drafter. It now scores the 24,576 highest-ranked token ids of each vocabulary half.

### Decode, one request

`tools/tune/bench_step.py`, 512 tokens out, medians of three (greedy) or four (temperature 0.6) runs. Previous build: `steps_previous_build_same_day.jsonl`. This build: `steps.jsonl`.

| Workload | Previous tok/s | This build | Previous ms per step | This build | Previous tokens per step | This build |
|---|---|---|---|---|---|---|
| code, greedy | 227.5 | 239.7 | 12.05 | 11.35 | 2.69 | 2.67 |
| prose, greedy | 172.4 | 182.1 | 11.43 | 10.71 | 2.01 | 1.99 |
| JSON, greedy | 310.7 | 327.6 | 12.55 | 11.90 | 3.85 | 3.85 |
| chat, greedy | 203.2 | 214.0 | 11.82 | 11.10 | 2.41 | 2.40 |
| code, temperature 1 | 227.4 | 231.9 | 12.14 | 11.37 | 2.74 | 2.68 |
| code, temperature 0.6 | 245.6 | 254.7 | 12.34 | 11.62 | 3.17 | 3.13 |
| prose, temperature 0.6 | 176.6 | 191.8 | 11.53 | 10.78 | 2.03 | 2.07 |
| JSON, temperature 0.6 | 292.5 | 321.4 | 12.65 | 12.07 | 3.79 | 3.90 |
| chat, temperature 0.6 | 213.4 | 219.4 | 12.21 | 11.27 | 2.63 | 2.48 |

Greedy text is the same on both builds, so those rows compare like with like: the step is 0.65 to 0.72 ms shorter and tokens per step are unchanged. Acceptance by draft position on greedy code: 0.66 / 0.43 / 0.27 / 0.16 on both.

### Size of the set, and what the ranking has to cover

First-draft acceptance fell in proportion to how much of the output the set could express. With a first ranking built from a novel and a config tree only, and 8,192 ids per half, the step was 1.1 to 1.3 ms shorter but chat's first draft was accepted 0.60 of the time against 0.69: that ranking covered 87% of held-out chat output. The shipped ranking (49,667 ids; `tools/tune/build_draft_vocab.py`) covers, per vocabulary half:

| Held-out text | 8,192 ids | 16,384 | 24,576 |
|---|---|---|---|
| Model output: code / chat / prose / JSON | 98.0% / 91.6% / 92.7% / 96.3% | 99.3% / 97.1% / 97.7% / 98.2% | 99.5% / 98.4% / 99.4% / 99.5% |
| C source, a file not in the ranking | 97.2% | 99.0% | 99.4% |
| Python source, a part not in the ranking | 97.9% | 99.5% | 99.7% |
| wikitext-2 test | 90.7% | 97.6% | 99.2% |

The model-output samples are small (177 to 9,248 tokens). At 16,384 ids the step is 0.05 ms shorter than at 24,576 and greedy tokens per step are the same; 24,576 is shipped for the coverage. The ranking is English text and code.

### Other rows

| | Previous build | This build |
|---|---|---|
| Aggregate at 1 / 2 / 4 / 8 requests (`concurrency.jsonl`) | 255 / 340 / 516 / 735 that morning; 253 / 336 / 509 / 747 the night before | 270 / 357 / 534 / 778 |
| GSM8K-200 at concurrency 4 | 194 / 200, 2 at the token cap, 625 tok/s that morning; 196 / 200 three times before | 194 / 200, 2 at the token cap, 649 tok/s |
| War and Peace, decode with the drafter at 2K / 60K / 110K / 190K (`depth_accept_warpeace.jsonl`) | 194.5 / 153.4 / 130.9 / 122.8 | 201.6 / 160.2 / 143.2 / 130.4 |
| The same, tokens per step | 2.34 / 2.00 / 1.81 / 1.94 | 2.29 / 1.92 / 1.88 / 1.96 |
| The same, prefill tok/s | 6,570 / 6,059 / 5,283 / 4,391 | 6,567 / 6,016 / 5,264 / 4,384 |
| KV pool | 225,883 tokens | 223,030 |
| Numbered lists, greedy, reasoning effort none: 1 request / 8 at once (`lists.json`, `tools/tune/lists_bench.py`) | not run | 476–486 / 1,481 tok/s; at low effort 510 / 1,420 |
| Memory in use per card after the whole suite | 22.7 GiB | 22.9 GiB |

The other projects' protocols on this build (`published_protocols-drafter.json`, `more_protocols-drafter.json`; "theirs" as published on 2026-11-22):

| Protocol | Row | Theirs | Previous build | This build |
|---|---|---|---|---|
| agent2x0r | Decode with the drafter at 1K / 4K / 16K / 32K / 60K | 172 / 166 / 178 / 178 / 170 | 216 / 228 / 216 / 211 / 200 | 240 / 234 / 248 / 244 / 215 |
| agent2x0r | Code / math / prose / JSON | 202 / 178 / 136 / 210 | 229 / 225 / 180 / 251 | 262 / 231 / 195 / 262 |
| agent2x0r | Aggregate at 1 / 2 / 4 / 8 users | 191 / 278 / 353 / 383 | 260 / 322 / 422 / 482 | 268 / 369 / 423 / 494 |
| HyperInkling | Chat / code / structured / prose, one run each | about 150 / 220 / 295 / 155 | 146 / 268 / 327 / 144 | 159 / 270 / 340 / 171 |
| lulakien | Code / explanation / story | 313 / 166 / 126 | 340 / 190 / 153 | 355 / 197 / 165 |
| lulakien | Aggregate at 2 / 4 / 8 streams | 260 / 339 / 445 | 324 / 397 / 674 | 336 / 420 / 687 |
| lulakien | Prefill at 2,000 / 16,000 / 64,000 tokens | 7,480 / 7,140 / 6,280 | 7,254–7,405 / 6,909–7,031 / 5,998–6,094 over three sessions | 7,190 / 6,844 / 5,950 after 20 minutes of load |
| ingine | Chat / code prompt, default reasoning effort | 235–239 / 219–224 | 232–235 / 261–263 | 249–255 / 278–280 |

### Checks

| Check | Result |
|---|---|
| Correctness gate (`tools/tune/gate.py`) | pass, with the previous build's figures: 98.37% against the one-card reference, decode against prefill 99.02% over 1,536 tokens and 99.16% over 6,912 tokens at 12K depth |
| Sampling (`tools/tune/sampling_check.py`: 6,000 seeded 5-token continuations of four prompts at temperature 1, on both builds; `sampling_*.json`) | at every position and prompt the total-variation distance between the builds (0.003 to 0.18) is below the distance between two halves of the previous build's own samples (0.085 to 0.56). Tokens outside the reduced set appear as often on this build as on the previous one: 12.4% against 12.3%, 9.9% against 10.3%, 8.1% against 7.6% and 6.3% against 6.9% at the four draft-verified positions where they are most common |
| Images 64×64 to 4096×1024; 4096×4096 | correct; rejected with HTTP 400 as before |
| Tool call through the `inkling` parser | name and JSON arguments correct |
| Server errors over the suite | 0 |

The built image was compared file by file with the tested tree (42 Python files and the ranking, by sha256). Upstream's unit tests were not re-run.

### Tried on the way, not adopted

| Change | Result |
|---|---|
| The drafter producing only as many tokens as the scheduler will verify (a shorter block per step, its own CUDA graphs) | step 0.5 to 0.8 ms shorter at 2 to 4 drafts; prose and chat +2% to +4%, greedy JSON −7% (first draft accepted 0.80 against 0.85: short blocks draft predictable text worse), 2 requests −8%, GSM8K-200 193 / 200. Correct, not a net gain |
| Draft-length policies that react to the previous step (one estimate per previous outcome; a 2 / 4 / 7 ladder), replayed on a 12,000-step trace | within 1% of the running-estimate policy at any reaction delay. After a miss the first draft is accepted 0.58 of the time, after three or more accepted 0.76: too little to exploit |
| vLLM's adaptive verification (`enable_adaptive_verification`: verify length from draft confidence) | does not start: `Inkling2AttentionBackend` and `InklingSconvBackend` replay only decode batches of equal length. The same trace puts a perfect per-step verify length at +16% to +19% |

## Tuning

2026-11-21, evening. The build before the reduced draft vocabulary (image tag `-tune1`). Same cards and settings as below, with `patches/sm86-tune.diff` (the image `inkling-2-mini-vllm-2x3090:g7cb5dae22-tune1`, booted from `docker-compose.yml` as shipped unless a row says otherwise: drafter on, vision loaded, `--max-num-seqs 8`). "Untuned" is the port as measured earlier the same day; `INKLING_SM86_TUNE=none` reproduces it. "Tuned" columns are the final image. The change-by-change tables were measured as each change went in, so their absolute figures predate the last three rows of the step table where noted. Data: `bench-results/tuning/`; the same suite on the build before those three rows is in `tuning/before-router-and-tiles/`. Tools: `tools/tune/`.

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

### Depth on plain prose, with draft acceptance

`tools/tune/depth_accept.py`. Final image as shipped (images and drafter loaded, draft-length policy on), 2026-11-22. The prompt is the first N tokens of one text with a unique first line, and the task is the same at every depth ("about 400 words explaining what the material above covers"). Temperature 0.6, low reasoning effort, 512 tokens out. Acceptance is per step, from the server's draft counters.

War and Peace (Project Gutenberg; no repeated lines), three runs per depth, means (`tuning/depth_accept_warpeace.jsonl`):

| Prompt tokens | Prefill tok/s | To first token | Decode tok/s, drafter on | ms per step | Tokens per step | Accepted at draft position 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 2,061 | 6,570 | 0.31 s | 194.5 | 12.0 | 2.34 | 0.65 / 0.37 / 0.21 |
| 60,062 | 6,059 | 9.9 s | 153.4 | 13.0 | 2.00 | 0.55 / 0.27 / 0.14 |
| 110,062 | 5,283 | 20.8 s | 130.9 | 13.8 | 1.81 | 0.51 / 0.22 / 0.07 |
| 190,061 | 4,391 | 43.3 s | 122.8 | 15.8 | 1.94 | 0.52 / 0.26 / 0.13 |

The same run on the page-snapshot YAML that every other depth row in this file used (see [Depth corpus](#reproduce)), two runs per depth, means (`tuning/depth_accept.jsonl`):

| Prompt tokens | Prefill tok/s | To first token | Decode tok/s, drafter on | ms per step | Tokens per step | Accepted at draft position 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 2,061 | 6,815 | 0.30 s | 206.5 | 12.3 | 2.56 | 0.67 / 0.42 / 0.24 |
| 60,060 | 6,208 | 9.7 s | 169.7 | 13.5 | 2.29 | 0.59 / 0.37 / 0.20 |
| 110,062 | 5,422 | 20.3 s | 152.0 | 14.5 | 2.21 | 0.58 / 0.35 / 0.20 |
| 190,061 | 4,470 | 42.5 s | 132.2 | 15.8 | 2.09 | 0.59 / 0.34 / 0.14 |

- **Prefill is 2% to 3% lower on prose** than on the YAML at 60K to 190K (4% at 2K, where one run is noisy). The depth rows elsewhere in this file were measured on the YAML and are high by about that much for natural text.
- **With the drafter, decode at depth is 7% to 14% lower on prose**, because fewer drafts are accepted: 1.8 to 2.0 tokens per step against 2.1 to 2.3.
- **Acceptance falls from short context to depth on prose**: the first draft is accepted 0.65 of the time at 2K and 0.51 to 0.55 from 60K to 190K. The output is not identical across depths (the model is describing more material), so this is depth and task together.
- The step itself grows from 12.0 ms at 2K to 15.8 ms at 190K on either text.
- The other dual-3090 project's context-scaling protocol, with its own fixed task, gave 2.7 to 3.0 tokens per step from 1K to 60K with no trend (`tuning/published_protocols-drafter.json`, `acc`).
- These are the deepest fills of the shipped config: 190K of a 196,608 window, five times, no server errors, 20.2 GiB in use per card afterwards. The first 2K run on the YAML (765 tok/s prefill, kernel compilation after a restart) is left out.

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
| Prefill at 2,000 / 16,000 / 64,000 tokens | 7,480 / 7,140 / 6,280 | 7,405 / 7,031 / 6,094; later sessions 7,355 / 6,950 / 6,031, 7,254 / 6,909 / 5,998 and, after 20 minutes of load, 7,182 / 6,841 / 5,947 |

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

Upstream `tests/models/inkling2` at `fc144315` (the head of vllm-project/vllm#69028 at the time), gates widened with `tools/widen_test_gates.py`, on one RTX 3090. Recipe: [NOTES.md](NOTES.md#upstream-unit-tests).

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

- **Depth corpus:** `--corpus-dir` takes any tree of text; the planted facts and the question are the same, the filler text is yours. The depth rows in this file up to 2026-11-22 were meant to use the source and docs of a private config repo. The reader took files in path order, and a hidden directory of browser page snapshots (YAML accessibility-tree dumps) sorted first, so those prompts are that YAML, in which about one line in five repeats an earlier one. A unique first line keeps the prefix cache out either way. On plain prose prefill measures 2% to 3% lower and drafter-on decode at depth 7% to 14% lower ([Depth on plain prose](#depth-on-plain-prose-with-draft-acceptance)); drafter-off decode should not depend on the text and was not re-measured; retrieval was not re-run on prose. The reader now skips hidden files and directories.
- **Unpacked image:** the rows not labelled `packaged image` came from `tools/serve.sh '"device=0,1"' 8000 /tmp/serve.log --tensor-parallel-size 2 --max-model-len 196608 --max-num-batched-tokens 2048 --max-num-seqs 4 --gpu-memory-utilization 0.92 --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` on a copy of the image unpacked with `tools/fetch_rootfs.sh`, with the same kernel swap and patches applied.
