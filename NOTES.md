# Port notes

How the Ampere port of Inkling 2 Mini works, what the tuning changed, how it compares with the other Ampere routes, and how it was checked. To just run it, see the [README](README.md). All measurements: [BENCH.md](BENCH.md).
## Status (2026-11-21)

- **Not supported upstream.** vLLM's Inkling 2 code targets Blackwell (SM100/103), Hopper (SM90) and DGX Spark (SM120/121). [vllm-project/vllm#69028](https://github.com/vllm-project/vllm/pull/69028) merged on 2026-11-21. The `inkling-2-mini` image was pushed the day before from that branch, and no vLLM release includes the model yet.
- **An upstream Ampere path is open:** [vllm-project/vllm#69165](https://github.com/vllm-project/vllm/pull/69165). When it ships in an image or a release, start from that.
- **Pinned on purpose.** Base image by digest, wheel by sha256, every source edit anchored on the exact upstream text, the tuning patch applied with no fuzz. `docker build` fails on any other build of vLLM instead of patching it blindly.
- **Numerics follow vLLM's Hopper path:** BF16 activations against the NVFP4 / MXFP8 expert weights. The checkpoint was trained with quantized activations, which only Blackwell runs natively. Upstream reports the same GSM8K score on both paths, and the checks here agree.
- **Not the only Ampere route, and not the fastest on every row.** Twelve others appeared within two days of release, and their numbers move by the hour. [How it compares](#how-it-compares) has them next to these, including where they are ahead.

## What blocks Ampere, and what was changed

| # | Blocker | Where | Change |
|---|---|---|---|
| 0 | The image's compiled `_C` / `_moe_C` kernels are Blackwell-only (`TORCH_CUDA_ARCH_LIST="10.0+PTX 10.3+PTX"`): the first call fails with "no kernel image is available" | `install_sm86_kernels.py` | swap both for the general-architecture libraries from the per-commit wheel of the same main commit. Operator schemas are identical except the two fused inkling2 kernels below, which this path never calls |
| 1 | Capability floor: the `inkling2` quant method refuses anything below 9.0 before weights load | `quant_config.py` | 90 → 80 |
| 2 | Fused conv + mix kernel (`csrc/inkling2`) uses mbarriers and bulk async copies, built for sm_90/100/103 | `ops/conv_mix.py` | take the Triton `short_conv` + `attn_res` fallback that DGX Spark uses |
| 3 | Router GEMM uses TMA tensor descriptors and ~197 KB of shared memory (sm_86 has 99 KB) | `ops/linear.py` | take the DGX Spark path (no TMA, 2 stages) |
| 4 | About 25 Triton kernels use programmatic dependent launch (`griddepcontrol`, sm_90+ PTX) | Triton `gdc.py`, `compiler.py` | below sm_90 emit nothing and launch without PDL. Kernels then run in stream order, which is what `gdc_wait` waited for |
| 5 | Vision tower uses FlashAttention 3 (Hopper-only) | `nvidia/vision.py` | FlashAttention 2, as on DGX Spark |

Blockers 1–5 are `apply_sm86_patch.py`: anchored edits, idempotent, `.orig` copies kept, `--revert` to undo.

- **Not needed:** the expert GEMMs (the Humming library supports BF16 activations from sm_80 and JIT-compiles for the card), the attention kernel (Triton, fits sm_86's shared memory as written), NVFP4 rounding (software path), all-reduce (the model's fused NVLink-multicast collective is skipped off capability 9/10 and vLLM's own custom all-reduce takes over).
- **Not available with this port:** fp8 KV cache (the kernels in this build use Triton's `fp8e4nv`, sm_89+), and the fused all-reduce. #69165 adds a software fp8 path.

## Tuning

`patches/sm86-tune.diff`, applied after the port. A decode step without the drafter went from 8.1 ms to 6.95 ms (123 → 143 tok/s), and a step that verifies 7 drafted tokens from 14.9 ms to 13.0 ms. The reduced draft vocabulary then takes about 0.7 ms off every step with the drafter, and the last three rows are prefill: +13% at every depth above 8K, and +3% to +13% on prompts of 1.5K to 7K. Each row was measured on its own and passes the correctness gate ([BENCH.md](BENCH.md#tuning)).

| Change | Origin | Measured here |
|---|---|---|
| Small-matmul kernel for 1–16 rows in place of cuBLAS | snailbrainx; the tiles are fitted to sm_86's 99 KB of shared memory here | step 14.9 → 14.35 ms |
| Drafter's Markov fold on the same kernel | snailbrainx | → 14.25 ms |
| One launch for the q / k / v projections | snailbrainx | 14.9 → 14.5 ms on its own |
| Loop-controller kernels | snailbrainx | 14.9 → 14.3 ms on its own |
| Expert-GEMM kernel config for decode | here: sweep of 144 configs on a 3090 | 13.45 → 13.2 ms; without the drafter 7.8 → 7.4 ms |
| One-kernel expert alignment, single request only | snailbrainx; limited here to the sizes where its output is identical to the compiled op's | 13.2 → 13.0 ms; 7.4 → 7.1 ms |
| Draft length chosen per step from recent acceptance | here | against a fixed 7: prose +13%, chat +6%, code −1%, JSON −2% |
| 3 verified drafts when two or more requests run | a vLLM option, set in the compose file | +12% to +21% aggregate at 2–16 requests |
| Expert-GEMM kernel config for prefill | here | +2% prefill, bit-identical output |
| Long-prompt loop in the full-attention layers | snailbrainx | +1% prefill at 24K, +4% at 120K |
| Router matmul tile for decode (16 wide, 2 warps, 3 stages) | here: 8 configs | without the drafter 7.09 → 6.99 ms |
| Attention decode tile of 64 keys in place of 128 | here: 6 configs | → 6.95 ms; no change at depth |
| Full-attention prefill tile of 32 keys, 8 warps | here: 6 configs | +0.5% to +1% prefill at 24K–64K |
| Reduced draft vocabulary: the drafter scores the 24,576 likeliest token ids of each vocabulary half | here | step −0.65 to −0.72 ms with the drafter; greedy prose 172 → 182, chat 203 → 214, code 228 → 240, JSON 311 → 328 tok/s, same tokens per step |
| Conv-state cache backed only at the tail of each step, plus one block every 512 tokens | HyperInkling (the tail-only manager); the checkpoints are added here | KV pool at 2,048-token prefill chunks 223,030 → 391,858 tokens; decode and prefill unchanged |
| 8,192-token prefill chunks in place of 2,048 | a vLLM option, set in the compose file; it only fits with the row above | prefill 7,180 → 8,130 tok/s at 8K, 6,780 → 7,750 at 24K, 5,250 → 5,920 at 120K; KV pool 246,986 |
| The prefill expert-GEMM config at every batch above about 1,400 tokens, not only where the library already picked 64-row blocks, and 48-row blocks below about 3,600 | here: block sweeps at 12 batch sizes | prefill +13% at 1.7K tokens, +7% at 3K, +3% to +4% at 5K–7K, then +1.3% at 2K from the 48-row blocks; unchanged where chunks are full; bit-identical output |

`HyperInkling` is [Pento95/HyperInkling](https://github.com/Pento95/HyperInkling) (Apache-2.0), a single-card build. `snailbrainx` is [snailbrainx/inkling-2-mini-sm80](https://github.com/snailbrainx/inkling-2-mini-sm80) (Apache-2.0), tuned on an sm_80 card. Its tile sizes do not all run on a 3090 as published, and its small-matmul tiles, once they fit, are within 1–2% of the best of 114 configs per weight shape measured here.

- **Switch:** `INKLING_SM86_TUNE=none` in the container's environment restores the untuned port. A comma list of `gemv, fold, seg3, ctrl, humming, humming_big, align, far, router, tiles, adaptive, dvocab, convtail` keeps only those. Without `convtail` the server only boots with `--max-num-batched-tokens 2048`.
- **Draft length:** the drafter always drafts 7 tokens. The scheduler verifies between 2 and 7 of them, from a running estimate of how often drafts survive. Unverified drafts are dropped, so the output distribution is unchanged. The step costs it weighs were measured on this rig; `INKLING_SM86_STEP_MS` replaces them for another one.
- **Reduced draft vocabulary:** every step with the drafter spent about 1.2 ms on full-vocabulary work for it: a second pass over the output head (0.54 ms), a vocabulary-sized projection for each of the 7 draft positions (0.46 ms) and 7 transfers of full-vocabulary logits between the cards (0.23 ms). The drafter now scores 24,576 token ids per vocabulary half, 24% of them.
  - **Output is unchanged in distribution.** The kept logits are written into a full-vocabulary row that is minus infinity elsewhere, so a draft is sampled from, and verified against, the same distribution, and the model still verifies with its full vocabulary. A token outside the set can never be drafted; the model produces it itself. Checked by the correctness gate and by 6,000 seeded continuations at temperature 1 against the previous build ([BENCH.md](BENCH.md#reduced-draft-vocabulary)).
  - **The ranking is English text and code** (`patches/draft_vocab_rank.json`: model output, wikitext-2, HellaSwag, a novel, Python and C source, configs and docs). It covers about 99% of held-out text of those kinds. For other languages the drafter would rarely be able to propose the right token: set `INKLING_SM86_DVOCAB=off`, or build a ranking with `tools/tune/build_draft_vocab.py` and point `INKLING_SM86_DVOCAB` at it. `INKLING_SM86_DVOCAB_K` sets the size.
  - **Cost:** the KV pool is 1.3% smaller.
- **Conv-state cache:** each of the 48 layer visits keeps the inputs of its 4-tap convolutions in a paged cache, 16 KB per token per card. vLLM backs it for every token in flight (two prefill chunks under async scheduling), which reserved 3.4 GiB per card for one request at 2,048-token chunks, more than a 196K request's attention KV, and left no room for larger chunks. A step only reads the state of the 3 tokens before it, so this build backs the tail of each step and skips the stores for the rest of a chunk.
  - **Output is unchanged:** resumed state is bit-identical to vLLM's layout under the same chunking ([BENCH.md](BENCH.md#prefix-reuse)).
  - **Prefix cache:** tail-only state on its own loses the reuse of a prefix shared by different conversations. One block every 512 tokens, with `--prefix-cache-retention-interval 512` in the compose file, restores it and makes it apply from the second request instead of the third. The checkpoints follow that flag: at 0 there are none.
  - **What 8,192-token chunks cost:** the pool is 246,986 tokens instead of 391,858, and while one request prefills a long prompt the others get a batch of tokens about every 1.3 s instead of every 0.4 s (their rate during that prefill is 2 tok/s instead of 5). For the largest pool or the shortest pauses, set `--max-num-batched-tokens 2048`.
- **Tried, not adopted:** drafting only as many tokens as the scheduler will verify (shorter drafter blocks: prose and chat +2% to +4%, JSON −7%, because short blocks draft predictable text slightly worse), draft-length policies that react to the previous step (no better than the running estimate on a 12,000-step trace), a fixed draft of 4 (7 wins on predictable text), the other repo's expert-GEMM override (it does not launch on sm_86), its prefill tiles (no gain here), block verification (no gain), the draft-length policy for two or more requests (no gain), NCCL buffer, channel and NVLink chunk settings (no gain), a 16 MiB limit on vLLM's NVLink all-reduce so prefill chunks use it (no change), 16,384-token prefill chunks (2% over 8,192 for most of the KV pool), more splits of the decode attention at depth (no change on two cards), and the software fp8 KV path from #69165 ([Long windows](#long-windows)). Numbers: [BENCH.md](BENCH.md#tried-not-adopted).

## How it compares

Other Ampere routes for this model, with the numbers their authors published as of 2026-11-22 02:50 UTC; the two-3090 rows were checked again on 2026-11-23 13:45 UTC, when more single-card routes existed than are listed here. Hardware, power limits, weights and prompts differ, so only the "same protocol" tables below are like for like. None of the other projects' figures were reproduced here.

| Project | Hardware | Weights | Decode, no drafter | With drafter | Prefill | Window |
|---|---|---|---|---|---|---|
| **This repo** | 2× RTX 3090, NVLink, 300 W | original | 143 | 183–328 | 5.8–8.0K | 196K; with other flags 426K with images and the drafter, 467K–590K text only |
| [agent2x0r/inkling2-sm86-kernels](https://github.com/agent2x0r/inkling2-sm86-kernels) (vLLM) | 2× RTX 3090, NVLink, 225 W | original | 115 | 136–210 | 4.9–5.7K | 65K |
| [lulakien/inkling2-mini-3090-recipe](https://github.com/lulakien/inkling2-mini-3090-recipe) (vLLM) | 2× RTX 3090, PCIe, 280 W | original experts; attention, head and drafter in INT8; needs the `gpu-unlock` kernel module for its prefill figures | 130 | 126–313; 129–323 at 350 W | 6.3–7.5K; 6.6–7.8K at 350 W | 262K |
| [tonyd2wild/Inkling-2-Mini-W4A16-RTX3090](https://github.com/tonyd2wild/Inkling-2-Mini-W4A16-RTX3090) (vLLM) | 2× RTX 3090, NVLink, 240 W | experts re-quantized to INT4 | 94 | 138 | 3.1K at 32K | 459K |
| [vllm#69165](https://github.com/vllm-project/vllm/pull/69165) (vLLM, open PR) | 2× RTX 3090, PCIe | original | 96; 103 on its real prompts | 177 on its real prompts | not reported | 4K tested |
| [snailbrainx/inkling-2-mini-sm80#1](https://github.com/snailbrainx/inkling-2-mini-sm80/pull/1) (vLLM, open PR on the project below) | 2× RTX 3090, no custom all-reduce | original; bf16 or FP8 E5M2 KV | not reported here | not reported here | not reported here | 409,600 with bf16 KV and the drafter; 1,048,576 with FP8 E5M2 KV and no drafter |
| [snailbrainx/inkling-2-mini-sm80](https://github.com/snailbrainx/inkling-2-mini-sm80) (vLLM) | 1× CMP 170HX (sm_80, 64 GB) | original | not reported | 162–289 | 3.9–7.3K | 262K |
| [Pento95/HyperInkling](https://github.com/Pento95/HyperInkling) (vLLM) | 1× RTX 3090 Ti, 350 W | experts re-quantized to INT4, attention and head to INT8; emulated fp8 KV | 135–136 | 150–295; about 490 copying a table | 5.5–6.1K | 262K |
| [MKRWW/inkling2-ampere](https://github.com/MKRWW/inkling2-ampere) (vLLM) | 1× RTX 3090 | original, or all-NVFP4 | 72 | not reported | 0.8K | 176–256K |
| [skyne98/inkling-ingine-3090](https://github.com/skyne98/inkling-ingine-3090) (own engine) | 1× RTX 3090, 350 W | original, abliterated; dense weights and KV in fp8 | 140 | 219–239 | 4.3–5.7K | 64–256K |
| [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) (own engine) | 1× RTX 3090 | UD-Q4_K_XL GGUF | 128 | 150–210 | 2.1–2.7K | 262–344K |
| [rafaelkupper/inkling-2-mini-ninfer-3090](https://github.com/rafaelkupper/inkling-2-mini-ninfer-3090) (own engine) | 1× RTX 3090 | UD-IQ4_XS GGUF | 126 with a 2-token draft | | not reported | 65K |
| [JoaFoschiatti/inkling-2-mini-rtx3090-sm86](https://github.com/JoaFoschiatti/inkling-2-mini-rtx3090-sm86) (llama.cpp fork) | 1× RTX 3090, 280 W | UD-Q4_K_XL GGUF | 122–126 | 194 at 32K | 2.3K | 128K |
| [ManyGlue0/InklingEngine](https://github.com/ManyGlue0/InklingEngine) (own engine) | 1× RTX 3090 | UD-Q4_K_XL GGUF, 6-bit KV | 113–135 | not reported | 4.4K at 2K | 262K |

**Same protocol as agent2x0r** (its prompts and parameters: 512 tokens out, temperature 0.6, low reasoning effort; re-implemented in `tools/tune/published_protocols.py`). Their cards were capped at 225 W, these at 300 W.

| | Theirs | Here |
|---|---|---|
| Decode without the drafter, 1K / 16K / 60K context | 115 / 110 / 99 | 140 / 134 / 120 |
| Decode with the drafter, 1K / 16K / 60K | 172 / 178 / 170 | 240 / 248 / 215 |
| Prefill, 4K / 16K / 60K | 5,575 / 5,709 / 4,954 | 7,547 / 7,635 / 6,803 |
| Code / math / prose / JSON, drafter on | 202 / 178 / 136 / 210 | 262 / 231 / 195 / 262 |
| Aggregate at 1 / 2 / 4 / 8 users, drafter on | 191 / 278 / 353 / 383 | 268 / 369 / 423 / 494 |

**Same protocol as HyperInkling** (its prompts: 768 tokens out, temperature 0.2 or 0.5, default reasoning; its `doc` category needs a file that is not in its repo and is left out). Its numbers are from one 3090 Ti running an INT4 re-quantization; these are two 3090s running the original weights. Its benchmark script has not changed since 2026-11-21, so both of its columns are on the prompts re-run here.

| | Theirs, 2026-11-21 | Theirs, 2026-11-22 | Here |
|---|---|---|---|
| Without the drafter | about 135 | 135–136 | 142–143 |
| Chat | about 145 | about 150 | 159 |
| Code | about 230 | about 220 | 270 |
| Structured output | about 310 | about 295 | 340 |
| Prose | about 145 | about 155 | 171 |

One run each, as in their protocol. With the drafter on, a single run moves by 10–30% with how many drafts the sampled text lets through: over three builds the code row has read 226, 268 and 270.

**Same protocol as lulakien** (its prompts: 1,500 tokens out, not streamed, temperature 0, low reasoning effort; prefill as it measures it, a one-token completion over random token ids). Its server is text only at 280 W over PCIe (it also publishes a 350 W run), with INT8 attention weights and the [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, which its README says its prefill figures depend on. Here: stock cards, original weights, vision loaded. `tools/tune/more_protocols.py`.

| | Theirs | Here |
|---|---|---|
| Code / explanation / story, drafter on | 313 / 166 / 126; at 350 W 323 / 170 / 129 | 356 / 196 / 165 |
| Aggregate at 2 / 4 / 8 streams | 260 / 339 / 445 | 336 / 420 / 687 |
| Without the drafter (here on other prompts; it is flat across prompts) | 130 | 143 |
| Prefill at 2K / 16K / 64K | 7,480 / 7,140 / 6,280; at 350 W 7,840 / 7,480 / 6,620 | 7,275–7,345 / 7,643–7,653 / 6,616–6,618 |

**Same prompts as ingine** (512 tokens out, greedy, three runs, rate after the first token). Its reasoning-effort setting is not published; these are at the template default, and at low effort the two rows are 286 and 313. Its 140 tok/s without a drafter is on short prompts (its "32K" is the window setting), so the comparable figure here is 143.

| | Theirs | Here |
|---|---|---|
| Chat | 235–239 | 249–255 |
| Code | 219–224 | 278–280 |
| Without a drafter (here on other prompts) | 140 | 143 |
| Without a drafter, after a 100K prompt (here 120K, other text) | 105 | 107 |

Where this build is ahead:

- **lulakien's, ingine's and HyperInkling's own prompts:** every decode and concurrency row above.
- **The other NVLink dual-3090 port:** every row of agent2x0r's protocol, above.
- **Decode without a drafter:** 143 tok/s against 72–140 for every project that publishes one.
- **Prefill above one chunk:** on lulakien's protocol 7,643–7,653 tok/s at 16K and 6,616–6,618 at 64K, against its 7,140 / 6,280 at 280 W and 7,480 / 6,620 at 350 W with the kernel module. snailbrainx reports 7,264 / 6,802 at 8K / 32K on its sm_80 card; here 7,980–8,130 at 8K and 7,330 at 33K on other prompts.
- **Window on a bf16 KV cache:** 425,984 with images and the drafter, 466,944 text only with the drafter, 589,824 without it, each booted and filled ([Long windows](#long-windows)), against 409,600 (the snailbrainx PR, text only with the drafter) and tonyd2wild's 458,752 on INT4 experts.
- **Many requests at once:** with `--max-num-seqs 64`, 1,167 tok/s at 16 requests, 1,634 at 32 and 2,058 at 64 with the drafter (2,114 at 64 without it), on this repo's mixed prompts. vllm#69165 reports 663 at 16 and 781 at 64 with the drafter and 1,144 at 64 without, on its own prompts at a 4K window.
- **Several requests at once, among the dual-3090 routes:** 747–778 tok/s at 8 mixed requests, against 445 for lulakien and 383 for agent2x0r. Against snailbrainx's 8-user figures it is level on code and prose (762 and 726 here on the previous build, 748 and 683 there).

Where someone else reports more, as of 2026-11-23:

- **Prefill of a 2,000-token prompt:** lulakien reports 7,480 tok/s at 280 W and 7,840 at 350 W. Here 7,275–7,345 on its protocol with cards warm from a 15-minute suite and 7,508–7,530 on the same kind of request shortly after boot, so level with its 280 W figure at best and behind its 350 W one. Both cards sit at their 300 W cap for the whole prefill here.
- **Very repetitive output, partly:** HyperInkling reported about 490 tok/s copying a table out of a long document; its filler text is not published, so that was not re-run. snailbrainx reports 289 tok/s on lists and 1,243 across 8 users without publishing its prompts; on a list task of this repo's own (`tools/tune/lists_bench.py`, greedy, no reasoning) the previous build gave 476–486 and 1,481, which is not a like-for-like comparison.
- **Window with a quantized KV cache:** the snailbrainx PR reports the model's native 1,048,576 on two 3090s with an FP8 E5M2 KV cache and no drafter. Not offered here: this build keeps the bf16 cache.

The builds that are ahead on a row re-quantize the dense weights or the KV cache, use a kernel module that changes the cards' firmware state, or run at 350 W. This one serves the checkpoint as released, bf16 KV, stock cards at 300 W.

## Long windows

The shipped config loads the vision tower and the drafter, prefills in 8,192-token chunks and serves 196,608 tokens. The same image serves longer windows on the bf16 cache with smaller chunks, and longer again text only. Each row is a change to the flags in `docker-compose.yml`, plus `--max-num-seqs 4`, booted and filled with a retrieval prompt ([BENCH.md](BENCH.md#long-windows)).

| Mode | Flags changed | KV pool | Window | Prompt | Facts found | Prefill tok/s | To first token |
|---|---|---|---|---|---|---|---|
| As shipped | | 246,986 | 196,608 | 190,178 | 5 / 5 | 5,061 | 38 s |
| Images, drafter | `--max-model-len 262144` | 281,123 | 262,144 | 250,180 | 5 / 5 | 4,496 | 56 s |
| Images, drafter, small chunks | `--max-num-batched-tokens 2048`, `--max-model-len 425984` | 436,045 | 425,984 | 415,174 | 5 / 5 | 3,003 | 138 s |
| Text only, drafter | the same chunks, `--limit-mm-per-prompt '{"image":0}'`, `--gpu-memory-utilization 0.95`, `--max-model-len 466944` | 471,527 | 466,944 | 456,176 | 5 / 5 in two fills; a third ran to the token cap without answering | 2,842 | 161 s |
| Text only, no drafter | the same chunks, `--limit-mm-per-prompt '{"image":0}'`, no `--speculative-config`, `--gpu-memory-utilization 0.97`, `--max-model-len 589824` | 607,708 | 589,824 | 578,175 | 5 / 5 | 2,433 | 238 s |

- **Not the default** because the long modes give up prefill speed (2,048-token chunks), then images, then the drafter, and a full window takes minutes to prefill.
- **Decode at these depths:** 56 tok/s without the drafter after the 578K prompt. With the drafter 127–239 on these short, predictable answers; on plain prose expect about 130 at 190K and less beyond.
- **Before the conv-state change** the text-only ceilings were 262,144 with the drafter and 368,640 without ([BENCH.md](BENCH.md#long-windows-1)).
- **Software fp8 KV cache** (the approach of #69165, tried on an earlier build without the drafter): the pool grew to 374,234 tokens with the vision tower loaded and retrieval stayed 5 / 5 to 120K, but prefill fell from 7,100 to 2,306 tok/s at 8K and decode from 143 to 123 tok/s. Not adopted; the patch does not include it.

## Serving notes

- **Reasoning effort:** send `reasoning_effort`, or `chat_template_kwargs: {"reasoning_effort": "low"}`. Levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; the template default is `high`. `none` still reasons.
- **Cap `max_tokens`.** The model can loop in its reasoning until the cap: 1 of 200 GSM8K questions here.
- **Images:** up to 3584×3584 px (8,168 image tokens). Larger ones get an HTTP 400; downscale on the client.
- **`--max-num-batched-tokens` is 8192.** It is the prefill chunk and the image ceiling. 2048 gives a larger KV pool and shorter pauses for other streams during a long prefill, at 13% less prefill ([Tuning](#tuning)).
- **`--prefix-cache-retention-interval 512`** lets a request reuse a prefix it shares with an earlier, different conversation, up to the last multiple of 512 tokens.
- **Output:** reasoning arrives in `reasoning`; tool calls use the `inkling` parser.
- **Drafter:** on in the compose file, 7 drafted tokens. The `num_speculative_tokens_per_batch_size` entry verifies 7 for one request and 3 for two or more. Delete the two speculative-config lines to turn the drafter off.
- **Concurrency:** `--max-num-seqs 8`. With `--max-num-seqs 64`, and the draft schedule's upper bound raised to match (`[[1,1,7],[2,64,3]]`), the aggregate is 1,167 tok/s at 16 requests, 1,634 at 32 and 2,058 at 64, and the KV pool is 200,397 tokens. The per-step draft length applies to one request at a time; with two or more it is the fixed 3, which measured as fast and looped less on GSM8K.

## Other setups

Only the first two rows were run.

| Setup | Status | Notes |
|---|---|---|
| 2× RTX 3090, NVLink, TP2 | measured | `docker-compose.yml` as shipped |
| 1× RTX 3090, TP1 | measured on the untuned port, not practical | text only, no drafter: 17.64 GiB of weights, KV pool 8,639 tokens, 95 tok/s. Flags: `--tensor-parallel-size 1 --max-model-len 8192 --max-num-batched-tokens 1024 --max-num-seqs 2 --gpu-memory-utilization 0.95 --limit-mm-per-prompt '{"image":0}'`, no `--speculative-config` |
| 2× 24 GB sm_86 without NVLink | not tested | the compose file assumes a bridge (custom all-reduce on). #69165 reports 96 tok/s on two bridgeless 3090s with its own patch set |
| 1× 48 GB sm_86 (RTX A6000) at TP1 | not tested | at TP1 the weights take 17.64 GiB, so the window should be large |
| sm_80 (A100, CMP 170HX), sm_89 (RTX 40-series, L4) | not tested | the port's patches gate on capability major 8, so these take the same paths. The expert-GEMM configs apply on sm_86 only; the draft-length cost table is this rig's |

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | official image by digest + `install_sm86_kernels.py` + `apply_sm86_patch.py` + `patches/sm86-tune.diff`; each fails the build on any mismatch |
| `docker-compose.yml` | the serving config, commented flag by flag |
| `install_sm86_kernels.py` | blocker 0: downloads the general-architecture wheel (sha256-pinned) and replaces the two libraries |
| `apply_sm86_patch.py` | blockers 1–5 |
| `patches/sm86-tune.diff` | the tuning: eight model files, the scheduler hook and the draft-length policy |
| `tools/preflight.py` | can this image's kernels run on the card at all: compiled kernels, patched Triton, capability helpers |
| `tools/vbench.py` | decode / prefill / retrieval bench |
| `tools/gsm8k_eval.py`, `tools/image_sweep.py` | GSM8K with vLLM's chat protocol; image-shape sweep |
| `tools/tf_probe.py` | token-level agreement between two servers, and a passage-NLL sanity check |
| `tools/tune/` | what the tuning was measured with: step bench, correctness gate, kernel microbenchmarks, profiler summary, concurrency bench, the two published protocols ([README](tools/tune/README.md)) |
| `tools/widen_test_gates.py`, `tools/tests/conftest.py`, `tools/humming_seedcheck.py` | run upstream's unit tests on sm_8x |
| `tools/fetch_rootfs.sh`, `tools/vrun.sh`, `tools/serve.sh` | run the image without storing it in Docker (unpack to a directory, bind-mount over a small CUDA base). How the port was developed while the Docker root was full. Library paths are for an Arch host; not needed to use the port |
| `bench-results/` | CSVs behind BENCH.md, with a column dictionary; `bench-results/tuning/` for the tuned build |

## How it was checked

A boot is not evidence. Each stage had to pass before the next.

| Stage | Proof | Result |
|---|---|---|
| 0 Preflight | compiled kernels execute; a Triton kernel with `gdc_wait` + `launch_pdl` compiles and runs on sm_86 | pass (after the kernel swap) |
| 1 Kernels, one 3090 | upstream unit tests: core ops, attention, expert layers | pass on everything this path uses; remaining failures are fp8-only kernels |
| 2 Tiny looped model | decode matches prefill, generation with and without the drafter | pass |
| 3 One card, real checkpoint | loads in 17.64 GiB, 95 tok/s, passage NLL 1.77 | pass |
| 4 Two cards | boots in 41 s, custom all-reduce over NVLink, 122 tok/s | pass |
| 5 Correctness | passage NLL; TP2 vs TP1 teacher-forced; GSM8K-200; retrieval to 120K | pass |
| 6 Drafter | DSpark, 7 tokens, probabilistic | works; GSM8K 96.5% vs 98.0% (n.s.) |
| 7 Vision | image sweep | pass to ~2080 px |
| 8 Packaged image | `docker compose build`, boot from the compose file, re-run the checks | pass |
| 9 Tuned image | every tuning A/B'd on its own; built image compared file by file with the tested tree; booted from the compose file | against the one-card reference the top token agrees at 98.4% of positions, median \|Δ logprob\| 0.0015 (untuned 98.8%, 0.0012); decode agrees with prefill at 99.0% (untuned 98.6%) and at 12K depth 98.1% (untuned 98.1%); GSM8K-200 98.0%; retrieval 5 / 5 to 120K; images and a tool call pass |

Stages 1–7 ran on the untuned port. Upstream's unit tests were not re-run on the tuned tree.

To check a build on your own cards:

```bash
# kernels (one GPU, no model needed)
docker run --rm --gpus '"device=0"' -v $PWD/tools:/work --entrypoint python3 \
    inkling-2-mini-vllm-2x3090:g7cb5dae22-tune2 /work/preflight.py

# against the running server
tools/tf_probe.py nll http://localhost:8000                # about 1.77 nats/token; a broken forward pass sits far above 3
tools/tf_probe.py compare http://localhost:8000 bench-results/tf_reference_tp1.json
tools/gsm8k_eval.py "my box" -n 200 --concurrency 4 --csv /tmp/gsm8k.csv
tools/image_sweep.py "my box" --csv /tmp/images.csv
tools/vbench.py "my box" http://localhost:8000 /tmp/runs.csv --depth-tokens 8000 24000 --corpus-dir <any source tree>
tools/tune/bench_step.py "my box" /tmp/steps.jsonl         # ms per decode step, the number to tune on
```

### Upstream unit tests

Run on one card with no server on it.

1. Make a directory `work/` holding `tests/models/inkling2/` and `tests/v1/e2e/spec_decode/utils.py` from vLLM at `fc144315` (the head of #69028 when this was done), empty `__init__.py` files under `tests/v1/`, and `tools/tests/conftest.py` as `work/tests/conftest.py`.
2. `python3 tools/widen_test_gates.py work` (anchored; fails if the tests have changed).
3. Per file:

```bash
docker run --rm --gpus '"device=0"' --ipc host -v $PWD/work:/work -w /work \
    -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 --entrypoint python3 \
    inkling-2-mini-vllm-2x3090:g7cb5dae22-tune2 -m pytest tests/models/inkling2/test_core_ops.py -q -p no:cacheprovider
```

Results on the untuned port: [BENCH.md](BENCH.md#unit-tests).

## Open items

| Item | Why it matters |
|---|---|
| Upstream's unit tests on the tuned tree | the tuning was gated end to end, not by the unit tests |
| #69165 on this rig | NVLink, the drafter and vision are outside what that PR reports. Only its software fp8 KV path was tried here ([Long windows](#long-windows)) |
| The drafter's fixed cost | a step with one verified draft costs 10.0 ms against 6.95 with none. The drafter still drafts all 7 when few are verified |
| Prefill all-reduce | 11.5% of prefill is NCCL all-reduce, 96 per 2,048-token chunk. Moving those to vLLM's own NVLink kernel (limit raised to 16 MiB) changed nothing, and larger chunks do not boot |
| Prefill above 300 W | both cards sit at the cap for the whole prefill. Not tried higher |
| Reasoning loops against draft length, on the full GSM8K (1,319) | across six runs of the first 200, the two runs that verified up to 7 drafts had 3 questions loop to the token cap and the four runs with 3 or none had 1 ([BENCH.md](BENCH.md#reasoning-loops-and-draft-length)). Too few to call |
| Sustained use near the end of the window | the shipped config was filled to 190K five times and the text-only modes close to their windows once each; none was run for long |
| Sustained serving | only short runs so far |
