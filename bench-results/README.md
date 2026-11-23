# bench-results

All runs: 2× RTX 3090 at 300 W, NVLink, vLLM `0.1.dev25855+g7cb5dae22` with this repo's kernel swap and patches, TP2 + EP2 unless a row says TP1, bf16 KV, 2026-11-21.

| File | Written by | One row is |
|---|---|---|
| `runs.csv` | `../tools/vbench.py` | one workload on one server config |
| `gsm8k.csv` | `../tools/gsm8k_eval.py` | one GSM8K question on one arm |
| `images.csv` | `../tools/image_sweep.py` | one image shape |
| `token_agreement.csv` | `../tools/tf_probe.py` (`collect`, `compare`, `nll`) | one candidate server against the reference |
| `tf_reference_tp1.json` | `../tools/tf_probe.py collect` | reference token ids and teacher-forced scores from the one-card server |
| `unit_tests.csv` | hand-collected from pytest summaries | one upstream test file |

## tuning/

The tuned build ([BENCH.md](../BENCH.md#tuning)).

| File | Written by | One row is |
|---|---|---|
| `runs.csv`, `gsm8k.csv`, `images.csv` | the same tools as above | the final tuned image booted from `docker-compose.yml`; `runs.csv` also has a no-drafter pass |
| `depth_accept_warpeace.jsonl`, `depth_accept.jsonl` | `../tools/tune/depth_accept.py` | one request at one context depth on the final image as shipped: prefill and decode rate, decode steps, tokens per step, acceptance by draft position. The first is on War and Peace, the second on the page-snapshot YAML the other depth rows used |
| `long_window.csv` | `../tools/vbench.py` | one depth prompt on a text-only, long-window boot of the same image (columns as `runs.csv`; the label gives the mode and window) |
| `conv-state-tail/` | the same tools, plus `../tools/tune/prefix_reuse.py` and `stream_beside_prefill.py` for the tables that have no file here | the current image (`-tune3`): the full suite; `prefill_by_chunk.csv` (one boot per prefill chunk size; the label names it) and the same on vLLM's conv-state layout at a 20,000-token window; `prefill_depth.csv`; `long_window.csv` (one boot per mode, the label gives mode and window); `images_large.csv`; `decode_depth_by_attention_splits.jsonl` (not adopted); `memory_stress_*.csv`; `mb_humming_prefill_*.jsonl` and `prefill_by_expert_block.csv` (expert-GEMM row blocks: the microbenchmark, and the server before and after); `final-image/` (the suite again on the final image) and `before-48-row-blocks/` (the same one change earlier); `concurrency_64_seqs.jsonl` (`--max-num-seqs 64`, with and without the drafter) |
| `reduced-draft-vocab/` | the same tools, plus `../tools/tune/sampling_check.py` | the previous image (`-tune2`): the full suite, the previous build's step bench from the same morning, and the token ids of 6,000 seeded continuations on each build |
| `before-router-and-tiles/` | the same tools | the same suite on the tuned build before its last three changes (router tile, attention decode tile, full-attention prefill tile). `gsm8k_run2.csv` and `concurrency_final_run2.jsonl` are repeats after a restart |
| `step_cost_by_batch.jsonl` | `../tools/tune/bench_concurrency.py --ignore-eos` with every verified draft accepted | one batch size at one verified-draft count; ms per step = 1000 × batch × (drafts + 1) / aggregate tok/s |
| `policy_all_batches_gsm8k.csv`, `policy_all_batches_concurrency.jsonl` | as above | the draft-length policy applied at every batch size (not adopted) |
| `steps_by_arm.jsonl` | `../tools/tune/bench_step.py` | one request on one arm: tok/s, steps, tokens per step, ms per step, acceptance by draft position. `label` names the arm (which tunings were on, how many draft tokens) |
| `steps_final.jsonl` | `../tools/tune/bench_step.py` | the same on the final tuned image, greedy and at temperature 0.6 |
| `steps_router_and_decode_tiles.jsonl` | `../tools/tune/bench_step.py` | one no-drafter request per router or attention-decode tile config; `label` gives the tile as block, warps, stages |
| `prefill_by_arm.csv` | `../tools/vbench.py` | one depth prompt on one arm (columns as `runs.csv`) |
| `prefill_by_attention_tile.csv`, `depth_decode_by_tile.csv` | `../tools/vbench.py` | one depth prompt per attention tile config: prefill tiles with the drafter on, decode tiles with it off. The first 8K row of an arm can include kernel compilation |
| `prefill_by_nccl_setting.csv`, `prefill_by_allreduce_limit.csv` | `../tools/vbench.py` | one depth prompt per NCCL environment setting, and per size limit of vLLM's NVLink all-reduce (not adopted) |
| `fp8_software_kv.csv` | `../tools/vbench.py` | the software fp8 KV path, no drafter (not adopted, not in the patch) |
| `concurrency_by_draft_length.jsonl`, `concurrency_final.jsonl`, `concurrency_by_kind.jsonl` | `../tools/tune/bench_concurrency.py` | one concurrency level: aggregate and per-stream tok/s |
| `published_protocols-drafter.json`, `-no-drafter.json` | `../tools/tune/published_protocols.py` | per-trial values for the two re-implemented protocols |
| `more_protocols-drafter.json` | `../tools/tune/more_protocols.py` | per-run values for two more re-implemented protocols, with the figures those projects published |
| `mb_gemv.jsonl` | `../tools/tune/mb_gemv.py` | one weight shape: cuBLAS µs and the 40 fastest tile configs `(BN, BK, warps, stages)` |
| `mb_humming_nvfp4.jsonl`, `mb_humming_mxfp8.jsonl` | `../tools/tune/mb_humming.py` | one batch size: the heuristic config's µs and the 40 fastest candidates |

Arm labels in `steps_by_arm.jsonl`: `stock` and `overlay off` are the untuned port; `gemv`, `fold`, `seg3`, `ctrl`, `humming`, `align`, `far` are tunings of `INKLING_SM86_TUNE` (`router` and `tiles` came later and are not in this file); `ship` is the full set; `k=N` is the number of draft tokens; `dyn k=N of 7` drafts 7 and verifies N; `adaptive on` / `off` is the per-step draft length.

## Labels

| Label fragment | Meaning |
|---|---|
| `c196608` | `--max-model-len 196608` |
| `mbt2048` | `--max-num-batched-tokens 2048` |
| `no-drafter` | no `--speculative-config` |
| `dspark7-prob` | `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}` |
| `packaged image` | the image built from the Dockerfile, booted from `docker-compose.yml` |

## runs.csv

| Column | Meaning |
|---|---|
| `workload` | one of the four short prompts in `vbench.py`, or `deep N`: an N-token prompt of filler text with five planted facts (what the filler was: see below) |
| `temperature` | sampling temperature of the request |
| `prompt_n`, `gen_n` | `usage.prompt_tokens` and `usage.completion_tokens` |
| `ttft_s` | seconds from request to the first streamed token (content, reasoning or tool call) |
| `pp_tok_s` | `prompt_n / ttft_s`. Meaningful on the deep rows only |
| `tg_tok_s` | `(gen_n - 1)` over the time after the first token |
| `needles`, `found` | planted facts and how many came back as `NAME=NUMBER` (deep rows) |
| `finish_reason`, `wall_s` | as returned; client wall time |

Every request starts with a unique line, so the prefix cache does not help prefill. The deep rows were meant to use the source and docs of a private config repo as filler. The reader took files in path order and a hidden directory of browser page snapshots (YAML accessibility-tree dumps, about one line in five repeating an earlier one) sorted first, so the prompts up to 2026-11-22 are that YAML. On plain prose prefill measures 2% to 3% lower and drafter-on decode at depth 7% to 14% lower (`tuning/depth_accept_warpeace.jsonl` against `tuning/depth_accept.jsonl`); retrieval was not re-run on prose. `vbench.py` takes the tree as `--corpus-dir` and now skips hidden files and directories.

## gsm8k.csv

`label, idx, gold, pred, correct, finish_reason, completion_tokens, wall_s`. The two arms join on `idx`. `wall_s` is per request under concurrency 4.

## images.csv

`label, width, height, ok, correct, prompt_n, gen_n, wall_s, gpu0_used_mib, gpu1_used_mib, error`. `prompt_n` and `gen_n` are empty in these rows: the script read them from a field vLLM does not return, and now reads `usage`. `error` holds the HTTP body for rejected shapes.

## token_agreement.csv

| Column | Meaning |
|---|---|
| `candidate`, `reference` | server being scored, and the server whose greedy tokens are scored |
| `positions` | generated positions compared (8 prompts × 192 tokens) |
| `same_top1` | share of positions where both servers rank the same token first |
| `reference_token_rank1` | share where the reference's greedy token is rank 1 on the candidate under teacher forcing |
| `logprob_gap_*` | absolute difference in the reference token's log-probability |
| `passage_nll_nats_per_token` | mean NLL of a fixed 168-token English passage on the candidate |

## unit_tests.csv

| Column | Meaning |
|---|---|
| `file` | file under upstream `tests/models/inkling2` (`fc144315`) |
| `passed`, `failed`, `skipped`, `errors` | pytest summary on one RTX 3090 |
| `failures_on_path` | failures in code the Ampere serving path uses |
| `note` | what the remaining failures are |
