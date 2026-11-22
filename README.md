# Inkling 2 Mini on vLLM, on a pair of RTX 3090s

vLLM does not run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Ampere cards. This repo is a Dockerfile that patches the official vLLM image so it does, a tuning patch that makes it fast there, the serving config for two RTX 3090s on an NVLink bridge, and the measurements.

| | |
|---|---|
| Model | `thinkingmachines/Inkling-2-Mini`, the original checkpoint (experts in NVFP4 / MXFP8, 18.2 GiB), with its bundled DSpark drafter and vision tower |
| Engine | `vllm/vllm-openai:inkling-2-mini` by digest (vLLM `0.1.dev25855+g7cb5dae22`): compiled kernels swapped, five source patches, one tuning patch |
| Tested on | 2× RTX 3090 (sm_86, 24 GB) at 300 W, NVLink, tensor parallel 2 |
| Window | 196,608 tokens (KV pool 223,030); filled to 190K. Text only: 262,144 with the drafter, 368,640 without ([long windows](NOTES.md#long-windows)) |
| Decode, one request | 143 tok/s without the drafter; with it 182 (prose), 214 (chat), 240 (code), 328 (JSON tool calls), medians of three |
| Decode at depth | 131 tok/s at 32K, 107 at 120K, without the drafter; with it, 143 at 110K and 130 at 190K on plain prose |
| Prefill | 7,030 tok/s at 8K, 6,700 at 24K, 5,170 at 120K (23 s to first token) |
| Several requests | 534 tok/s aggregate at 4, 778 at 8 |
| Images | up to about 2080×2080 px |
| Checks | GSM8K-200 97.0% to 98.0% over five runs; retrieval 5 / 5 to 120K; two cards match one card token for token within the model's own decode/prefill noise |
| llama.cpp on the same cards | 50 tok/s decode, 564 tok/s prefill at 24K |

All numbers: [BENCH.md](BENCH.md). How the port works, the tuning, and how it compares with the other Ampere routes: [NOTES.md](NOTES.md).

## Serving

You need:

| | |
|---|---|
| GPUs | two 24 GB sm_86 cards. Measured on RTX 3090s with an NVLink bridge. Other cards and bridgeless pairs: [other setups](NOTES.md#other-setups) |
| Driver | one that supports CUDA 13 (the image is torch 2.14.0+cu130). Tested with 620.73.04 |
| Docker | with the NVIDIA Container Toolkit. Tested with Docker 29.10, Compose 5.8, toolkit 1.20.1 |
| Disk | 26 GB for the image, 20 GB for the checkpoint |
| Host | x86_64 Linux |

Download the checkpoint, build the patched image, and start the server:

```bash
git clone https://github.com/peaster/inkling-2-mini-vllm-2x3090
cd inkling-2-mini-vllm-2x3090

hf download thinkingmachines/Inkling-2-Mini   # 20 GB into ~/.cache/huggingface
docker compose build                          # pulls the 9.3 GiB base image, downloads one wheel, patches
docker compose up -d
docker compose logs -f                        # ready at "Application startup complete"
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Name three uses for a spare GPU."}],
  "max_tokens": 2048,
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

- **Boot:** about 40 s with warm JIT caches and the checkpoint in page cache, 91 s with cold caches, longer on the first read of the checkpoint from slow storage. The first large prompt after a cold boot also pays a few seconds of kernel compilation.
- **Checkpoint:** the container reads `~/.cache/huggingface` read-only with `HF_HUB_OFFLINE=1`. It never downloads at boot.
- **Port:** 8000, no API key. `docker-compose.yml` has a note on binding it to localhost.

Things to know when you call it:

- **Reasoning effort:** send `reasoning_effort`, or `chat_template_kwargs: {"reasoning_effort": "low"}`. Levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; the template default is `high`. `none` still reasons.
- **Cap `max_tokens`.** The model can loop in its reasoning until the cap.
- **Output:** reasoning arrives in `reasoning`; tool calls use the `inkling` parser.
- **Images:** up to about 2080×2080 px. Larger ones get an HTTP 400; downscale on the client.
- **Drafter:** on by default. Delete the two speculative-config lines in `docker-compose.yml` to turn it off.
- **Concurrency:** `--max-num-seqs 8`. Every flag is commented in `docker-compose.yml`.
- **Upstream:** vLLM has an open Ampere PR, [vllm-project/vllm#69165](https://github.com/vllm-project/vllm/pull/69165). When it ships in a release, start from that instead of this Dockerfile.

## Write-up

The story of the port, the comparison with the other Ampere routes, and an eval of the model itself: [Inkling 2 Mini, First Impressions](https://peaster.io/articles/local-LLMs/inkling-2-mini-first-impressions).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). This repo holds no vLLM binaries and no weights: the build downloads the official image and one official wheel. Part of the tuning patch is adapted from snailbrainx/inkling-2-mini-sm80 (Apache-2.0). Inkling 2 Mini is released by Thinking Machines under Apache-2.0.
