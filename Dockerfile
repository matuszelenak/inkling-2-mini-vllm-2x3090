# Inkling 2 Mini on Ampere (sm_86): the official vLLM image for this model plus
# the changes needed to run it on RTX 3090s. Not supported upstream: vLLM targets
# Blackwell, Hopper and DGX Spark for this model (vllm-project/vllm#69028, merged
# 2026-11-21, not in a release yet). An upstream Ampere path is proposed in
# vllm-project/vllm#69165.
#
# Base: the image the vLLM recipe names, pinned by index digest (tag pushed
# 2026-11-20; vLLM 0.1.dev25855+g7cb5dae22, torch 2.14.0+cu130, Triton 3.8.0,
# humming-kernels 0.1.20). Both scripts check they are patching that build
# and fail the build otherwise: bump the digest and re-derive, do not force.
FROM vllm/vllm-openai:inkling-2-mini@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e

COPY install_sm86_kernels.py apply_sm86_patch.py patches/sm86-tune.diff patches/draft_vocab_rank.json /tmp/sm86/

# 1. Compiled kernels. The image's _C / _moe_C extensions are Blackwell-only;
#    swap in the general-arch ones from the matching per-commit wheel
#    (downloaded from wheels.vllm.ai, sha256-pinned).
# 2. Python. Five anchored edits: capability floor 90 -> 80; conv+mix, the
#    router GEMM and the vision tower onto the fallbacks DGX Spark already
#    uses; Triton emits no griddepcontrol and requests no PDL launch below
#    sm_90.
# 3. Tuning (patches/sm86-tune.diff, applied with no fuzz): faster decode and
#    prefill on Ampere. INKLING_SM86_TUNE=none at run time switches it off.
#    patches/draft_vocab_rank.json is the token ranking its reduced draft
#    vocabulary reads (English text and code; NOTES.md, "Tuning").
RUN set -eux; \
    site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    test -d "$site/vllm/models/inkling2"; \
    python3 /tmp/sm86/install_sm86_kernels.py "$site"; \
    python3 /tmp/sm86/apply_sm86_patch.py "$site"; \
    patch -p1 --fuzz=0 -d "$site" < /tmp/sm86/sm86-tune.diff; \
    install -m 644 /tmp/sm86/draft_vocab_rank.json "$site/vllm/models/inkling2/draft_vocab_rank.json"; \
    python3 -m py_compile "$site/vllm/models/inkling2/quant_config.py" \
        "$site/vllm/models/inkling2/nvidia/ops/conv_mix.py" \
        "$site/vllm/models/inkling2/nvidia/ops/linear.py" \
        "$site/vllm/models/inkling2/nvidia/vision.py" \
        "$site/vllm/models/inkling2/nvidia/attention.py" \
        "$site/vllm/models/inkling2/nvidia/dspark.py" \
        "$site/vllm/models/inkling2/speculator.py" \
        "$site/vllm/models/inkling2/nvidia/loop.py" \
        "$site/vllm/models/inkling2/nvidia/ops/gemv.py" \
        "$site/vllm/models/inkling2/nvidia/ops/moe_sm90.py" \
        "$site/vllm/models/inkling2/nvidia/ops/moe_align_small.py" \
        "$site/vllm/models/inkling2/nvidia/ops/attention_sm90.py" \
        "$site/vllm/v1/core/sched/scheduler.py" \
        "$site/vllm/v1/spec_decode/dynamic/adaptive_k.py" \
        "$site/triton/language/extra/cuda/gdc.py" \
        "$site/triton/backends/nvidia/compiler.py"; \
    rm -rf /tmp/sm86
