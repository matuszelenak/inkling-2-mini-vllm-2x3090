"""The tail-only conv-state manager of patches/sm86-tune.diff at this deployment's sizes: 8-token conv blocks,
a 4-tap window, 7 drafts, two steps in flight, prefix hits aligned to 64 tokens, 2048- to 8192-token chunks.
No GPU, no model. Run it inside the built image:

    docker run --rm --entrypoint python3 -v $PWD/tools/tune:/t:ro -w /tmp \
        inkling-2-mini-vllm-2x3090:g7cb5dae22-tune3 -m pytest -q -p no:cacheprovider /t/test_conv_tail.py
"""
import pytest
import torch

from vllm.models.inkling.nvidia.sconv_swa_attn import SconvStateManager
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup
from vllm.v1.request import Request

BLOCK, WINDOW, DRAFTS, ALIGN = 8, 4, 7, 64


def layer(steps=2, drafts=DRAFTS, ckpt=0):
    spec = AttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    return SconvStateManager.layer_config(
        spec, sliding_window=WINDOW, num_speculative_tokens=drafts, max_in_flight_steps=steps,
        num_lookahead_tokens=drafts, hit_alignment_tokens=ALIGN, checkpoint_tokens=ckpt,
    )


def manager(cfg, caching=False):
    group = KVCacheGroup({"layer": cfg.spec}, cfg.manager_config, BLOCK, False)
    pool = BlockPool(num_gpu_blocks=8192, enable_caching=caching, hash_block_size=BLOCK)
    return SconvStateManager(
        group, block_pool=pool, enable_caching=caching, kv_cache_group_id=0,
        scheduler_block_size=ALIGN, max_admission_blocks_per_request=10**9,
    )


def resident(m, rid):
    return [i for i, b in enumerate(m.req_to_blocks[rid]) if not b.is_null]


def hit_len(m, req, max_length):
    kw = dict(block_hashes=req.block_hashes, max_length=max_length, kv_cache_group_ids=[0],
              block_pool=m.block_pool, block_size=BLOCK, manager_config=m.manager_config, alignment_tokens=ALIGN)
    try:
        return SconvStateManager.find_longest_cache_hit(**kw)[1]
    except TypeError:
        return SconvStateManager.find_longest_cache_hit(drop_eagle_block=False, **kw)[1]


@pytest.mark.parametrize("chunk", [2048, 4096, 8192])
def test_chunk_holds_a_tail_not_the_chunk(chunk):
    cfg = layer()
    m = manager(cfg)
    cap = SconvStateManager.max_admission_blocks(cfg.manager_config, BLOCK, 2 * chunk, 10**6)
    assert cap is not None and cap < 64, cap  # the stock manager needs 2 * chunk / 8 blocks here
    done = 0
    for end in (chunk, 2 * chunk, 2 * chunk + 1000):
        m.remove_skipped_blocks("r", done)
        m.allocate_new_blocks("r", end + DRAFTS, end)
        done = end
        assert len(resident(m, "r")) <= cap, (len(resident(m, "r")), cap)
    # the next step reads the taps of the last WINDOW - 1 tokens
    assert (done - WINDOW + 1) // BLOCK in resident(m, "r")
    assert (done - 1) // BLOCK in resident(m, "r")


@pytest.mark.parametrize("steps", [1, 2])
def test_taps_survive_rejected_drafts(steps):
    m = manager(layer(steps))
    for target in range(5000, 5000 + 2 * BLOCK):
        rid = str(target)
        m.allocate_new_blocks(rid, target + DRAFTS, target)
        accepted = target - DRAFTS * steps  # every draft in flight rejected
        blocks = m.req_to_blocks[rid]
        assert blocks[0].is_null
        assert not any(b.is_null for b in blocks[(accepted - WINDOW + 1) // BLOCK:])


@pytest.mark.parametrize("prompt_len,chunk", [(6000, 2048), (6016, 2048), (4097, 2048), (10000, 4096)])
def test_follow_up_hits_at_last_aligned_position(prompt_len, chunk):
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK, sha256)
    m = manager(layer(), caching=True)

    def request(rid, ids):
        return Request(request_id=rid, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1),
                       pooling_params=None, block_hasher=hasher)

    first = request("a", list(range(prompt_len)))
    done = 0
    while done < prompt_len:
        end = min(done + chunk, prompt_len)
        m.remove_skipped_blocks("a", done)
        m.allocate_new_blocks("a", end, end)
        m.cache_blocks(first, end, replay_boundaries=[])
        done = end
    assert hit_len(m, request("b", list(range(prompt_len + 500))), prompt_len + 499) == prompt_len // ALIGN * ALIGN
    # a prompt that branches off in the middle of the first chunk finds no conv state there
    assert hit_len(m, request("c", list(range(chunk // 2)) + [10**6] * 600), chunk // 2 + 599) == 0
    # one that branches off just after the first chunk's end reuses that chunk
    assert hit_len(m, request("d", list(range(chunk + 5)) + [10**6] * 600), chunk + 604) == chunk


CKPT = 512


@pytest.mark.parametrize("chunk", [2048, 4096, 8192])
def test_checkpoints_inside_a_chunk(chunk):
    """Each chunk keeps the block below every 512-token boundary until it is processed; the admission
    count covers them; a prompt that branches off mid-chunk resumes at the last checkpoint."""
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK, sha256)
    cfg = layer(ckpt=CKPT)
    m = manager(cfg, caching=True)
    cap = SconvStateManager.max_admission_blocks(cfg.manager_config, BLOCK, 2 * chunk, 10**6)
    plain = SconvStateManager.max_admission_blocks(layer().manager_config, BLOCK, 2 * chunk, 10**6)
    assert plain < cap <= plain + 2 * chunk // CKPT + 2

    def request(rid, ids):
        return Request(request_id=rid, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1),
                       pooling_params=None, block_hasher=hasher)

    prompt_len = 3 * chunk + 700
    first = request("a", list(range(prompt_len)))
    done, prev = 0, 0
    while done < prompt_len:
        end = min(done + chunk, prompt_len)
        # async scheduling: the chunk before last is still in flight when this one is allocated
        m.remove_skipped_blocks("a", prev)
        free = m.block_pool.get_num_free_blocks()
        need = m.get_num_blocks_to_allocate("a", end, [], prev, 0, end)
        m.allocate_new_blocks("a", end, end)
        assert free - m.block_pool.get_num_free_blocks() == need
        m.cache_blocks(first, end, replay_boundaries=[])
        held = resident(m, "a")
        assert len(held) <= cap, (len(held), cap)
        for boundary in range(done + CKPT, end - 100, CKPT):
            assert boundary // BLOCK - 1 in held
        prev, done = done, end
    # everything below the processed prefix's window is freed, checkpoints included
    m.remove_skipped_blocks("a", prompt_len)
    assert min(resident(m, "a")) >= (prompt_len - WINDOW + 1) // BLOCK
    for branch_at in (chunk // 2 + 37, chunk + 300, 2 * chunk + 1000, 3 * chunk + 650):
        ids = list(range(branch_at)) + [10**6] * 600
        assert hit_len(m, request(f"b{branch_at}", ids), branch_at + 599) >= branch_at // CKPT * CKPT, branch_at
        assert hit_len(m, request(f"c{branch_at}", ids), branch_at + 599) <= branch_at // ALIGN * ALIGN
