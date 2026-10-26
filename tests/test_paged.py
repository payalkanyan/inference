"""Stage 3: the paged KV cache must not change the output.

The gate is ``test_paged_matches_naive``. The tests around it attack the one new
idea -- that a sequence's KV no longer lives in one contiguous run -- from three
directions:

* the slot arithmetic in isolation,
* storage equivalence: the same tensors read back through a *scattered* block
  table as through a contiguous buffer,
* invariance: identical tokens whatever the block size, and whatever order the
  pool happens to hand out blocks in.

The last one matters most. Block tables are monotonic [0, 1, 2, ...] on a fresh
pool, which is exactly the arrangement in which a bug in the slot indirection is
invisible. Several tests here deliberately fragment the pool first.
"""

from __future__ import annotations

import pytest
import torch

from vox.blocks import BlockPoolExhausted
from vox.cache import ContiguousKVCache, PagedKVCache
from vox.config import Qwen2Config
from vox.generate import generate_cached, generate_naive, generate_paged
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.sequence import Sequence
from vox.weights import load_model

MAX_NEW_TOKENS = 16

PROMPTS = [
    "The capital of France is",
    "Count from one to ten:",
]


def tiny_config(**overrides) -> Qwen2Config:
    base = dict(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )
    base.update(overrides)
    return Qwen2Config(**base)


def drain_then_free(cache: PagedKVCache, keep_free: list[int]) -> None:
    """Leave the pool holding exactly ``keep_free``, in that order.

    Allocates every block, then releases the chosen ones out of order, so the
    next sequence gets a scattered, non-monotonic block table instead of the
    tidy [0, 1, 2, ...] a fresh pool would give it.
    """
    victims = [Sequence(token_ids=[0]) for _ in range(cache.num_blocks)]
    for victim in victims:
        cache.allocate_for(victim, 1)
    assert cache.pool.num_free_blocks == 0
    for block_id in keep_free:
        cache.free(victims[block_id])
    assert list(cache.pool.free_block_ids) == keep_free


# --------------------------------------------------------------------------
# Slot arithmetic. slot = block_id * block_size + offset_within_block.
# --------------------------------------------------------------------------

def test_slots_follow_the_block_table():
    cache = PagedKVCache(tiny_config(), num_blocks=8, block_size=4)
    # Logical positions 0..5 over a deliberately out-of-order table.
    slots = cache.slots([6, 2], 0, 6, device="cpu")
    assert slots.tolist() == [24, 25, 26, 27, 8, 9]


def test_slots_for_a_mid_sequence_range():
    cache = PagedKVCache(tiny_config(), num_blocks=8, block_size=4)
    slots = cache.slots([6, 2], 3, 5, device="cpu")
    assert slots.tolist() == [27, 8]


def test_block_size_one_gives_one_slot_per_block():
    cache = PagedKVCache(tiny_config(), num_blocks=8, block_size=1)
    assert cache.slots([5, 0, 3], 0, 3, device="cpu").tolist() == [5, 0, 3]


def test_slots_past_the_allocated_blocks_are_refused():
    cache = PagedKVCache(tiny_config(), num_blocks=8, block_size=4)
    with pytest.raises(ValueError, match="past the 1 block"):
        cache.slots([3], 0, 5, device="cpu")


# --------------------------------------------------------------------------
# Storage equivalence: paging must be invisible to whoever reads it back.
# --------------------------------------------------------------------------

def test_scattered_storage_reads_back_identically_to_contiguous():
    """The core claim of the paged cache, isolated from the model.

    The same keys and values, written through a non-monotonic block table and
    read back, must be bit-identical to the contiguous cache's answer.
    """
    cfg = tiny_config()
    n = 7
    k = torch.randn(1, cfg.num_key_value_heads, n, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, n, cfg.head_dim)

    contiguous = ContiguousKVCache(cfg, max_seq_len=16)
    flat = contiguous.build_state(n, device="cpu", dtype=torch.float32)
    contiguous.write(0, k, v, flat)

    paged = PagedKVCache(cfg, num_blocks=8, block_size=2)
    scattered = Sequence(token_ids=[0] * n)
    scattered.block_table = [5, 1, 6, 0]          # covers 8 slots, out of order
    state = paged.build_state(scattered, n, device="cpu", dtype=torch.float32)
    paged.write(0, k, v, state)

    got_k, got_v = paged.read(0, state)
    want_k, want_v = contiguous.read(0, flat)
    assert torch.equal(got_k, want_k)
    assert torch.equal(got_v, want_v)
    assert torch.equal(got_k, k)


def test_writing_one_sequence_does_not_disturb_another_in_the_same_pool():
    cfg = tiny_config()
    paged = PagedKVCache(cfg, num_blocks=8, block_size=4)

    first = Sequence(token_ids=[0] * 4)
    paged.allocate_for(first, 4)
    ones = torch.ones(1, cfg.num_key_value_heads, 4, cfg.head_dim)
    state_a = paged.build_state(first, 4, device="cpu", dtype=torch.float32)
    paged.write(0, ones, ones, state_a)

    second = Sequence(token_ids=[0] * 4)
    paged.allocate_for(second, 4)
    twos = torch.full((1, cfg.num_key_value_heads, 4, cfg.head_dim), 2.0)
    state_b = paged.build_state(second, 4, device="cpu", dtype=torch.float32)
    paged.write(0, twos, twos, state_b)

    assert first.block_table != second.block_table
    assert paged.read(0, state_a)[0].unique().tolist() == [1.0]
    assert paged.read(0, state_b)[0].unique().tolist() == [2.0]


def test_paged_writes_are_per_layer_isolated():
    cfg = tiny_config()
    paged = PagedKVCache(cfg, num_blocks=8, block_size=4)
    s = Sequence(token_ids=[0] * 4)
    paged.allocate_for(s, 4)
    state = paged.build_state(s, 4, device="cpu", dtype=torch.float32)

    for layer in range(cfg.num_hidden_layers):
        filler = torch.full((1, cfg.num_key_value_heads, 4, cfg.head_dim), float(layer))
        paged.write(layer, filler, filler, state)
    for layer in range(cfg.num_hidden_layers):
        assert paged.read(layer, state)[0].unique().tolist() == [float(layer)]


def test_appending_across_a_block_boundary_keeps_earlier_tokens():
    cfg = tiny_config()
    paged = PagedKVCache(cfg, num_blocks=8, block_size=2)
    s = Sequence(token_ids=[0] * 3)

    paged.allocate_for(s, 2)
    first = torch.ones(1, cfg.num_key_value_heads, 2, cfg.head_dim)
    paged.write(0, first, first, paged.build_state(s, 2, device="cpu", dtype=torch.float32))
    s.num_computed_tokens = 2

    paged.allocate_for(s, 1)                       # crosses into a second block
    assert len(s.block_table) == 2
    second = torch.full((1, cfg.num_key_value_heads, 1, cfg.head_dim), 9.0)
    state = paged.build_state(s, 1, device="cpu", dtype=torch.float32)
    paged.write(0, second, second, state)

    got_k, _ = paged.read(0, state)
    assert got_k.shape[2] == 3
    assert torch.equal(got_k[:, :, :2], first)
    assert torch.equal(got_k[:, :, 2:], second)


def test_masking_decisions_match_the_contiguous_cache():
    """Both caches route through mask_for, so their decisions cannot diverge."""
    cfg = tiny_config()
    contiguous = ContiguousKVCache(cfg, max_seq_len=32)
    paged = PagedKVCache(cfg, num_blocks=8, block_size=4)
    s = Sequence(token_ids=[0] * 32)
    paged.allocate_for(s, 12)

    for computed, q_len in [(0, 5), (7, 1), (4, 3)]:
        contiguous.length = computed
        s.num_computed_tokens = computed
        a = contiguous.build_state(q_len, device="cpu", dtype=torch.float32)
        b = paged.build_state(s, q_len, device="cpu", dtype=torch.float32)
        assert (a.start, a.end, a.is_causal) == (b.start, b.end, b.is_causal)
        assert (a.mask is None) == (b.mask is None)
        if a.mask is not None:
            assert torch.equal(a.mask, b.mask)


# --------------------------------------------------------------------------
# End to end against the real model. These need the weights.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_dir()


@pytest.fixture(scope="module")
def tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


@pytest.fixture(scope="module")
def vox_model(model_dir):
    return load_model(model_dir, dtype=torch.float32, device="cpu")


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("block_size", [1, 4, 16])
@torch.inference_mode()
def test_paged_prefill_logits_are_bit_identical_to_uncached(vox_model, tokenizer, block_size):
    """Gathering through a block table must be exactly, not approximately, equal.

    Prefill from an empty cache takes the is_causal path and the keys only
    round-trip through a pool, so on float32 CPU there is no reason for a single
    bit to move. Asserting equality rather than a tolerance means a real
    regression cannot hide inside a threshold. (This is a CPU/float32 claim; a
    CUDA kernel may legitimately reassociate.)
    """
    tokens = _encode(tokenizer, PROMPTS[0])
    ids = torch.tensor([tokens], dtype=torch.long)

    cache = PagedKVCache(
        vox_model.cfg, num_blocks=64, block_size=block_size, dtype=vox_model.dtype
    )
    s = Sequence(token_ids=list(tokens))
    cache.allocate_for(s, len(tokens))
    state = cache.build_state(s, len(tokens), device="cpu", dtype=vox_model.dtype)

    paged = vox_model(ids, state=state)
    uncached = vox_model(ids)

    deviation = (paged - uncached).abs().max().item()
    assert torch.equal(paged, uncached), f"paged prefill drifted by {deviation:.3e}"


def test_paged_matches_naive(vox_model, tokenizer):
    """THE stage-3 gate: paging must not change a single token."""
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    for text in PROMPTS:
        tokens = _encode(tokenizer, text)
        naive = generate_naive(vox_model, tokens, params).token_ids
        paged = generate_paged(vox_model, tokens, params).token_ids
        assert paged == naive, (
            f"divergence for {text!r}\n"
            f"  naive: {naive}\n  ->     {tokenizer.decode(naive)!r}\n"
            f"  paged: {paged}\n  ->     {tokenizer.decode(paged)!r}"
        )


def test_paged_matches_contiguous(vox_model, tokenizer):
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    tokens = _encode(tokenizer, PROMPTS[0])
    assert (
        generate_paged(vox_model, tokens, params).token_ids
        == generate_cached(vox_model, tokens, params).token_ids
    )


@pytest.mark.parametrize("block_size", [1, 3, 8, 16, 64])
def test_block_size_does_not_change_the_output(vox_model, tokenizer, block_size):
    """Block size is a memory-management knob and must not be observable.

    1 is maximal fragmentation (a block per token); 3 is not a power of two,
    which is where modular arithmetic goes wrong; 64 holds the whole sequence.
    """
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    tokens = _encode(tokenizer, PROMPTS[0])
    expected = generate_naive(vox_model, tokens, params).token_ids
    got = generate_paged(vox_model, tokens, params, block_size=block_size).token_ids
    assert got == expected


def test_a_fragmented_pool_gives_the_same_output(vox_model, tokenizer):
    """The test a fresh pool cannot do.

    On a fresh pool a block table is [0, 1, 2, ...] and the slot indirection is
    indistinguishable from plain offsets. Here the pool is drained and selectively
    released so the sequence gets a scattered, descending-ish table instead.
    """
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    tokens = _encode(tokenizer, PROMPTS[0])
    expected = generate_naive(vox_model, tokens, params).token_ids

    cache = PagedKVCache(vox_model.cfg, num_blocks=16, block_size=4, dtype=vox_model.dtype)
    drain_then_free(cache, keep_free=[9, 2, 14, 5, 0, 11, 7, 3])

    got = generate_paged(vox_model, tokens, params, cache=cache).token_ids
    assert got == expected


def test_blocks_are_released_when_generation_finishes(vox_model, tokenizer):
    cache = PagedKVCache(vox_model.cfg, num_blocks=16, block_size=4, dtype=vox_model.dtype)
    before = cache.pool.num_free_blocks

    generate_paged(
        vox_model,
        _encode(tokenizer, PROMPTS[0]),
        SamplingParams(temperature=0.0, max_new_tokens=8),
        cache=cache,
    )

    assert cache.pool.num_free_blocks == before
    assert all(b.is_free for b in cache.pool.blocks)


def test_blocks_are_released_when_generation_raises(vox_model, tokenizer):
    """The finally-block path. Stage 7's cancellation reuses exactly this."""
    cache = PagedKVCache(vox_model.cfg, num_blocks=16, block_size=4, dtype=vox_model.dtype)
    before = cache.pool.num_free_blocks

    class Boom(RuntimeError):
        pass

    def explode(*args, **kwargs):
        raise Boom

    params = SamplingParams(temperature=0.0, max_new_tokens=8)
    import vox.generate as generate_module

    original = generate_module.sample
    generate_module.sample = explode
    try:
        with pytest.raises(Boom):
            generate_paged(vox_model, _encode(tokenizer, PROMPTS[0]), params, cache=cache)
    finally:
        generate_module.sample = original

    assert cache.pool.num_free_blocks == before


def test_running_out_of_blocks_raises_the_catchable_error(vox_model, tokenizer):
    """Stage 5 turns this into preemption; for now it must at least be specific."""
    cache = PagedKVCache(vox_model.cfg, num_blocks=2, block_size=4, dtype=vox_model.dtype)
    with pytest.raises(BlockPoolExhausted):
        generate_paged(
            vox_model,
            _encode(tokenizer, PROMPTS[0]),
            SamplingParams(temperature=0.0, max_new_tokens=32),
            cache=cache,
        )
    assert cache.pool.num_free_blocks == 2   # and still cleaned up after


def test_pool_is_reusable_across_requests(vox_model, tokenizer):
    cache = PagedKVCache(vox_model.cfg, num_blocks=16, block_size=4, dtype=vox_model.dtype)
    params = SamplingParams(temperature=0.0, max_new_tokens=8)

    first = generate_paged(vox_model, _encode(tokenizer, PROMPTS[0]), params, cache=cache)
    second = generate_paged(vox_model, _encode(tokenizer, PROMPTS[1]), params, cache=cache)
    repeat = generate_paged(vox_model, _encode(tokenizer, PROMPTS[0]), params, cache=cache)

    assert first.token_ids == repeat.token_ids    # no leakage from the one between
    assert first.token_ids != second.token_ids
