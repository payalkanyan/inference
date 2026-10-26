"""Stage 2: the contiguous KV cache must not change the output.

The gate is ``test_cached_matches_naive`` -- same tokens as the uncached stage-1
path. Around it are the cheap structural tests that say *why* a failure happened:
whether the buffer is being written in the wrong place, the mask is wrong, or the
positions are wrong.
"""

from __future__ import annotations

import pytest
import torch

from vox.cache import ContiguousKVCache
from vox.config import Qwen2Config
from vox.generate import generate_cached, generate_naive
from vox.hub import resolve_model_dir
from vox.state import AttentionState, build_causal_mask
from vox.sampler import SamplingParams
from vox.weights import load_model

MAX_NEW_TOKENS = 24

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


# --------------------------------------------------------------------------
# The buffer itself. No model, no weights.
# --------------------------------------------------------------------------

def at(start: int, end: int, cache=None) -> AttentionState:
    """Minimal state for driving a cache directly in a unit test."""
    return AttentionState(cache, start, end, None, True)


def test_write_then_read_round_trips():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=8)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    cache.write(1, k, v, at(0, 3))

    got_k, got_v = cache.read(1, at(0, 3))
    assert torch.equal(got_k, k)
    assert torch.equal(got_v, v)


def test_writes_are_per_layer_isolated():
    """Catches every layer scribbling over layer 0, which still decodes fluently."""
    cache = ContiguousKVCache(tiny_config(), max_seq_len=8)
    for layer in range(3):
        filler = torch.full((1, 2, 2, 4), float(layer))
        cache.write(layer, filler, filler, at(0, 2))

    for layer in range(3):
        got_k, _ = cache.read(layer, at(0, 2))
        assert got_k.unique().tolist() == [float(layer)]


def test_appending_at_an_offset_leaves_earlier_tokens_alone():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=8)
    first = torch.ones(1, 2, 2, 4)
    second = torch.full((1, 2, 1, 4), 5.0)
    cache.write(0, first, first, at(0, 2))
    cache.advance(2)
    cache.write(0, second, second, at(cache.length, cache.length + 1))
    cache.advance(1)

    got_k, _ = cache.read(0, at(0, cache.length))
    assert got_k.shape[2] == 3
    assert torch.equal(got_k[:, :, :2], first)
    assert torch.equal(got_k[:, :, 2:], second)


def test_read_is_bounded_by_length_so_stale_bytes_are_invisible():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=8)
    cache.write(0, torch.ones(1, 2, 4, 4), torch.ones(1, 2, 4, 4), at(0, 4))
    cache.advance(4)
    cache.reset()
    assert cache.length == 0
    got_k, _ = cache.read(0, at(0, cache.length))
    assert got_k.shape[2] == 0


def test_write_past_the_end_is_refused():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=4)
    big = torch.ones(1, 2, 5, 4)
    with pytest.raises(ValueError, match="overflow"):
        cache.write(0, big, big, at(0, 5))


def test_advance_past_the_end_is_refused():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=4)
    cache.advance(4)
    with pytest.raises(ValueError, match="overflow"):
        cache.advance(1)


def test_cache_survives_allocation_inside_inference_mode():
    """A buffer allocated under inference_mode must still be writable later.

    Without the explicit `inference_mode(False)` in the constructor it becomes an
    inference tensor, and every write after the enclosing block raises.
    """
    with torch.inference_mode():
        cache = ContiguousKVCache(tiny_config(), max_seq_len=4)
    cache.write(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4), at(0, 1))
    cache.advance(1)
    assert cache.length == 1


def test_rejects_nonpositive_capacity():
    with pytest.raises(ValueError, match="positive"):
        ContiguousKVCache(tiny_config(), max_seq_len=0)


# --------------------------------------------------------------------------
# Masking. This is the part that is easy to get silently wrong.
# --------------------------------------------------------------------------

def test_aligned_mask_is_the_standard_causal_mask():
    got = build_causal_mask(4, 4, device="cpu", dtype=torch.float32)
    expected = torch.full((4, 4), float("-inf")).triu(1)
    assert torch.equal(got, expected)


def test_decode_mask_blocks_nothing():
    """One query at the end of n keys may attend to all of them."""
    got = build_causal_mask(1, 6, device="cpu", dtype=torch.float32)
    assert torch.equal(got, torch.zeros(1, 6))


def test_chunk_mask_offsets_correctly():
    """Two queries at the end of 5 keys: query 0 is at position 3, query 1 at 4."""
    got = build_causal_mask(2, 5, device="cpu", dtype=torch.float32)
    ninf = float("-inf")
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, ninf], [0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.equal(got, expected)


def test_state_for_prefill_uses_is_causal_and_no_mask():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=16)
    state = cache.build_state(5, device="cpu", dtype=torch.float32)
    assert (state.start, state.end) == (0, 5)
    assert state.is_causal and state.mask is None


def test_state_for_decode_uses_neither_flag_nor_mask():
    """is_causal would be actively wrong here: SDPA aligns top-left, so a single
    query over n keys would be restricted to key 0."""
    cache = ContiguousKVCache(tiny_config(), max_seq_len=16)
    cache.advance(7)
    state = cache.build_state(1, device="cpu", dtype=torch.float32)
    assert (state.start, state.end) == (7, 8)
    assert not state.is_causal and state.mask is None


def test_state_for_chunk_continuation_builds_a_mask():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=16)
    cache.advance(4)
    state = cache.build_state(3, device="cpu", dtype=torch.float32)
    assert (state.start, state.end) == (4, 7)
    assert not state.is_causal
    assert state.mask is not None and state.mask.shape == (3, 7)


def test_state_without_cache_matches_stage_one():
    state = AttentionState.uncached(5)
    assert state.cache is None and state.is_causal and state.mask is None
    assert (state.start, state.end) == (0, 5)


def test_state_refuses_to_exceed_capacity():
    cache = ContiguousKVCache(tiny_config(), max_seq_len=8)
    cache.advance(6)
    with pytest.raises(ValueError, match="cache holds"):
        cache.build_state(4, device="cpu", dtype=torch.float32)


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


@torch.inference_mode()
def test_prefill_logits_identical_with_and_without_cache(vox_model, tokenizer):
    """Prefill from an empty cache takes the is_causal path, so this should be
    bit-identical, not merely close -- the keys only round-trip through a buffer."""
    ids = torch.tensor([_encode(tokenizer, PROMPTS[0])], dtype=torch.long)
    cache = ContiguousKVCache(vox_model.cfg, 64, dtype=vox_model.dtype)

    uncached = vox_model(ids)
    cached = vox_model(ids, cache=cache)

    assert torch.equal(uncached, cached)
    assert cache.length == ids.shape[1]


@torch.inference_mode()
def test_decode_logits_track_full_recompute(vox_model, tokenizer):
    """Step through a few decode steps and compare against full recompute.

    Decode reduces over cached keys for a single query instead of inside one big
    SxS matmul, so the accumulation order differs and exact equality is not
    expected here -- only that the gap stays far below what could flip an argmax.
    """
    tokens = _encode(tokenizer, PROMPTS[0])
    cache = ContiguousKVCache(vox_model.cfg, 64, dtype=vox_model.dtype)
    logits = vox_model(torch.tensor([tokens], dtype=torch.long), cache=cache)[:, -1, :]
    next_id = int(logits.argmax())

    worst = 0.0
    for _ in range(6):
        # Append first: both sides must be looking at the same sequence, which is
        # the prompt plus every token decoded so far.
        tokens.append(next_id)
        cached_logits = vox_model(
            torch.tensor([[next_id]], dtype=torch.long), cache=cache
        )[:, -1, :]
        full_logits = vox_model(torch.tensor([tokens], dtype=torch.long))[:, -1, :]

        worst = max(worst, (cached_logits - full_logits).abs().max().item())
        assert int(cached_logits.argmax()) == int(full_logits.argmax())
        next_id = int(cached_logits.argmax())

    assert worst < 1e-3, f"decode drifted from full recompute by {worst:.3e}"


@torch.inference_mode()
def test_chunked_prefill_matches_single_prefill(vox_model, tokenizer):
    """Feed the same prompt in two pieces; the final logits must agree.

    This is the only thing that exercises build_causal_mask through the real
    model. Stage 5 turns it into a scheduling feature, but the attention side has
    to be right before then, so it is pinned here.
    """
    tokens = _encode(tokenizer, PROMPTS[1])
    split = len(tokens) // 2
    ids = torch.tensor([tokens], dtype=torch.long)

    whole = vox_model(ids)[:, -1, :]

    cache = ContiguousKVCache(vox_model.cfg, 64, dtype=vox_model.dtype)
    vox_model(ids[:, :split], cache=cache)
    chunked = vox_model(ids[:, split:], cache=cache)[:, -1, :]

    assert cache.length == len(tokens)
    deviation = (whole - chunked).abs().max().item()
    assert int(whole.argmax()) == int(chunked.argmax())
    assert deviation < 1e-3, f"chunked prefill drifted by {deviation:.3e}"


@torch.inference_mode()
def test_derived_positions_match_explicit_during_decode(vox_model, tokenizer):
    """The cache-derived position offset must equal passing position_ids by hand."""
    tokens = _encode(tokenizer, PROMPTS[0])
    ids = torch.tensor([tokens], dtype=torch.long)

    derived = ContiguousKVCache(vox_model.cfg, 64, dtype=vox_model.dtype)
    explicit = ContiguousKVCache(vox_model.cfg, 64, dtype=vox_model.dtype)
    vox_model(ids, cache=derived)
    vox_model(ids, cache=explicit)

    step = torch.tensor([[tokens[-1]]], dtype=torch.long)
    a = vox_model(step, cache=derived)
    b = vox_model(step, position_ids=torch.tensor([[len(tokens)]]), cache=explicit)
    assert torch.equal(a, b)


def test_cached_matches_naive(vox_model, tokenizer):
    """THE stage-2 gate: caching must not change a single token."""
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    for text in PROMPTS:
        tokens = _encode(tokenizer, text)
        naive = generate_naive(vox_model, tokens, params).token_ids
        cached = generate_cached(vox_model, tokens, params).token_ids
        assert cached == naive, (
            f"divergence for {text!r}\n"
            f"  naive : {naive}\n  ->      {tokenizer.decode(naive)!r}\n"
            f"  cached: {cached}\n  ->      {tokenizer.decode(cached)!r}"
        )


def test_reusing_one_cache_across_requests_is_clean(vox_model, tokenizer):
    """A reused buffer must not leak the previous request's keys into the next."""
    params = SamplingParams(temperature=0.0, max_new_tokens=8)
    shared = ContiguousKVCache(vox_model.cfg, 128, dtype=vox_model.dtype)

    first = generate_cached(vox_model, _encode(tokenizer, PROMPTS[0]), params, cache=shared)
    second = generate_cached(vox_model, _encode(tokenizer, PROMPTS[1]), params, cache=shared)
    fresh = generate_cached(vox_model, _encode(tokenizer, PROMPTS[1]), params)

    assert second.token_ids == fresh.token_ids
    assert first.token_ids != second.token_ids  # different prompts, different output


def test_cached_generation_respects_stop_tokens(vox_model, tokenizer):
    tokens = _encode(tokenizer, PROMPTS[0])
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    natural = generate_cached(vox_model, tokens, params).token_ids

    stop_on = natural[2]
    stopped = generate_cached(vox_model, tokens, params, stop_token_ids=[stop_on])
    assert stopped.finish_reason == "stop"
    assert stopped.token_ids == natural[:3]
