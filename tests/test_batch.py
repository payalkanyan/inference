"""Stage 4: flattening several sequences into one forward pass.

No weights, no model. These pin the two things flattening has to get right --
where each token's KV goes, and which keys each query is allowed to see -- in a
form where a failure points straight at the arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from vox.batch import build_batch, build_ragged_mask
from vox.cache import PagedKVCache
from vox.config import Qwen2Config
from vox.sequence import Sequence


def tiny_config(**overrides) -> Qwen2Config:
    base = dict(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )
    base.update(overrides)
    return Qwen2Config(**base)


def pool(num_blocks: int = 16, block_size: int = 4) -> PagedKVCache:
    return PagedKVCache(tiny_config(), num_blocks=num_blocks, block_size=block_size)


def prefilled(cache: PagedKVCache, token_ids: list[int], computed: int) -> Sequence:
    """A sequence with ``computed`` tokens already in the cache."""
    s = Sequence(token_ids=list(token_ids))
    cache.allocate_for(s, len(token_ids))
    s.num_computed_tokens = computed
    return s


# -- the mask -------------------------------------------------------------

def test_mask_is_causal_within_a_single_sequence():
    ids = torch.tensor([0, 0, 0])
    pos = torch.tensor([0, 1, 2])
    mask = build_ragged_mask(ids, pos, ids, pos, dtype=torch.float32)
    assert torch.equal(mask, torch.full((3, 3), float("-inf")).triu(1))


def test_mask_blocks_across_sequences_entirely():
    """Two 2-token sequences. Neither may see a single key of the other."""
    q_seq = torch.tensor([0, 0, 1, 1])
    q_pos = torch.tensor([0, 1, 0, 1])
    mask = build_ragged_mask(q_seq, q_pos, q_seq, q_pos, dtype=torch.float32)

    visible = (mask == 0)
    assert visible[:2, 2:].sum() == 0      # sequence 0 sees nothing of sequence 1
    assert visible[2:, :2].sum() == 0      # and vice versa
    assert visible[:2, :2].tolist() == [[True, False], [True, True]]
    assert visible[2:, 2:].tolist() == [[True, False], [True, True]]


def test_mask_lets_a_decode_see_its_whole_prefix_but_no_further():
    """One decode query at position 3 over a 4-key prefix, batched with another."""
    q_seq = torch.tensor([0, 1])
    q_pos = torch.tensor([3, 2])
    k_seq = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    k_pos = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    mask = build_ragged_mask(q_seq, q_pos, k_seq, k_pos, dtype=torch.float32)

    visible = (mask == 0)
    assert visible[0].tolist() == [True, True, True, True, False, False, False]
    assert visible[1].tolist() == [False, False, False, False, True, True, True]


def test_mask_uses_sequence_local_positions_not_flattened_ones():
    """A sequence whose prefix starts mid-way must not become more permissive.

    Sequence 1's query is at local position 0, so it sees exactly one key even
    though it sits at flattened index 2.
    """
    q_seq = torch.tensor([0, 0, 1])
    q_pos = torch.tensor([0, 1, 0])
    k_seq = torch.tensor([0, 0, 1])
    k_pos = torch.tensor([0, 1, 0])
    visible = build_ragged_mask(q_seq, q_pos, k_seq, k_pos, dtype=torch.float32) == 0
    assert visible[2].tolist() == [False, False, True]


# -- flattening -----------------------------------------------------------

def test_prefill_batch_flattens_tokens_and_positions():
    cache = pool()
    a = Sequence(token_ids=[10, 11, 12])
    b = Sequence(token_ids=[20, 21])
    cache.allocate_for(a, 3)
    cache.allocate_for(b, 2)

    batch = build_batch(cache, [a, b], [3, 2])

    assert batch.input_ids.tolist() == [[10, 11, 12, 20, 21]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 0, 1]]
    assert batch.num_query_tokens == 5
    assert batch.num_key_tokens == 5
    # Last query position of each sequence: index 2 and index 4.
    assert batch.logit_indices.tolist() == [2, 4]


def test_decode_batch_feeds_one_token_each_at_its_own_position():
    cache = pool()
    a = prefilled(cache, [1, 2, 3, 4], computed=3)
    b = prefilled(cache, [5, 6], computed=1)

    batch = build_batch(cache, [a, b], [1, 1])

    assert batch.input_ids.tolist() == [[4, 6]]
    assert batch.position_ids.tolist() == [[3, 1]]
    assert batch.logit_indices.tolist() == [0, 1]
    # Keys span each sequence's whole prefix: 4 for a, 2 for b.
    assert batch.num_key_tokens == 6


def test_mixed_prefill_and_decode_in_one_batch():
    """What continuous batching is for: a new prompt alongside a live decode."""
    cache = pool()
    decoding = prefilled(cache, [1, 2, 3], computed=2)
    arriving = Sequence(token_ids=[7, 8, 9, 10])
    cache.allocate_for(arriving, 4)

    batch = build_batch(cache, [decoding, arriving], [1, 4])

    assert batch.input_ids.tolist() == [[3, 7, 8, 9, 10]]
    assert batch.position_ids.tolist() == [[2, 0, 1, 2, 3]]
    assert batch.logit_indices.tolist() == [0, 4]
    assert batch.num_key_tokens == 3 + 4

    visible = batch.state.mask == 0
    assert visible[0].tolist() == [True, True, True, False, False, False, False]
    assert visible[1].tolist() == [False, False, False, True, False, False, False]


def test_write_slots_cover_only_the_new_tokens():
    cache = pool(block_size=4)
    a = prefilled(cache, [1, 2, 3, 4, 5], computed=4)
    assert a.block_table == [0, 1]

    batch = build_batch(cache, [a], [1])
    # Token 5 is logical position 4 -> block_table[1] == 1, offset 0 -> slot 4.
    assert batch.state.write_slots.tolist() == [4]
    # ...while the read covers all five.
    assert batch.state.read_slots.tolist() == [0, 1, 2, 3, 4]


def test_slots_follow_a_fragmented_block_table():
    """The arrangement a fresh pool never produces."""
    cache = pool(block_size=2)
    a = Sequence(token_ids=[1, 2, 3])
    a.block_table = [5, 1]
    batch = build_batch(cache, [a], [3])
    assert batch.state.write_slots.tolist() == [10, 11, 2]
    assert batch.state.read_slots.tolist() == [10, 11, 2]


def test_batched_slots_concatenate_per_sequence():
    cache = pool(block_size=2)
    a, b = Sequence(token_ids=[1, 2]), Sequence(token_ids=[3, 4])
    a.block_table = [3]
    b.block_table = [0]
    batch = build_batch(cache, [a, b], [2, 2])
    assert batch.state.write_slots.tolist() == [6, 7, 0, 1]


def test_batch_state_carries_a_mask_rather_than_a_causal_flag():
    cache = pool()
    a = Sequence(token_ids=[1, 2])
    cache.allocate_for(a, 2)
    batch = build_batch(cache, [a], [2])
    assert batch.state.is_causal is False
    assert batch.state.mask is not None
    assert batch.state.mask.shape == (2, 2)


# -- refusals -------------------------------------------------------------

def test_empty_batch_is_refused():
    with pytest.raises(ValueError, match="no sequences"):
        build_batch(pool(), [], [])


def test_mismatched_lengths_are_refused():
    cache = pool()
    a = Sequence(token_ids=[1, 2])
    cache.allocate_for(a, 2)
    with pytest.raises(ValueError, match="query lengths"):
        build_batch(cache, [a], [1, 1])


def test_scheduling_more_tokens_than_the_sequence_has_is_refused():
    cache = pool()
    a = Sequence(token_ids=[1, 2])
    cache.allocate_for(a, 2)
    with pytest.raises(ValueError, match="cannot compute up to"):
        build_batch(cache, [a], [5])


def test_zero_length_query_is_refused():
    cache = pool()
    a = Sequence(token_ids=[1, 2])
    cache.allocate_for(a, 2)
    with pytest.raises(ValueError, match="scheduled for 0 tokens"):
        build_batch(cache, [a], [0])


# -- stage 5: chunk continuation ------------------------------------------

def test_mask_for_a_second_prefill_chunk():
    """Queries partway through a prompt: positions 2 and 3 over 4 keys.

    The case no SDPA flag expresses. is_causal would align the implicit mask to
    the top left and let query 0 see only key 0.
    """
    q_seq, q_pos = torch.tensor([0, 0]), torch.tensor([2, 3])
    k_seq, k_pos = torch.tensor([0] * 4), torch.tensor([0, 1, 2, 3])
    visible = build_ragged_mask(q_seq, q_pos, k_seq, k_pos, dtype=torch.float32) == 0
    assert visible.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_a_chunk_reads_the_prefix_and_writes_only_its_own_slots():
    cache = pool(block_size=4)
    s = prefilled(cache, list(range(10)), computed=4)

    batch = build_batch(cache, [s], [3])          # tokens 4, 5, 6

    assert batch.input_ids.tolist() == [[4, 5, 6]]
    assert batch.position_ids.tolist() == [[4, 5, 6]]
    assert batch.state.write_slots.tolist() == [4, 5, 6]
    assert batch.state.read_slots.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert batch.logit_indices.tolist() == [2]


def test_a_chunk_batched_with_a_decode_keeps_both_correct():
    cache = pool(block_size=4)
    chunking = prefilled(cache, list(range(8)), computed=2)
    deciding = prefilled(cache, [90, 91, 92], computed=2)

    batch = build_batch(cache, [chunking, deciding], [3, 1])

    assert batch.input_ids.tolist() == [[2, 3, 4, 92]]
    assert batch.position_ids.tolist() == [[2, 3, 4, 2]]
    visible = batch.state.mask == 0
    # The chunk sees its own 5 keys and none of the decode's 3.
    assert visible[0].tolist() == [True, True, True, False, False, False, False, False]
    assert visible[2].tolist() == [True, True, True, True, True, False, False, False]
    assert visible[3].tolist() == [False, False, False, False, False, True, True, True]
