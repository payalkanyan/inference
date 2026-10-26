"""Stage 6: prefix caching.

The gate is ``test_prefix_caching_does_not_change_the_output``. Around it sit the
tests for the three ways this feature can be wrong, all of which produce fluent
and completely incorrect text rather than a crash:

* sharing a block whose tokens match but whose *prefix* does not (chaining),
* accepting a hash collision (verification),
* leaving an evicted block in the registry (deregistration).
"""

from __future__ import annotations

import pytest
import torch

from vox.cache import PagedKVCache
from vox.config import Qwen2Config
from vox.engine import Engine
from vox.generate import generate_naive
from vox.hub import resolve_model_dir
from vox.prefix import hash_block
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.sequence import Sequence
from vox.weights import load_model

MAX_NEW_TOKENS = 10


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )


def pool(num_blocks=32, block_size=4, **kwargs) -> PagedKVCache:
    return PagedKVCache(tiny_config(), num_blocks=num_blocks, block_size=block_size, **kwargs)


def computed(cache: PagedKVCache, token_ids: list[int]) -> Sequence:
    """A sequence whose whole token list has been prefilled and published."""
    s = Sequence(token_ids=list(token_ids))
    cache.allocate_for(s, len(token_ids))
    s.num_computed_tokens = len(token_ids)
    cache.register_computed_blocks(s)
    return s


# -- chaining -------------------------------------------------------------

def test_the_same_tokens_after_a_different_parent_hash_differently():
    """The property that makes block identity mean anything.

    Identical content at a different point in a prefix must not be confused,
    because its keys and values are a function of everything before it.
    """
    tokens = (1, 2, 3, 4)
    assert hash_block(None, tokens) != hash_block(999, tokens)
    assert hash_block(111, tokens) != hash_block(222, tokens)


def test_the_same_prefix_hashes_the_same_way_every_time():
    chain_a, chain_b = None, None
    for block in [(1, 2), (3, 4), (5, 6)]:
        chain_a = hash_block(chain_a, block)
        chain_b = hash_block(chain_b, block)
    assert chain_a == chain_b


def test_a_diverging_prefix_diverges_for_every_block_after_it():
    shared = hash_block(None, (1, 2))
    a = hash_block(shared, (3, 4))
    b = hash_block(shared, (9, 9))
    # ...and the divergence propagates, so a later identical block still differs.
    assert hash_block(a, (7, 7)) != hash_block(b, (7, 7))


# -- the registry ---------------------------------------------------------

def test_a_published_block_can_be_found_again():
    cache = pool(block_size=4)
    first = computed(cache, [1, 2, 3, 4, 5, 6, 7, 8])
    assert cache.pool.num_cached_blocks == 2

    chain = hash_block(None, (1, 2, 3, 4))
    assert cache.pool.find_cached(chain, None, (1, 2, 3, 4)) == first.block_table[0]


def test_a_hash_hit_with_the_wrong_tokens_is_a_miss():
    """Collision safety. The hash indexes; equality decides.

    A 64-bit collision is unlikely, but its consequence here is the worst in the
    project: one conversation's KV served to another, fluently and wrongly. So a
    hit whose stored tokens disagree has to be refused.
    """
    cache = pool(block_size=4)
    computed(cache, [1, 2, 3, 4, 5])
    chain = hash_block(None, (1, 2, 3, 4))
    assert cache.pool.find_cached(chain, None, (9, 9, 9, 9)) is None


def test_a_hash_hit_with_the_wrong_parent_is_a_miss():
    cache = pool(block_size=4)
    computed(cache, [1, 2, 3, 4, 5])
    chain = hash_block(None, (1, 2, 3, 4))
    assert cache.pool.find_cached(chain, 12345, (1, 2, 3, 4)) is None


def test_a_forced_collision_is_rejected_rather_than_served():
    """What the verification is actually for, with the collision made real."""
    cache = pool(block_size=4)
    victim = computed(cache, [1, 2, 3, 4, 5])
    stored_hash = cache.pool.blocks[victim.block_table[0]].block_hash

    # Pretend an unrelated prefix hashed to the same value.
    assert cache.pool.find_cached(stored_hash, None, (7, 7, 7, 7)) is None
    # The legitimate owner still finds it.
    assert cache.pool.find_cached(stored_hash, None, (1, 2, 3, 4)) is not None


def test_partial_blocks_are_never_published():
    """A block that is not full can still gain tokens, so it is not final."""
    cache = pool(block_size=4)
    computed(cache, [1, 2, 3, 4, 5, 6])      # one full block plus two tokens
    assert cache.pool.num_cached_blocks == 1


def test_freeing_keeps_the_block_findable():
    """Freeing drops a reference; it does not erase. This is what lets the next
    turn of a conversation pick up a prefix the previous turn released."""
    cache = pool(block_size=4)
    s = computed(cache, [1, 2, 3, 4, 5])
    block_id = s.block_table[0]
    cache.free(s)

    assert cache.pool.blocks[block_id].is_free
    assert cache.pool.blocks[block_id].is_cached
    chain = hash_block(None, (1, 2, 3, 4))
    assert cache.pool.find_cached(chain, None, (1, 2, 3, 4)) == block_id


def test_eviction_deregisters_the_block():
    """The silent-corruption guard.

    A block handed out for new data must leave the registry, or a later lookup
    returns a block now holding somebody else's KV.
    """
    cache = pool(num_blocks=2, block_size=4)
    s = computed(cache, [1, 2, 3, 4, 5, 6, 7, 8])
    chain = hash_block(None, (1, 2, 3, 4))
    first_block = s.block_table[0]
    cache.free(s)
    assert cache.pool.num_cached_blocks == 2

    # Drain the pool; the oldest free block is the one we just published.
    other = Sequence(token_ids=[9] * 8)
    cache.allocate_for(other, 8)

    assert first_block in other.block_table
    assert cache.pool.find_cached(chain, None, (1, 2, 3, 4)) is None
    assert cache.pool.num_cached_blocks == 0


def test_claiming_a_live_block_is_plain_sharing():
    cache = pool(block_size=4)
    s = computed(cache, [1, 2, 3, 4, 5])
    block_id = s.block_table[0]
    assert cache.pool.blocks[block_id].ref_count == 1

    cache.pool.claim_cached(block_id)
    assert cache.pool.blocks[block_id].ref_count == 2


def test_claiming_a_free_block_takes_it_off_the_free_list():
    cache = pool(block_size=4)
    s = computed(cache, [1, 2, 3, 4, 5])
    block_id = s.block_table[0]
    cache.free(s)
    free_before = cache.pool.num_free_blocks

    cache.pool.claim_cached(block_id)
    assert cache.pool.num_free_blocks == free_before - 1
    assert cache.pool.blocks[block_id].ref_count == 1
    assert block_id not in cache.pool.free_block_ids


def test_registering_a_free_block_is_refused():
    cache = pool()
    with pytest.raises(RuntimeError, match="cannot register free block"):
        cache.pool.register(0, 123, None, (1, 2))


# -- attaching a prefix ---------------------------------------------------

def test_a_second_request_with_the_same_prefix_shares_the_blocks():
    """The whole point: separate requests, same system prompt, same memory."""
    cache = pool(block_size=4)
    first = computed(cache, list(range(12)))

    second = Sequence(token_ids=list(range(12)) + [99])
    reused = cache.attach_cached_prefix(second)

    assert reused == 12
    assert second.block_table == first.block_table
    assert second.num_computed_tokens == 12
    assert all(cache.pool.blocks[b].ref_count == 2 for b in first.block_table)


def test_sharing_stops_at_the_first_divergence():
    cache = pool(block_size=4)
    computed(cache, [1, 2, 3, 4, 5, 6, 7, 8])

    diverging = Sequence(token_ids=[1, 2, 3, 4, 50, 60, 70, 80, 99])
    reused = cache.attach_cached_prefix(diverging)

    assert reused == 4                       # the first block only
    assert len(diverging.block_table) == 1


def test_a_different_prefix_shares_nothing():
    cache = pool(block_size=4)
    computed(cache, [1, 2, 3, 4, 5])
    other = Sequence(token_ids=[9, 9, 9, 9, 9])
    assert cache.attach_cached_prefix(other) == 0
    assert other.block_table == []


def test_one_token_is_always_left_to_compute():
    """A fully cached prompt would leave the forward pass nothing to do, and
    there would be no logits to sample the next token from."""
    cache = pool(block_size=4)
    computed(cache, list(range(8)))

    identical = Sequence(token_ids=list(range(8)))
    reused = cache.attach_cached_prefix(identical)

    assert reused == 4                       # not 8
    assert identical.num_uncomputed_tokens >= 1


def test_only_whole_blocks_are_reused():
    cache = pool(block_size=4)
    computed(cache, list(range(10)))

    repeat = Sequence(token_ids=list(range(10)) + [99])
    # 11 tokens, 10 reusable -> two whole blocks, the 2-token tail is recomputed.
    assert cache.attach_cached_prefix(repeat) == 8


def test_attaching_is_a_no_op_when_disabled():
    cache = pool(block_size=4, enable_prefix_caching=False)
    computed(cache, list(range(12)))
    second = Sequence(token_ids=list(range(12)) + [99])
    assert cache.attach_cached_prefix(second) == 0
    assert cache.pool.num_cached_blocks == 0


def test_attaching_is_skipped_for_a_sequence_already_underway():
    cache = pool(block_size=4)
    computed(cache, list(range(12)))
    partial = Sequence(token_ids=list(range(12)) + [99])
    partial.num_computed_tokens = 4
    assert cache.attach_cached_prefix(partial) == 0


def test_detaching_gives_the_blocks_back():
    cache = pool(block_size=4)
    first = computed(cache, list(range(12)))
    second = Sequence(token_ids=list(range(12)) + [99])
    cache.attach_cached_prefix(second)

    cache.detach_prefix(second)
    assert second.block_table == []
    assert second.num_computed_tokens == 0
    assert second.num_cached_tokens == 0
    assert all(cache.pool.blocks[b].ref_count == 1 for b in first.block_table)


def test_a_shared_block_survives_one_owner_finishing():
    """Stage 7's cancellation guarantee, now with sharing that is real."""
    cache = pool(block_size=4)
    first = computed(cache, list(range(12)))
    second = Sequence(token_ids=list(range(12)) + [99])
    cache.attach_cached_prefix(second)
    shared = list(first.block_table)

    cache.free(first)
    assert all(not cache.pool.blocks[b].is_free for b in shared)
    assert second.block_table == shared


def test_hit_ratio_is_reported():
    cache = pool(block_size=4)
    computed(cache, list(range(16)))
    second = Sequence(token_ids=list(range(16)) + [99])
    cache.attach_cached_prefix(second)
    assert second.num_cached_tokens == 16
    assert second.prefix_hit_ratio == pytest.approx(16 / 17)


# -- end to end -----------------------------------------------------------

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


#: A long, mostly-unchanging prefix followed by a short new user segment, which
#: is the shape this whole project is for. Long enough to span several blocks, so
#: that reuse is limited by what actually differs rather than by block geometry.
SYSTEM = (
    "You are a concise voice assistant for a scheduling service. Answer in one "
    "short sentence and never more than twenty words. Speak plainly and do not "
    "use lists. You have access to a calendar tool that can read and create "
    "events, a weather tool that reports current conditions for a city, and a "
    "contacts tool that looks up phone numbers by name. Never invent facts, and "
    "if a tool is unavailable say so rather than guessing. Previous turns of this "
    "conversation are included above for context. User question: "
)
TURNS = ["What is the capital of France?", "And of Italy?", "Name a colour."]


@pytest.fixture(scope="module")
def turns(tokenizer):
    """Three turns that all re-send the same long system prompt: the voice case."""
    return [tokenizer(SYSTEM + t, add_special_tokens=False)["input_ids"] for t in TURNS]


def greedy(n: int = MAX_NEW_TOKENS) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_new_tokens=n)


def run_turns(vox_model, turns, *, enable_prefix_caching: bool, block_size=16):
    cache = PagedKVCache(
        vox_model.cfg,
        num_blocks=512,
        block_size=block_size,
        dtype=vox_model.dtype,
        enable_prefix_caching=enable_prefix_caching,
    )
    engine = Engine(vox_model, cache, SchedulerConfig())
    outputs = {}
    # One turn at a time, as a conversation actually arrives.
    for index, ids in enumerate(turns):
        engine.add_request(f"turn{index}", ids, greedy())
        outputs.update(engine.run_until_done(max_steps=500))
    return engine, outputs


def test_prefix_caching_does_not_change_the_output(vox_model, turns):
    """THE stage-6 gate: reusing a prefix must produce identical tokens."""
    _, without = run_turns(vox_model, turns, enable_prefix_caching=False)
    engine, with_cache = run_turns(vox_model, turns, enable_prefix_caching=True)

    assert engine.num_cached_tokens_reused > 0, "nothing was actually reused"
    for name in without:
        assert with_cache[name].token_ids == without[name].token_ids, (
            f"{name} diverged with prefix caching on\n"
            f"  off: {without[name].token_ids}\n"
            f"  on : {with_cache[name].token_ids}"
        )


def test_prefix_caching_matches_the_naive_reference(vox_model, turns):
    engine, outputs = run_turns(vox_model, turns, enable_prefix_caching=True)
    assert engine.num_cached_tokens_reused > 0
    for index, ids in enumerate(turns):
        expected = generate_naive(vox_model, ids, greedy()).token_ids
        assert outputs[f"turn{index}"].token_ids == expected


def common_prefix_len(a: list[int], b: list[int]) -> int:
    length = 0
    for left, right in zip(a, b):
        if left != right:
            break
        length += 1
    return length


def test_later_turns_reuse_exactly_the_blocks_they_share(vox_model, turns):
    """The claim the project rests on: the shared system prompt is paid for once.

    Asserted as a property rather than a threshold. Reuse should be the longest
    prefix this turn shares with any earlier one, rounded *down* to a block
    boundary -- a partial block cannot be shared -- and capped so one token is
    always left to compute.
    """
    block_size = 16
    _, outputs = run_turns(vox_model, turns, enable_prefix_caching=True, block_size=block_size)

    assert outputs["turn0"].num_cached_tokens == 0          # nothing to reuse yet

    for index in (1, 2):
        prompt = turns[index]
        shared = max(common_prefix_len(prompt, turns[j]) for j in range(index))
        expected = min(shared, len(prompt) - 1) // block_size * block_size
        assert outputs[f"turn{index}"].num_cached_tokens == expected, (
            f"turn{index} shares {shared} tokens with an earlier turn, so "
            f"{expected} should have been reused"
        )


def test_a_long_shared_prefix_is_mostly_free(vox_model, turns):
    """With a prefix this long, most of each later prompt is not recomputed."""
    _, outputs = run_turns(vox_model, turns, enable_prefix_caching=True)
    for index in (1, 2):
        output = outputs[f"turn{index}"]
        shared = max(common_prefix_len(turns[index], turns[j]) for j in range(index))
        # The shared part really is most of the prompt...
        assert shared / len(turns[index]) > 0.8
        # ...and the cache captured all of it that whole blocks can cover.
        assert output.num_cached_tokens >= shared - 16


@pytest.mark.parametrize("block_size", [1, 8, 16])
def test_block_size_does_not_change_cached_output(vox_model, turns, block_size):
    _, without = run_turns(vox_model, turns, enable_prefix_caching=False)
    engine, with_cache = run_turns(
        vox_model, turns, enable_prefix_caching=True, block_size=block_size
    )
    assert engine.num_cached_tokens_reused > 0
    for name in without:
        assert with_cache[name].token_ids == without[name].token_ids


def test_concurrent_requests_sharing_a_prefix_agree(vox_model, turns):
    """All three turns in flight at once, sharing the system prompt live."""
    cache = PagedKVCache(vox_model.cfg, num_blocks=512, block_size=16, dtype=vox_model.dtype)
    engine = Engine(vox_model, cache, SchedulerConfig())
    for index, ids in enumerate(turns):
        engine.add_request(f"turn{index}", ids, greedy())
    outputs = engine.run_until_done(max_steps=500)

    for index, ids in enumerate(turns):
        expected = generate_naive(vox_model, ids, greedy()).token_ids
        assert outputs[f"turn{index}"].token_ids == expected
    assert cache.pool.num_free_blocks == cache.num_blocks


def test_everything_is_released_even_with_blocks_still_cached(vox_model, turns):
    engine, _ = run_turns(vox_model, turns, enable_prefix_caching=True)
    pool_ = engine.cache.pool
    assert pool_.num_free_blocks == pool_.num_blocks     # nothing referenced
    assert pool_.num_cached_blocks > 0                   # but still findable
