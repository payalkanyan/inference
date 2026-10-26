"""Stage 4/5: iteration-level scheduling, chunked prefill, preemption.

No model and no weights -- these drive the scheduler directly, so each test pins
one policy claim rather than inferring it from generated text.
"""

from __future__ import annotations

import pytest

from vox.blocks import BlockPoolExhausted
from vox.cache import PagedKVCache
from vox.config import Qwen2Config
from vox.scheduler import Scheduler, SchedulerConfig
from vox.sequence import Sequence, SequenceStatus


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )


def make(num_blocks=64, block_size=4, clock=None, **cfg) -> tuple[Scheduler, PagedKVCache]:
    cache = PagedKVCache(tiny_config(), num_blocks=num_blocks, block_size=block_size)
    kwargs = {} if clock is None else {"clock": clock}
    return Scheduler(cache, SchedulerConfig(**cfg), **kwargs), cache


def seq(n_tokens: int) -> Sequence:
    return Sequence(token_ids=list(range(n_tokens)))


def decoding(scheduler: Scheduler, prompt_tokens: int) -> Sequence:
    """A running sequence that has prefilled its prompt and produced one token.

    That leaves it with exactly one uncomputed token, which is what a decode is.
    Building it by actually appending the generated token matters: a sequence
    whose ``num_computed_tokens`` sits below ``prompt_len`` is mid-prefill, not
    decoding, and the scheduler treats the two differently.
    """
    s = seq(prompt_tokens)
    scheduler.cache.allocate_for(s, prompt_tokens)
    s.num_computed_tokens = prompt_tokens
    s.append_token(999)
    s.status = SequenceStatus.RUNNING
    scheduler.running.append(s)
    return s


# -- admission ------------------------------------------------------------

def test_nothing_queued_schedules_nothing():
    scheduler, _ = make()
    assert scheduler.schedule().is_empty
    assert not scheduler.has_work


def test_a_waiting_request_is_admitted_and_prefilled_whole_when_it_fits():
    scheduler, _ = make()
    s = seq(10)
    scheduler.add(s)
    assert s.status is SequenceStatus.WAITING

    out = scheduler.schedule()
    assert out.sequences == [s]
    assert out.query_lens == [10]
    assert out.kinds == ["prefill"]
    assert out.newly_admitted == [s]
    assert s.status is SequenceStatus.RUNNING
    assert not scheduler.waiting


def test_admission_is_fifo():
    scheduler, _ = make(max_num_seqs=1)
    first, second = seq(4), seq(4)
    scheduler.add(first)
    scheduler.add(second)
    assert scheduler.schedule().sequences == [first]
    assert scheduler.schedule().sequences == [first]
    assert scheduler.waiting[0] is second


def test_max_num_seqs_caps_the_batch():
    scheduler, _ = make(max_num_seqs=2)
    for _ in range(5):
        scheduler.add(seq(3))
    out = scheduler.schedule()
    assert len(out.sequences) == 2
    assert len(scheduler.waiting) == 3


def test_a_prompt_too_large_for_the_pool_is_rejected_at_admission():
    """Rejected up front rather than chunk-prefilled until memory runs out.

    Admitting it would mean filling the pool and then preempting it forever,
    since it can never hold its own context.
    """
    scheduler, cache = make(num_blocks=2, block_size=4)
    with pytest.raises(ValueError, match="needs 5 blocks; the pool only has 2"):
        scheduler.add(seq(20))
    assert not scheduler.waiting
    assert cache.pool.num_free_blocks == 2


# -- chunked prefill ------------------------------------------------------

def test_a_prompt_larger_than_the_budget_is_chunked():
    """No overrun and no starvation: it takes what fits and comes back."""
    scheduler, _ = make(max_num_batched_tokens=8)
    big = seq(30)
    scheduler.add(big)

    out = scheduler.schedule()
    assert out.sequences == [big]
    assert out.query_lens == [8]
    assert out.num_query_tokens <= scheduler.config.max_num_batched_tokens


def test_chunks_cover_the_whole_prompt_over_successive_iterations():
    scheduler, _ = make(max_num_batched_tokens=7)
    big = seq(30)
    scheduler.add(big)

    chunks = []
    for _ in range(20):
        out = scheduler.schedule()
        if out.is_empty:
            break
        chunks.extend(out.query_lens)
        # Stand in for the forward pass.
        big.num_computed_tokens += out.query_lens[0]
        if not big.is_catching_up:
            break

    assert sum(chunks) == 30
    assert all(c <= 7 for c in chunks)
    assert big.num_computed_tokens == 30


def test_the_budget_is_split_across_admissions_by_chunking():
    """Two prompts of 6 under a budget of 10: the second takes a 4-token chunk.

    Before chunked prefill the second would have waited an entire iteration.
    """
    scheduler, _ = make(max_num_batched_tokens=10)
    a, b, c = seq(6), seq(6), seq(6)
    for s in (a, b, c):
        scheduler.add(s)

    out = scheduler.schedule()
    assert out.sequences == [a, b]
    assert out.query_lens == [6, 4]
    assert out.num_query_tokens == 10
    assert scheduler.waiting[0] is c


def test_a_long_prompt_shares_the_iteration_with_live_decodes():
    """The point of chunking: nobody's decoding stalls for a big prefill."""
    scheduler, _ = make(max_num_batched_tokens=8)
    live = decoding(scheduler, 4)
    big = seq(30)
    scheduler.add(big)

    out = scheduler.schedule()
    assert out.sequences == [live, big]
    assert out.query_lens == [1, 7]        # decode took its token first
    assert out.kinds == ["decode", "prefill"]


def test_a_chunk_shrinks_to_the_memory_available_rather_than_preempting():
    """A prefill can always give ground; this is why preemption is for decodes."""
    scheduler, cache = make(num_blocks=4, block_size=4, max_num_batched_tokens=1024)
    hog = decoding(scheduler, 8)           # holds 2 of the 4 blocks
    arriving = seq(16)
    scheduler.add(arriving)

    out = scheduler.schedule()
    assert arriving in out.sequences
    # Two blocks free, minus the one the decode takes -> room for a 4-token chunk.
    chunk = out.query_lens[out.sequences.index(arriving)]
    assert 0 < chunk < 16
    assert cache.pool.num_free_blocks == 0
    assert not out.preempted


# -- decode priority ------------------------------------------------------

def test_running_sequences_decode_one_token_each():
    scheduler, _ = make()
    a = decoding(scheduler, 5)
    b = decoding(scheduler, 9)

    out = scheduler.schedule()
    assert out.sequences == [a, b]
    assert out.query_lens == [1, 1]
    assert out.kinds == ["decode", "decode"]
    assert out.num_prefills == 0


def test_decodes_are_reserved_for_before_prefills():
    scheduler, _ = make(max_num_batched_tokens=12)
    live = decoding(scheduler, 6)
    arriving = seq(11)
    scheduler.add(arriving)

    out = scheduler.schedule()
    assert out.sequences == [live, arriving]
    assert out.query_lens == [1, 11]
    assert out.num_query_tokens == 12


def test_a_prefill_is_deferred_when_decodes_use_the_whole_budget():
    scheduler, _ = make(max_num_batched_tokens=2, max_num_seqs=8)
    a = decoding(scheduler, 4)
    b = decoding(scheduler, 4)
    scheduler.add(seq(5))

    out = scheduler.schedule()
    assert out.sequences == [a, b]
    assert len(scheduler.waiting) == 1


def test_a_newly_admitted_sequence_is_not_scheduled_twice():
    scheduler, _ = make()
    s = seq(4)
    scheduler.add(s)
    assert scheduler.schedule().sequences.count(s) == 1


def test_an_urgently_admitted_sequence_is_not_scheduled_twice():
    """Regression: the priority pass admits into `running` before the decode and
    catch-up passes walk it, so without a guard the same sequence is scheduled
    twice and num_computed_tokens advances twice for one forward pass.

    The plain version above cannot catch this, because ordinary admission is the
    last pass and nothing walks `running` after it.
    """
    scheduler, _ = make(clock=lambda: 100.0)
    urgent = Sequence(token_ids=list(range(6)), arrival_time=100.0, ttft_budget_s=0.01)
    scheduler.add(urgent)

    out = scheduler.schedule()
    assert out.sequences.count(urgent) == 1
    assert out.admitted_urgently == [urgent]
    assert sum(q for s_, q in zip(out.sequences, out.query_lens) if s_ is urgent) == 6


def test_a_one_token_urgent_prompt_is_not_scheduled_twice():
    """The decode pass is the other place a double-schedule could slip through:
    an urgent admission with a single uncomputed token looks like a decoder."""
    scheduler, _ = make(clock=lambda: 100.0)
    urgent = Sequence(token_ids=[7], arrival_time=100.0, ttft_budget_s=0.01)
    scheduler.add(urgent)
    out = scheduler.schedule()
    assert out.sequences.count(urgent) == 1
    assert out.query_lens == [1]


def test_decode_takes_a_new_block_only_on_crossing_a_boundary():
    scheduler, _ = make(block_size=4)
    s = decoding(scheduler, 4)             # 4 tokens -> 1 block, next is position 4
    assert len(s.block_table) == 1

    scheduler.schedule()
    assert len(s.block_table) == 2         # position 4 opens a second block

    s.num_computed_tokens = 5
    s.append_token(998)
    scheduler.schedule()
    assert len(s.block_table) == 2         # position 5 is inside it


# -- preemption -----------------------------------------------------------

def four_decoders_on_a_full_pool() -> tuple[Scheduler, PagedKVCache, list[Sequence]]:
    """Four sequences, each holding one block of a four-block pool, each needing
    a second block for its next token. Two of them must give way."""
    scheduler, cache = make(num_blocks=4, block_size=4)
    sequences = [decoding(scheduler, 4) for _ in range(4)]
    assert cache.pool.num_free_blocks == 0
    assert all(scheduler._needs_new_block(s) for s in sequences)
    return scheduler, cache, sequences


def test_preemption_frees_exactly_enough_and_no_more():
    scheduler, cache, (a, b, c, d) = four_decoders_on_a_full_pool()

    out = scheduler.schedule()

    assert out.preempted == [d, c]          # newest first, and only two needed
    assert out.sequences == [a, b]
    assert out.query_lens == [1, 1]


def test_the_oldest_sequence_is_never_the_victim():
    """The progress guarantee: evicting the newest means the oldest finishes."""
    scheduler, _, (a, b, c, d) = four_decoders_on_a_full_pool()
    out = scheduler.schedule()
    assert a not in out.preempted
    assert a in out.sequences


def test_a_preempted_sequence_keeps_its_tokens_and_loses_its_kv():
    scheduler, cache, (_, _, _, d) = four_decoders_on_a_full_pool()
    tokens_before = list(d.token_ids)

    scheduler.schedule()

    assert d.status is SequenceStatus.PREEMPTED
    assert d.token_ids == tokens_before     # nothing generated is thrown away
    assert d.num_computed_tokens == 0       # but all of its KV is
    assert d.block_table == []
    assert d.num_preemptions == 1
    assert d.was_preempted


def test_a_preempted_sequence_goes_to_the_front_of_the_queue():
    scheduler, _, (_, _, c, d) = four_decoders_on_a_full_pool()
    newcomer = seq(4)
    scheduler.add(newcomer)

    scheduler.schedule()

    # c was preempted after d, so it is now ahead of it, and both are ahead of
    # the request that merely arrived.
    assert list(scheduler.waiting) == [c, d, newcomer]


def test_a_sequence_preempted_this_iteration_is_not_re_admitted_at_once():
    """Otherwise it would be evicted and recomputed in a tight loop."""
    scheduler, _, (_, _, c, d) = four_decoders_on_a_full_pool()
    out = scheduler.schedule()
    assert c not in out.sequences
    assert d not in out.sequences
    assert c not in out.newly_admitted


def test_a_preempted_sequence_resumes_by_recomputing_everything():
    scheduler, _, sequences = four_decoders_on_a_full_pool()
    victim = sequences[-1]
    scheduler.schedule()
    assert victim.num_computed_tokens == 0

    # Retire everyone still running so there is room again.
    for s in list(scheduler.running):
        scheduler.finish(s, "stop")

    out = scheduler.schedule()
    assert victim in out.sequences
    index = out.sequences.index(victim)
    assert out.query_lens[index] == victim.num_tokens   # the whole context again
    assert out.kinds[index] == "prefill"


def test_recompute_past_the_prompt_is_labelled_as_such():
    """A sequence that had been generating reports "recompute", not "prefill"."""
    scheduler, _ = make()
    s = seq(4)
    scheduler.cache.allocate_for(s, 4)
    s.num_computed_tokens = 4
    for token in (101, 102, 103):
        s.append_token(token)
    s.status = SequenceStatus.RUNNING
    scheduler.running.append(s)

    scheduler.cache.free(s)
    s.reset_for_recompute()
    s.num_computed_tokens = 4              # prompt rebuilt, generated tokens not
    assert s.work_kind == "recompute"


def test_a_context_that_outgrows_the_pool_fails_loudly():
    """One sequence, nobody to evict, and it no longer fits. Not a livelock."""
    scheduler, cache = make(num_blocks=1, block_size=4)
    s = seq(4)
    cache.allocate_for(s, 4)
    s.num_computed_tokens = 4
    s.append_token(99)                     # position 4 needs a second block
    s.status = SequenceStatus.RUNNING
    scheduler.running.append(s)

    with pytest.raises(BlockPoolExhausted, match="no longer fits"):
        scheduler.schedule()


def test_nothing_is_preempted_when_there_is_room():
    scheduler, _ = make(num_blocks=64)
    for _ in range(4):
        decoding(scheduler, 4)
    out = scheduler.schedule()
    assert out.preempted == []
    assert len(out.sequences) == 4


# -- retirement -----------------------------------------------------------

def test_finishing_releases_blocks_and_leaves_the_running_set():
    scheduler, cache = make()
    s = decoding(scheduler, 8)
    free_before = cache.pool.num_free_blocks

    scheduler.finish(s, "stop")
    assert s.status is SequenceStatus.FINISHED
    assert s.finish_reason == "stop"
    assert s.block_table == []
    assert cache.pool.num_free_blocks > free_before
    assert scheduler.running == []


def test_freed_blocks_let_a_queued_request_in():
    # 8 tokens -> 2 of the 3 blocks, leaving exactly enough for its own next
    # token and nothing for anybody else. (12 would make the holder itself
    # unrunnable, which test_a_context_that_outgrows_the_pool_fails_loudly covers.)
    scheduler, _ = make(num_blocks=3, block_size=4)
    holder = decoding(scheduler, 8)
    waiting = seq(8)
    scheduler.add(waiting)

    assert scheduler.schedule().sequences == [holder]
    scheduler.finish(holder, "stop")
    assert scheduler.schedule().sequences == [waiting]


def test_has_work_tracks_both_queues():
    scheduler, _ = make()
    assert not scheduler.has_work
    s = seq(4)
    scheduler.add(s)
    assert scheduler.has_work
    scheduler.schedule()
    assert scheduler.has_work
    scheduler.finish(s, "stop")
    assert not scheduler.has_work


def test_identical_sequences_are_distinct_entities():
    """Two turns replaying the same prompt must not be interchangeable.

    Sequence uses identity equality precisely so that retiring one of them
    cannot remove the other from the running set.
    """
    scheduler, _ = make()
    a, b = decoding(scheduler, 4), decoding(scheduler, 4)
    assert a.token_ids == b.token_ids and a is not b
    assert a != b

    scheduler.finish(a, "stop")
    assert scheduler.running == [b]


def test_a_priority_admission_cannot_spend_the_decode_reservation():
    """Regression: `_make_room` guarantees every running sequence can advance one
    token, and the priority pass runs *before* the decode pass. An admission
    there must not allocate the blocks that guarantee depends on, or the decode
    it was reserved for fails with BlockPoolExhausted.

    Two decoders each need a second block and exactly two blocks are free, so
    there is nothing spare: the resuming sequence has to wait a turn.
    """
    scheduler, cache = make(num_blocks=4, block_size=4, clock=lambda: 100.0)
    first = decoding(scheduler, 4)
    second = decoding(scheduler, 4)
    assert cache.pool.num_free_blocks == 2
    assert all(scheduler._needs_new_block(s) for s in (first, second))

    # Committed work, preempted during some earlier iteration.
    resuming = Sequence(token_ids=list(range(8)), arrival_time=100.0)
    resuming.first_token_time = 100.0
    scheduler.add(resuming)

    out = scheduler.schedule()

    assert out.sequences == [first, second]
    assert out.query_lens == [1, 1]
    assert resuming in scheduler.waiting
    assert resuming.block_table == []
    assert cache.pool.num_free_blocks == 0      # both went to the decodes
