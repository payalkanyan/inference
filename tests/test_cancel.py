"""Stage 7: barge-in.

The headline claim of the project is in ``test_cancel_frees_on_the_same_tick``
and ``test_cancelling_leaves_a_shared_prefix_standing``. Everything else here
guards the ways a cancel path goes wrong in practice: racing a completion,
cancelling something that never ran, and -- the one that matters most -- taking
somebody else's memory with it.
"""

from __future__ import annotations

import pytest
import torch

from vox.cache import PagedKVCache
from vox.engine import Engine
from vox.generate import generate_naive
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.sequence import SequenceStatus
from vox.turn import Turn
from vox.weights import load_model

MAX_NEW_TOKENS = 12

SYSTEM = (
    "You are a concise voice assistant for a scheduling service. Answer in one "
    "short sentence and never more than twenty words. Speak plainly and do not "
    "use lists. You have access to a calendar tool that can read and create "
    "events, a weather tool that reports current conditions for a city, and a "
    "contacts tool that looks up phone numbers by name. Never invent facts. "
    "User question: "
)


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


@pytest.fixture(scope="module")
def prefix(tokenizer):
    return tuple(tokenizer(SYSTEM, add_special_tokens=False)["input_ids"])


def segment(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])


def greedy(n: int = MAX_NEW_TOKENS) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_new_tokens=n)


def engine_for(vox_model, *, num_blocks=512, block_size=16, **cfg) -> Engine:
    cache = PagedKVCache(
        vox_model.cfg, num_blocks=num_blocks, block_size=block_size, dtype=vox_model.dtype
    )
    return Engine(vox_model, cache, SchedulerConfig(**cfg))


def turn(tokenizer, prefix, request_id, text, **kwargs) -> Turn:
    return Turn(
        request_id=request_id,
        prefix_token_ids=prefix,
        segment_token_ids=segment(tokenizer, text),
        params=greedy(),
        **kwargs,
    )


# -- the headline ---------------------------------------------------------

def test_cancel_frees_on_the_same_tick(vox_model, tokenizer, prefix):
    """No iteration elapses between cancel() and the blocks being back.

    This is the claim the whole project is organised around, so it is asserted
    on the tick counter rather than on wall clock: cancellation does not go
    through the engine loop at all, so the number of iterations between asking
    and having the memory back is exactly zero.
    """
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "a", "What is the capital of France?"))

    for _ in range(4):                      # get it well into decoding
        engine.step()
    sequence = engine.requests["a"]
    assert sequence.num_output_tokens > 0
    assert sequence.block_table

    iterations_before = engine.num_iterations
    record = engine.cancel("a")

    assert record is not None
    assert record.ticks_to_free == 0
    assert engine.num_iterations == iterations_before   # nothing ran
    assert record.blocks_released > 0
    assert sequence.block_table == []
    assert sequence.status is SequenceStatus.CANCELLED
    assert sequence.finish_reason == "cancelled"
    # Reported for interest, not asserted tightly: this is a few list operations.
    assert record.wall_seconds < 0.1


def test_cancelling_leaves_a_shared_prefix_standing(vox_model, tokenizer, prefix):
    """A barge-in must not take the system prompt with it.

    Two turns of the same conversation share the prefix blocks. Cancelling one
    has to release exactly the blocks it alone owns and leave the shared ones
    referenced, which is the reference count doing the work it was built for in
    stage 3.
    """
    engine = engine_for(vox_model)

    # First turn runs to completion so its blocks are published and findable.
    engine.add_turn(turn(tokenizer, prefix, "warm", "What is the capital of France?"))
    engine.run_until_done(max_steps=500)

    engine.add_turn(turn(tokenizer, prefix, "b", "And of Italy?"))
    engine.add_turn(turn(tokenizer, prefix, "c", "Name a colour."))
    for _ in range(4):
        engine.step()

    shared = set(engine.requests["b"].block_table) & set(engine.requests["c"].block_table)
    assert shared, "the two turns did not end up sharing any blocks"

    record = engine.cancel("b")

    assert record.blocks_retained == len(shared)
    assert record.blocks_released == record.blocks_held - len(shared)
    for block_id in shared:
        assert not engine.cache.pool.blocks[block_id].is_free
        assert block_id in engine.requests["c"].block_table


def test_the_survivor_of_a_barge_in_is_unaffected(vox_model, tokenizer, prefix):
    """Cancelling one request must not perturb a single token of another."""
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "doomed", "And of Italy?"))
    engine.add_turn(turn(tokenizer, prefix, "survivor", "Name a colour."))
    for _ in range(3):
        engine.step()

    engine.cancel("doomed")
    outputs = engine.run_until_done(max_steps=500)

    alone = generate_naive(
        vox_model, list(prefix) + list(segment(tokenizer, "Name a colour.")), greedy()
    ).token_ids
    assert outputs["survivor"].token_ids == alone


def test_freed_blocks_are_usable_on_the_same_tick(vox_model, tokenizer, prefix):
    """The point of freeing immediately: the replacement turn starts at once.

    A barge-in is followed by a new utterance, so the memory has to be available
    to the turn that caused the interruption, not one iteration later.
    """
    engine = engine_for(vox_model, num_blocks=24, block_size=16)
    engine.add_turn(turn(tokenizer, prefix, "doomed", "What is the capital of France?"))
    for _ in range(3):
        engine.step()

    free_before = engine.cache.pool.num_free_blocks
    engine.cancel("doomed")
    assert engine.cache.pool.num_free_blocks > free_before

    # Admitted on the very next scheduling pass, with no iteration in between.
    engine.add_turn(turn(tokenizer, prefix, "barge", "Name a colour."))
    scheduled = engine.scheduler.schedule()
    assert engine.requests["barge"] in scheduled.sequences


# -- the states a cancel can arrive in ------------------------------------

def test_cancelling_a_queued_request_releases_its_adopted_prefix(vox_model, tokenizer, prefix):
    """A waiting request may already hold blocks: it adopts its prefix at
    admission, and a cancel before it ever runs still has to give them back."""
    engine = engine_for(vox_model, num_blocks=24, block_size=16, max_num_seqs=1)
    engine.add_turn(turn(tokenizer, prefix, "warm", "What is the capital of France?"))
    engine.run_until_done(max_steps=500)

    engine.add_turn(turn(tokenizer, prefix, "running", "And of Italy?"))
    engine.add_turn(turn(tokenizer, prefix, "queued", "Name a colour."))
    engine.step()                           # max_num_seqs=1, so one of them waits

    queued = engine.requests["queued"]
    assert queued.status is SequenceStatus.WAITING

    record = engine.cancel("queued")
    assert record is not None
    assert record.ticks_to_free == 0
    assert record.tokens_generated == 0
    assert not record.had_first_token
    assert queued.block_table == []
    assert queued not in engine.scheduler.waiting


def test_cancelling_a_finished_request_is_not_an_error(vox_model, tokenizer, prefix):
    """A barge-in racing a completion is normal traffic, not a bug."""
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "a", "Name a colour."))
    engine.run_until_done(max_steps=500)

    assert engine.cancel("a") is None
    assert engine.requests["a"].finish_reason == "length"   # unchanged
    assert engine.cancellations == []


def test_cancelling_twice_records_once(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "a", "Name a colour."))
    engine.step()

    assert engine.cancel("a") is not None
    assert engine.cancel("a") is None
    assert len(engine.cancellations) == 1


def test_cancelling_an_unknown_request_is_a_bug(vox_model):
    engine = engine_for(vox_model)
    with pytest.raises(KeyError):
        engine.cancel("never-submitted")


def test_a_cancelled_request_is_never_scheduled_again(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "a", "Name a colour."))
    engine.step()
    engine.cancel("a")

    assert not engine.has_work
    assert engine.step() == []
    assert engine.requests["a"] not in engine.scheduler.running


def test_cancelling_before_the_first_token(vox_model, tokenizer, prefix):
    """Barge-in during prefill: there is no output yet, but memory is held."""
    engine = engine_for(vox_model, max_num_batched_tokens=8)
    engine.add_turn(turn(tokenizer, prefix, "a", "What is the capital of France?"))
    engine.step()                           # one chunk only

    sequence = engine.requests["a"]
    assert sequence.needs_prefill
    assert not sequence.has_first_token

    record = engine.cancel("a")
    assert record.ticks_to_free == 0
    assert record.tokens_generated == 0
    assert not record.had_first_token
    assert record.blocks_released > 0


def test_everything_comes_back_after_a_cancel(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(turn(tokenizer, prefix, "a", "Name a colour."))
    for _ in range(3):
        engine.step()
    engine.cancel("a")

    pool = engine.cache.pool
    assert pool.num_used_blocks == 0
    assert all(block.is_free for block in pool.blocks)
    # ...and the prefix it computed is still findable for the next turn.
    assert pool.num_cached_blocks > 0


def test_cancellations_accumulate_in_order(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    for name in ("a", "b"):
        engine.add_turn(turn(tokenizer, prefix, name, "Name a colour."))
    engine.step()

    engine.cancel("a")
    engine.cancel("b")
    assert [r.request_id for r in engine.cancellations] == ["a", "b"]
    assert all(r.ticks_to_free == 0 for r in engine.cancellations)


def test_two_barge_ins_in_one_conversation(vox_model, tokenizer, prefix):
    """The demo's shape: a turn is interrupted, replaced, and interrupted again."""
    engine = engine_for(vox_model)
    questions = ["What is the capital of France?", "And of Italy?", "Name a colour."]

    for index, question in enumerate(questions[:2]):
        engine.add_turn(turn(tokenizer, prefix, f"t{index}", question))
        for _ in range(3):
            engine.step()
        engine.cancel(f"t{index}")

    engine.add_turn(turn(tokenizer, prefix, "t2", questions[2]))
    outputs = engine.run_until_done(max_steps=500)

    assert len(engine.cancellations) == 2
    assert all(r.ticks_to_free == 0 for r in engine.cancellations)
    alone = generate_naive(
        vox_model, list(prefix) + list(segment(tokenizer, questions[2])), greedy()
    ).token_ids
    assert outputs["t2"].token_ids == alone
    assert engine.cache.pool.num_used_blocks == 0
