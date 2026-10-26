"""Stage 7: the Turn abstraction and deadline-aware admission.

Time is injected everywhere here. Deadline logic asserted against a real clock is
either flaky or vacuous, so the scheduler and engine both take a ``clock`` and
these tests drive it by hand.
"""

from __future__ import annotations

import pytest
import torch

from vox.cache import PagedKVCache
from vox.config import Qwen2Config
from vox.engine import Engine
from vox.sampler import SamplingParams
from vox.scheduler import Scheduler, SchedulerConfig
from vox.sequence import Sequence, SequenceStatus
from vox.turn import Turn


class FakeClock:
    """A clock that only moves when a test says so."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )


def make(clock, num_blocks=64, block_size=4, **cfg):
    cache = PagedKVCache(tiny_config(), num_blocks=num_blocks, block_size=block_size)
    return Scheduler(cache, SchedulerConfig(**cfg), clock=clock), cache


def waiting(scheduler, n_tokens: int, *, budget=None, arrival=None) -> Sequence:
    s = Sequence(
        token_ids=list(range(n_tokens)),
        arrival_time=scheduler.clock() if arrival is None else arrival,
        ttft_budget_s=budget,
    )
    scheduler.add(s)
    return s


def decoding(scheduler, prompt_tokens: int) -> Sequence:
    s = Sequence(token_ids=list(range(prompt_tokens)), arrival_time=scheduler.clock())
    scheduler.cache.allocate_for(s, prompt_tokens)
    s.num_computed_tokens = prompt_tokens
    s.append_token(999)
    s.first_token_time = scheduler.clock()
    s.status = SequenceStatus.RUNNING
    scheduler.running.append(s)
    return s


# -- the Turn type --------------------------------------------------------

def test_a_turn_is_a_prefix_plus_a_segment():
    t = Turn("r1", tuple(range(100)), (500, 501), ttft_budget_s=0.3)
    assert t.prefix_len == 100
    assert t.num_tokens == 102
    assert t.token_ids == list(range(100)) + [500, 501]


def test_a_turn_with_nothing_new_is_refused():
    """A bare prefix has no question in it."""
    with pytest.raises(ValueError, match="needs a new segment"):
        Turn("r1", tuple(range(10)), ())


def test_a_turn_may_have_no_prefix():
    t = Turn("r1", (), (1, 2, 3))
    assert t.prefix_len == 0
    assert t.token_ids == [1, 2, 3]


# -- deadline ordering ----------------------------------------------------

def test_a_tighter_deadline_is_admitted_first_even_if_it_arrived_later():
    """Earliest deadline first, not first come first served."""
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_seqs=1)

    relaxed = waiting(scheduler, 4, budget=5.0)
    clock.advance(0.01)
    urgent = waiting(scheduler, 4, budget=0.2)

    assert scheduler.schedule().sequences == [urgent]


def test_requests_without_a_deadline_yield_to_those_with_one():
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_seqs=1)

    anytime = waiting(scheduler, 4, budget=None)
    clock.advance(0.01)
    deadlined = waiting(scheduler, 4, budget=10.0)

    assert scheduler.schedule().sequences == [deadlined]


def test_requests_without_deadlines_keep_arrival_order_among_themselves():
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_seqs=1)
    first = waiting(scheduler, 4)
    clock.advance(0.01)
    waiting(scheduler, 4)
    assert scheduler.schedule().sequences == [first]


def test_a_resuming_sequence_outranks_a_new_one_with_a_deadline():
    """Stage 5's progress guarantee survives deadline ordering.

    A preempted sequence has already produced a token, so its TTFT is settled;
    what it needs is to finish. Letting new arrivals jump it would reopen the
    starvation that newest-first eviction exists to prevent.
    """
    clock = FakeClock()
    scheduler, cache = make(clock, max_num_seqs=1)

    resuming = Sequence(token_ids=list(range(8)), arrival_time=clock())
    resuming.first_token_time = clock()
    resuming.append_token(999)
    scheduler.add(resuming)

    waiting(scheduler, 4, budget=0.01)      # desperately urgent, but new
    assert scheduler.schedule().sequences == [resuming]


# -- urgency outranks throughput -----------------------------------------

def test_an_at_risk_request_is_admitted_ahead_of_the_decode_pass():
    """The one place latency is deliberately bought at throughput's expense."""
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_batched_tokens=6, at_risk_margin_s=0.15)
    live = decoding(scheduler, 4)
    urgent = waiting(scheduler, 5, budget=0.1)   # deadline inside the margin

    out = scheduler.schedule()
    assert out.admitted_urgently == [urgent]
    assert out.sequences == [urgent, live]       # the prefill went first
    assert out.query_lens == [5, 1]


def test_a_relaxed_request_waits_for_the_normal_pass():
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_batched_tokens=6, at_risk_margin_s=0.15)
    live = decoding(scheduler, 4)
    relaxed = waiting(scheduler, 5, budget=10.0)

    out = scheduler.schedule()
    assert out.admitted_urgently == []
    assert out.sequences == [live, relaxed]      # decode first, as usual


def test_urgent_admission_cannot_starve_the_decodes():
    """An urgent new turn must not freeze conversations already in progress.

    The decode pass keeps a reservation of one token per running sequence, so
    however many urgent prefills arrive, everyone already talking still advances.
    """
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_batched_tokens=4, at_risk_margin_s=0.15)
    live = [decoding(scheduler, 4) for _ in range(3)]
    urgent = waiting(scheduler, 20, budget=0.05)

    out = scheduler.schedule()
    for sequence in live:
        assert sequence in out.sequences
    # One token was left over for the urgent prefill, not the twenty it wanted.
    assert out.query_lens[out.sequences.index(urgent)] == 1


def test_a_deadline_already_past_still_counts_as_urgent():
    """Recorded, not abandoned. Shedding doomed work is a product decision."""
    clock = FakeClock()
    scheduler, _ = make(clock, max_num_batched_tokens=6)
    live = decoding(scheduler, 4)
    overdue = waiting(scheduler, 5, budget=0.1)
    clock.advance(10.0)                          # long past the deadline

    out = scheduler.schedule()
    assert out.admitted_urgently == [overdue]
    assert live in out.sequences


def test_urgency_stops_mattering_once_the_first_token_is_out():
    clock = FakeClock()
    scheduler, _ = make(clock)
    s = waiting(scheduler, 4, budget=0.01)
    assert s.is_at_risk(clock(), 0.15)
    s.first_token_time = clock()
    assert not s.is_at_risk(clock(), 0.15)


# -- end to end -----------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    from vox.hub import resolve_model_dir

    return resolve_model_dir()


@pytest.fixture(scope="module")
def tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


@pytest.fixture(scope="module")
def vox_model(model_dir):
    from vox.weights import load_model

    return load_model(model_dir, dtype=torch.float32, device="cpu")


SYSTEM = (
    "You are a concise voice assistant for a scheduling service. Answer in one "
    "short sentence and never more than twenty words. Speak plainly and do not "
    "use lists. You have access to a calendar tool that can read and create "
    "events, a weather tool that reports current conditions for a city, and a "
    "contacts tool that looks up phone numbers by name. Never invent facts. "
    "User question: "
)


@pytest.fixture(scope="module")
def prefix(tokenizer):
    return tuple(tokenizer(SYSTEM, add_special_tokens=False)["input_ids"])


def engine_for(vox_model, clock, *, num_blocks=512, block_size=16, **cfg) -> Engine:
    cache = PagedKVCache(
        vox_model.cfg, num_blocks=num_blocks, block_size=block_size, dtype=vox_model.dtype
    )
    return Engine(vox_model, cache, SchedulerConfig(**cfg), clock=clock)


def a_turn(tokenizer, prefix, request_id, text, **kwargs) -> Turn:
    return Turn(
        request_id=request_id,
        prefix_token_ids=prefix,
        segment_token_ids=tuple(tokenizer(text, add_special_tokens=False)["input_ids"]),
        params=SamplingParams(temperature=0.0, max_new_tokens=6),
        **kwargs,
    )


def test_a_met_deadline_is_recorded_as_met(vox_model, tokenizer, prefix):
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour.", ttft_budget_s=0.3))

    engine.step()                                # clock does not move on its own
    output = engine.step()[0]

    assert output.ttft_s == 0.0
    assert not output.deadline_missed
    assert engine.num_deadline_misses == 0


def test_a_missed_deadline_is_recorded_and_the_request_still_runs(
    vox_model, tokenizer, prefix
):
    """The spec's requirement: record the miss, do not silently continue *or*
    silently drop. A late first token is still delivered."""
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour.", ttft_budget_s=0.2))

    clock.advance(1.0)                           # the first token is now late
    outputs = engine.run_until_done(max_steps=200)

    assert outputs["a"].deadline_missed
    assert outputs["a"].ttft_s == pytest.approx(1.0)
    assert engine.num_deadline_misses == 1
    assert outputs["a"].num_output_tokens == 6   # and it ran to completion
    assert outputs["a"].finish_reason == "length"


def test_a_request_without_a_budget_can_never_miss(vox_model, tokenizer, prefix):
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour."))
    clock.advance(100.0)
    outputs = engine.run_until_done(max_steps=200)
    assert not outputs["a"].deadline_missed
    assert engine.num_deadline_misses == 0


# -- declared prefixes ----------------------------------------------------

def test_a_declared_prefix_is_reported_against_its_own_denominator(
    vox_model, tokenizer, prefix
):
    """A turn that declared a long reusable prefix and reused little of it is a
    problem worth seeing, and the whole-prompt ratio hides it."""
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "first", "What is the capital of France?"))
    engine.run_until_done(max_steps=300)

    engine.add_turn(a_turn(tokenizer, prefix, "second", "And of Italy?"))
    outputs = engine.run_until_done(max_steps=300)

    second = outputs["second"]
    assert second.declared_prefix_hit_ratio > 0.8
    assert second.declared_prefix_hit_ratio >= second.prefix_hit_ratio


def test_reuse_does_not_speculate_past_the_declared_prefix(vox_model, tokenizer, prefix):
    """The bound a declaration buys: the new segment is where prompts differ."""
    clock = FakeClock()
    engine = engine_for(vox_model, clock, block_size=1)   # no block granularity to hide behind
    engine.add_turn(a_turn(tokenizer, prefix, "first", "Name a colour."))
    engine.run_until_done(max_steps=300)

    # Same prefix *and* the same segment, so without the bound the engine would
    # happily reuse into the segment too.
    engine.add_turn(a_turn(tokenizer, prefix, "repeat", "Name a colour."))
    outputs = engine.run_until_done(max_steps=300)

    assert outputs["repeat"].num_cached_tokens == len(prefix)


def test_a_request_without_a_turn_has_no_declared_prefix(vox_model, tokenizer, prefix):
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_request("plain", list(prefix) + [100], SamplingParams(0.0, 2))
    outputs = engine.run_until_done(max_steps=200)
    assert outputs["plain"].declared_prefix_hit_ratio == 0.0


# -- keeping a declared prefix resident -----------------------------------

def test_a_retained_prefix_survives_eviction_pressure(vox_model, tokenizer, prefix):
    """What declaring a prefix buys operationally.

    Other conversations churn through a small pool. Without retention the system
    prompt is evicted and the next turn pays full price for it; with retention
    the blocks are referenced, so eviction cannot take them.
    """
    clock = FakeClock()
    engine = engine_for(vox_model, clock, num_blocks=24, block_size=16)

    engine.add_turn(a_turn(tokenizer, prefix, "warm", "Name a colour."))
    engine.run_until_done(max_steps=300)

    pinned = engine.retain_prefix(list(prefix))
    assert pinned > 0
    assert engine.num_retained_blocks > 0

    # Churn: unrelated prompts long enough to want the whole pool.
    for index in range(6):
        filler = [9000 + index] * 200
        engine.add_request(f"filler{index}", filler, SamplingParams(0.0, 2))
        engine.run_until_done(max_steps=300)

    engine.add_turn(a_turn(tokenizer, prefix, "later", "And of Italy?"))
    outputs = engine.run_until_done(max_steps=300)
    assert outputs["later"].declared_prefix_hit_ratio > 0.8


def test_retention_can_be_released(vox_model, tokenizer, prefix):
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "warm", "Name a colour."))
    engine.run_until_done(max_steps=300)

    engine.retain_prefix(list(prefix))
    assert engine.num_retained_blocks > 0

    engine.release_prefix(list(prefix))
    assert engine.num_retained_blocks == 0
    assert engine.cache.pool.num_used_blocks == 0


def test_retaining_something_never_computed_pins_nothing():
    """Retention references existing blocks; it does not compute them."""
    cache = PagedKVCache(tiny_config(), num_blocks=16, block_size=4)
    engine = Engine.__new__(Engine)          # no model needed for this path
    engine.cache = cache
    engine._retained = {}
    assert Engine.retain_prefix(engine, list(range(32))) == 0
    assert Engine.num_retained_blocks.fget(engine) == 0
