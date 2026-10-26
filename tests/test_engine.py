"""Stage 4: continuous batching must not change the output.

The gate is ``test_batched_output_matches_running_each_alone``. Everything a
batch introduces -- sharing a forward pass, joining mid-flight, leaving at
different times -- has to be invisible in the tokens, so most of these tests are
variations on "and this still produces exactly what it would have alone".
"""

from __future__ import annotations

import pytest
import torch

from vox.batch import build_batch
from vox.cache import PagedKVCache
from vox.engine import Engine
from vox.generate import generate_naive
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.sequence import Sequence
from vox.weights import load_model

MAX_NEW_TOKENS = 10

PROMPTS = {
    "france": "The capital of France is",
    "count": "Count from one to ten:",
    "colour": "The sky is",
    "code": "def add(a, b):",
}


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
def encoded(tokenizer):
    return {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in PROMPTS.items()}


@pytest.fixture(scope="module")
def alone(vox_model, encoded):
    """What each prompt produces with no batching at all."""
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
    return {k: generate_naive(vox_model, ids, params).token_ids for k, ids in encoded.items()}


def engine_for(vox_model, *, num_blocks=128, block_size=16, **cfg) -> Engine:
    cache = PagedKVCache(
        vox_model.cfg, num_blocks=num_blocks, block_size=block_size, dtype=vox_model.dtype
    )
    return Engine(vox_model, cache, SchedulerConfig(**cfg))


def greedy(n: int = MAX_NEW_TOKENS) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_new_tokens=n)


# -- the gate -------------------------------------------------------------

def test_single_request_matches_the_naive_reference(vox_model, encoded, alone):
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy())
    outputs = engine.run_until_done()
    assert outputs["france"].token_ids == alone["france"]


def test_batched_output_matches_running_each_alone(vox_model, encoded, alone):
    """THE stage-4 gate: sharing a forward pass must change nothing."""
    engine = engine_for(vox_model)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())

    outputs = engine.run_until_done()

    assert set(outputs) == set(encoded)
    for name in encoded:
        assert outputs[name].token_ids == alone[name], (
            f"{name} diverged when batched\n"
            f"  alone  : {alone[name]}\n"
            f"  batched: {outputs[name].token_ids}"
        )


def test_requests_joining_mid_flight_perturb_nobody(vox_model, encoded, alone):
    """Continuous batching: admit new work without disturbing what is running."""
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy())

    collected: dict[str, list[int]] = {}
    arrivals = {2: "count", 4: "colour", 5: "code"}

    for step_index in range(200):
        if step_index in arrivals:
            name = arrivals[step_index]
            engine.add_request(name, encoded[name], greedy())
        if not engine.has_work:
            break
        for output in engine.step():
            collected[output.request_id] = output.token_ids

    assert set(collected) == set(encoded)
    for name in encoded:
        assert collected[name] == alone[name]


# -- invariance to scheduling knobs ---------------------------------------

@pytest.mark.parametrize("max_num_batched_tokens", [1024, 32, 8])
def test_token_budget_does_not_change_the_output(
    vox_model, encoded, alone, max_num_batched_tokens
):
    """A smaller budget reshapes the batches, not the answers.

    At 8 tokens every prompt here is oversized, so each one takes an iteration
    alone -- the path that deliberately overruns the budget.
    """
    engine = engine_for(vox_model, max_num_batched_tokens=max_num_batched_tokens)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


@pytest.mark.parametrize("max_num_seqs", [1, 2, 8])
def test_batch_size_cap_does_not_change_the_output(vox_model, encoded, alone, max_num_seqs):
    engine = engine_for(vox_model, max_num_seqs=max_num_seqs)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


@pytest.mark.parametrize("block_size", [1, 3, 16])
def test_block_size_does_not_change_batched_output(vox_model, encoded, alone, block_size):
    engine = engine_for(vox_model, num_blocks=512, block_size=block_size)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


def test_arrival_order_does_not_change_the_output(vox_model, encoded, alone):
    names = list(encoded)
    engine = engine_for(vox_model)
    for name in reversed(names):
        engine.add_request(name, encoded[name], greedy())
    outputs = engine.run_until_done()
    for name in names:
        assert outputs[name].token_ids == alone[name]


# -- leaving the batch ----------------------------------------------------

def test_sequences_finishing_at_different_lengths_all_get_what_they_asked_for(
    vox_model, encoded, alone
):
    """Finished requests leave mid-flight while the rest keep going."""
    wanted = {"france": 3, "count": 10, "colour": 6, "code": 8}
    engine = engine_for(vox_model)
    for name, n in wanted.items():
        engine.add_request(name, encoded[name], greedy(n))

    outputs = engine.run_until_done()
    for name, n in wanted.items():
        assert outputs[name].num_output_tokens == n
        assert outputs[name].finish_reason == "length"
        # A shorter request is a prefix of the same greedy continuation.
        assert outputs[name].token_ids == alone[name][:n]


def test_a_stop_token_retires_only_its_own_request(vox_model, encoded, alone):
    stop_on = alone["france"][2]
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy(), stop_token_ids={stop_on})
    engine.add_request("count", encoded["count"], greedy())

    outputs = engine.run_until_done()
    assert outputs["france"].finish_reason == "stop"
    assert outputs["france"].token_ids == alone["france"][:3]
    assert outputs["count"].finish_reason == "length"
    assert outputs["count"].token_ids == alone["count"]


def test_all_blocks_are_returned_once_everything_finishes(vox_model, encoded):
    engine = engine_for(vox_model)
    free_before = engine.cache.pool.num_free_blocks
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    engine.run_until_done()

    assert engine.cache.pool.num_free_blocks == free_before
    assert all(block.is_free for block in engine.cache.pool.blocks)
    assert not engine.has_work


def test_a_small_pool_serialises_requests_instead_of_failing(vox_model, encoded, alone):
    """Not enough memory for everyone at once: they queue and still come out right."""
    engine = engine_for(vox_model, num_blocks=6, block_size=16)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


# -- bookkeeping ----------------------------------------------------------

def test_step_reports_progress_for_every_scheduled_sequence(vox_model, encoded):
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy())
    engine.add_request("count", encoded["count"], greedy())

    first = engine.step()
    assert {o.request_id for o in first} == {"france", "count"}
    assert all(o.num_output_tokens == 1 for o in first)
    assert all(not o.finished for o in first)


def test_output_carries_the_prompt_back(vox_model, encoded):
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy(2))
    outputs = engine.run_until_done()
    assert outputs["france"].prompt_token_ids == encoded["france"]
    assert outputs["france"].finished


def test_mixed_sampling_params_in_one_batch_leave_greedy_rows_alone(vox_model, encoded, alone):
    """One sampled request beside a greedy one must not disturb the greedy one."""
    generator = torch.Generator().manual_seed(1234)
    cache = PagedKVCache(vox_model.cfg, num_blocks=128, block_size=16, dtype=vox_model.dtype)
    engine = Engine(vox_model, cache, SchedulerConfig(), generator=generator)

    engine.add_request("france", encoded["france"], greedy())
    engine.add_request(
        "count", encoded["count"], SamplingParams(temperature=1.0, max_new_tokens=MAX_NEW_TOKENS)
    )
    outputs = engine.run_until_done()

    assert outputs["france"].token_ids == alone["france"]
    assert outputs["count"].num_output_tokens == MAX_NEW_TOKENS


def test_empty_prompt_is_refused(vox_model):
    engine = engine_for(vox_model)
    with pytest.raises(ValueError, match="empty"):
        engine.add_request("nope", [], greedy())


def test_run_until_done_has_a_guard_against_looping_forever(vox_model, encoded):
    engine = engine_for(vox_model)
    engine.add_request("france", encoded["france"], greedy())
    with pytest.raises(RuntimeError, match="still working after"):
        engine.run_until_done(max_steps=2)


# -- numerics -------------------------------------------------------------

@torch.inference_mode()
def test_a_single_sequence_through_the_batch_path_is_bit_identical(vox_model, encoded):
    """The flattened path's explicit mask costs nothing on its own.

    The batch builder always materialises a mask instead of using SDPA's
    is_causal flag. For one sequence that must still be exactly the stage-1
    answer, which separates "the mask is wrong" from "batching moved the
    arithmetic" in the test below.
    """
    tokens = encoded["france"]
    cache = PagedKVCache(vox_model.cfg, num_blocks=64, block_size=16, dtype=vox_model.dtype)
    s = Sequence(token_ids=list(tokens))
    cache.allocate_for(s, len(tokens))
    batch = build_batch(cache, [s], [len(tokens)], dtype=vox_model.dtype)

    through_batch = vox_model(batch.input_ids, batch.position_ids, state=batch.state)
    reference = vox_model(torch.tensor([tokens], dtype=torch.long))
    assert torch.equal(through_batch, reference)


@torch.inference_mode()
def test_batching_moves_logits_only_within_tolerance(vox_model, encoded):
    """Sharing a forward pass is not bit-exact, and cannot be.

    Attention reduces over every key in the batch, so a sequence batched with
    others is reduced over a wider tensor than it would be alone. The values are
    the same; the accumulation is not. What has to hold is that the gap stays far
    below anything that could flip an argmax -- which is why the gate above is
    stated in tokens, not in logits.
    """
    tokens = encoded["france"]
    cache = PagedKVCache(vox_model.cfg, num_blocks=64, block_size=16, dtype=vox_model.dtype)
    sequences, query_lens = [], []
    for name in ("france", "colour", "count"):
        s = Sequence(token_ids=list(encoded[name]))
        cache.allocate_for(s, len(encoded[name]))
        sequences.append(s)
        query_lens.append(len(encoded[name]))

    batch = build_batch(cache, sequences, query_lens, dtype=vox_model.dtype)
    batched = vox_model(batch.input_ids, batch.position_ids, state=batch.state)
    shared = batched[:, : len(tokens), :]
    reference = vox_model(torch.tensor([tokens], dtype=torch.long))

    deviation = (shared - reference).abs().max().item()
    assert deviation < 1e-3, f"batching moved the logits by {deviation:.3e}"
    assert torch.equal(shared.argmax(-1), reference.argmax(-1))


# -- stage 5: chunked prefill ---------------------------------------------

def test_chunked_prefill_does_not_change_the_output(vox_model, encoded, alone):
    """A prompt split across iterations must produce what it would in one.

    The budget of 3 is well below every prompt here, so each one is prefilled in
    several pieces; the assertion on num_catch_up_chunks is there so this cannot
    quietly pass without chunking having happened.
    """
    engine = engine_for(vox_model, max_num_batched_tokens=3)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()

    assert engine.num_catch_up_chunks > len(encoded), "nothing was actually chunked"
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


def test_a_sequence_mid_prefill_does_not_emit_a_token(vox_model, encoded):
    """Logits partway through a prompt do not predict its next token.

    With a 2-token budget the first iteration computes only part of the prompt,
    so nothing may be sampled from it yet.
    """
    engine = engine_for(vox_model, max_num_batched_tokens=2)
    engine.add_request("france", encoded["france"], greedy())

    first = engine.step()
    assert len(first) == 1
    assert first[0].num_output_tokens == 0

    sequence = engine.scheduler.running[0]
    assert sequence.num_computed_tokens == 2
    assert sequence.needs_prefill


def test_chunking_interleaves_a_long_prompt_with_live_decodes(vox_model, tokenizer, encoded):
    """The reason chunked prefill exists: a big prompt must not stall decoding."""
    long_prompt = tokenizer(
        "You are a helpful assistant. " * 24 + "Name one colour.",
        add_special_tokens=False,
    )["input_ids"]
    assert len(long_prompt) > 64

    engine = engine_for(vox_model, max_num_batched_tokens=16, num_blocks=256)
    engine.add_request("france", encoded["france"], greedy())
    engine.add_request("long", long_prompt, greedy(4))

    saw_both = False
    while engine.has_work:
        scheduled = engine.scheduler.schedule()
        if scheduled.is_empty:
            continue
        if set(scheduled.kinds) == {"decode", "prefill"}:
            saw_both = True
        engine.run_scheduled(scheduled)

    assert saw_both, "a decode never shared an iteration with a prefill chunk"


@pytest.mark.parametrize("block_size", [1, 4])
def test_chunking_and_small_blocks_together(vox_model, encoded, alone, block_size):
    engine = engine_for(
        vox_model, max_num_batched_tokens=5, num_blocks=1024, block_size=block_size
    )
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done()
    for name in encoded:
        assert outputs[name].token_ids == alone[name]


# -- stage 5: preemption --------------------------------------------------

def test_preemption_does_not_change_the_output(vox_model, encoded, alone):
    """THE stage-5 gate: throwing KV away and recomputing it changes nothing.

    Eight blocks of four tokens is 32 tokens of capacity for four requests that
    will each reach about 17, so the pool thrashes. The assertion on
    num_preemptions is what keeps this test honest -- without it, a pool that
    happened to be big enough would pass while testing nothing.
    """
    engine = engine_for(vox_model, num_blocks=8, block_size=4)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done(max_steps=2000)

    assert engine.num_preemptions > 0, "no preemption happened; the pool was too big"
    for name in encoded:
        assert outputs[name].token_ids == alone[name], (
            f"{name} diverged across a preemption\n"
            f"  alone    : {alone[name]}\n"
            f"  preempted: {outputs[name].token_ids}"
        )


def test_preemption_is_reported_per_request(vox_model, encoded):
    engine = engine_for(vox_model, num_blocks=8, block_size=4)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    outputs = engine.run_until_done(max_steps=2000)

    assert sum(o.num_preemptions for o in outputs.values()) == engine.num_preemptions
    assert any(o.num_preemptions > 0 for o in outputs.values())


def test_everything_is_returned_after_preemption_thrashing(vox_model, encoded):
    engine = engine_for(vox_model, num_blocks=8, block_size=4)
    for name, ids in encoded.items():
        engine.add_request(name, ids, greedy())
    engine.run_until_done(max_steps=2000)

    assert engine.num_preemptions > 0
    assert engine.cache.pool.num_free_blocks == engine.cache.num_blocks
    assert all(block.is_free for block in engine.cache.pool.blocks)


def test_a_prompt_too_large_for_the_pool_is_refused(vox_model, encoded):
    # The shortest prompt here is 5 tokens, so 2 blocks of 4; one block is not
    # enough to ever hold it and the request is refused instead of thrashing.
    engine = engine_for(vox_model, num_blocks=1, block_size=4)
    with pytest.raises(ValueError, match="the pool only has 1"):
        engine.add_request("france", encoded["france"], greedy())
