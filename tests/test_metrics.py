"""Stage 8: instrumentation.

Two things to get right. The numbers have to be the numbers -- a queue time that
silently measures something else is worse than no queue time -- so the timing
tests drive an injected clock and assert exact values. And the file has to be
readable, so the JSONL is parsed back rather than merely written.
"""

from __future__ import annotations

import json

import pytest
import torch

from vox.cache import PagedKVCache
from vox.engine import Engine
from vox.hub import resolve_model_dir
from vox.metrics import MetricsRecorder, RequestMetrics
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.sequence import Sequence
from vox.turn import Turn
from vox.weights import load_model


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- field mapping, no model ---------------------------------------------

def test_timings_are_differences_from_arrival():
    s = Sequence(token_ids=[1, 2, 3, 4], arrival_time=100.0)
    s.admitted_time = 100.25
    s.first_token_time = 100.75
    s.finished_time = 103.0
    s.append_token(9)

    m = RequestMetrics.from_sequence(s)
    assert m.queue_s == pytest.approx(0.25)
    assert m.ttft_s == pytest.approx(0.75)
    assert m.total_s == pytest.approx(3.0)
    assert m.prompt_tokens == 4
    assert m.generated_tokens == 1


def test_timings_are_none_rather_than_zero_when_they_never_happened():
    """A request cancelled during prefill has no TTFT. Reporting 0.0 would be a
    lie that averages badly."""
    s = Sequence(token_ids=[1, 2, 3], arrival_time=100.0)
    m = RequestMetrics.from_sequence(s)
    assert m.queue_s is None
    assert m.ttft_s is None
    assert m.total_s is None


def test_queue_time_is_not_restarted_by_a_preemption():
    """`admitted_time` is the *first* admission, so queue time keeps meaning
    "how long before work started on this request"."""
    s = Sequence(token_ids=[1, 2, 3], arrival_time=100.0)
    s.admitted_time = 100.5
    s.reset_for_recompute()
    assert RequestMetrics.from_sequence(s).queue_s == pytest.approx(0.5)


def test_cancellation_is_distinguishable_from_completion():
    s = Sequence(token_ids=[1, 2, 3], arrival_time=0.0)
    s.finish_reason = "cancelled"
    assert RequestMetrics.from_sequence(s).was_cancelled
    s.finish_reason = "length"
    assert not RequestMetrics.from_sequence(s).was_cancelled


def test_preemption_is_reported():
    s = Sequence(token_ids=[1, 2, 3], arrival_time=0.0)
    assert not RequestMetrics.from_sequence(s).was_preempted
    s.reset_for_recompute()
    s.reset_for_recompute()
    m = RequestMetrics.from_sequence(s)
    assert m.was_preempted and m.num_preemptions == 2


# -- the recorder ---------------------------------------------------------

def test_a_recorder_without_a_path_only_collects():
    recorder = MetricsRecorder()
    recorder.record_request(RequestMetrics.from_sequence(Sequence(token_ids=[1])))
    assert len(recorder.requests) == 1
    assert recorder.path is None


def test_rows_are_written_as_they_arrive(tmp_path):
    """So an interrupted run still leaves usable data."""
    path = tmp_path / "run.jsonl"
    recorder = MetricsRecorder(path)
    recorder.record_request(RequestMetrics.from_sequence(Sequence(token_ids=[1, 2])))
    recorder.flush()

    assert path.exists()
    assert len(path.read_text().strip().splitlines()) == 1
    recorder.close()


def test_in_memory_retention_can_be_turned_off(tmp_path):
    recorder = MetricsRecorder(tmp_path / "run.jsonl", keep_in_memory=False)
    recorder.record_request(RequestMetrics.from_sequence(Sequence(token_ids=[1])))
    recorder.close()
    assert recorder.requests == []
    assert (tmp_path / "run.jsonl").read_text().strip()


def test_the_recorder_closes_itself(tmp_path):
    path = tmp_path / "run.jsonl"
    with MetricsRecorder(path) as recorder:
        recorder.record_request(RequestMetrics.from_sequence(Sequence(token_ids=[1])))
    assert recorder._file is None


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


@pytest.fixture(scope="module")
def prefix(tokenizer):
    text = (
        "You are a concise voice assistant for a scheduling service. Answer in "
        "one short sentence. You have a calendar tool and a weather tool. Never "
        "invent facts. User question: "
    )
    return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])


def a_turn(tokenizer, prefix, request_id, text, **kwargs) -> Turn:
    return Turn(
        request_id=request_id,
        prefix_token_ids=prefix,
        segment_token_ids=tuple(tokenizer(text, add_special_tokens=False)["input_ids"]),
        params=SamplingParams(temperature=0.0, max_new_tokens=6),
        **kwargs,
    )


def engine_for(vox_model, clock=None, metrics=None, **cfg) -> Engine:
    cache = PagedKVCache(vox_model.cfg, num_blocks=256, block_size=16, dtype=vox_model.dtype)
    kwargs = {} if clock is None else {"clock": clock}
    return Engine(vox_model, cache, SchedulerConfig(**cfg), metrics=metrics, **kwargs)


def test_a_record_is_emitted_for_every_finished_request(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    for name in ("a", "b"):
        engine.add_turn(a_turn(tokenizer, prefix, name, "Name a colour."))
    engine.run_until_done(max_steps=300)

    assert {m.request_id for m in engine.metrics.requests} == {"a", "b"}
    for m in engine.metrics.requests:
        assert m.finish_reason == "length"
        assert m.generated_tokens == 6
        assert m.ttft_s is not None and m.total_s is not None
        assert m.total_s >= m.ttft_s


def test_a_cancelled_request_gets_a_record_too(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour."))
    engine.step()
    engine.cancel("a")

    assert len(engine.metrics.requests) == 1
    record = engine.metrics.requests[0]
    assert record.was_cancelled and record.finish_reason == "cancelled"
    assert len(engine.metrics.cancellations) == 1
    assert engine.metrics.cancellations[0].ticks_to_free == 0


def test_one_iteration_record_per_tick(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour."))
    engine.run_until_done(max_steps=300)

    assert len(engine.metrics.iterations) == engine.num_iterations
    assert [m.tick for m in engine.metrics.iterations] == list(range(1, engine.num_iterations + 1))


def test_iteration_records_carry_the_two_time_series(vox_model, tokenizer, prefix):
    """Batch size and pool occupancy are what explain everything else."""
    engine = engine_for(vox_model)
    for name in ("a", "b", "c"):
        engine.add_turn(a_turn(tokenizer, prefix, name, "Name a colour."))
    engine.run_until_done(max_steps=300)

    series = engine.metrics.iterations
    assert max(m.batch_size for m in series) == 3
    assert all(0.0 <= m.block_utilization <= 1.0 for m in series)
    assert all(m.blocks_used + m.blocks_free == engine.cache.num_blocks for m in series)
    assert all(m.forward_s > 0 for m in series)
    assert all(m.query_tokens >= m.batch_size for m in series)
    assert all(m.num_prefill_chunks + m.num_decodes == m.batch_size for m in series)


def test_prefix_reuse_is_reported_per_request(vox_model, tokenizer, prefix):
    engine = engine_for(vox_model)
    engine.add_turn(a_turn(tokenizer, prefix, "first", "What is the capital of France?"))
    engine.run_until_done(max_steps=300)
    engine.add_turn(a_turn(tokenizer, prefix, "second", "And of Italy?"))
    engine.run_until_done(max_steps=300)

    by_id = {m.request_id: m for m in engine.metrics.requests}
    assert by_id["first"].prefix_cached_tokens == 0
    assert by_id["second"].prefix_cached_tokens > 0
    assert by_id["second"].declared_prefix_hit_ratio > 0.8
    assert by_id["second"].declared_prefix_len == len(prefix)


def test_deadline_fields_round_trip(vox_model, tokenizer, prefix):
    clock = FakeClock()
    engine = engine_for(vox_model, clock)
    engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour.", ttft_budget_s=0.2))
    clock.advance(1.0)
    engine.run_until_done(max_steps=300)

    record = engine.metrics.requests[0]
    assert record.ttft_budget_s == 0.2
    assert record.deadline_missed
    assert record.ttft_s == pytest.approx(1.0)


# -- the file -------------------------------------------------------------

def test_the_jsonl_file_parses_and_holds_all_three_record_types(
    vox_model, tokenizer, prefix, tmp_path
):
    path = tmp_path / "voice.jsonl"
    with MetricsRecorder(path) as recorder:
        engine = engine_for(vox_model, metrics=recorder)
        engine.add_turn(a_turn(tokenizer, prefix, "doomed", "Name a colour."))
        engine.step()
        engine.cancel("doomed")
        engine.add_turn(a_turn(tokenizer, prefix, "done", "And of Italy?"))
        engine.run_until_done(max_steps=300)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = {row["type"] for row in rows}
    assert kinds == {"request", "iteration", "cancellation"}

    cancellation = next(r for r in rows if r["type"] == "cancellation")
    assert cancellation["ticks_to_free"] == 0
    assert "blocks_retained" in cancellation

    requests = [r for r in rows if r["type"] == "request"]
    assert {r["request_id"] for r in requests} == {"doomed", "done"}
    assert next(r for r in requests if r["request_id"] == "doomed")["was_cancelled"]

    iterations = [r for r in rows if r["type"] == "iteration"]
    assert len(iterations) == engine.num_iterations
    assert all("block_utilization" in r for r in iterations)


def test_every_row_is_one_line_of_valid_json(vox_model, tokenizer, prefix, tmp_path):
    """Line-delimited, so a run can be streamed rather than loaded whole."""
    path = tmp_path / "run.jsonl"
    with MetricsRecorder(path) as recorder:
        engine = engine_for(vox_model, metrics=recorder)
        engine.add_turn(a_turn(tokenizer, prefix, "a", "Name a colour."))
        engine.run_until_done(max_steps=300)

    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert "\n" not in line
        assert isinstance(row["type"], str)
