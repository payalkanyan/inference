"""Instrumentation.

Three kinds of record, all written to one JSONL file with a ``type`` field so a
run can be grepped or loaded with one pass:

``request``
    One per finished or cancelled request: queue time, TTFT, total time, tokens
    generated, prefix hit ratio, and whether it was preempted or cancelled.

``iteration``
    One per engine tick: batch size, query tokens, block pool utilisation, and
    how long the forward pass took. This is the "over time" series -- batch size
    and pool occupancy are the two things that explain everything else.

``cancellation``
    One per ``Engine.cancel`` that actually cancelled something. Carries the
    headline number of the project: how many engine ticks and how much wall
    clock passed between asking and the blocks being back in the pool.

Each record defines its own ``to_row()`` rather than leaning on
``dataclasses.asdict``, because several of the interesting fields are derived and
because the row *is* the file format -- it should be visible in one place.

Phase 1 writes the numbers and stops there. No plots, no analysis, no comparison
against anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from vox.sequence import Sequence


@dataclass
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    generated_tokens: int

    #: Seconds spent waiting before this request was first scheduled.
    queue_s: float | None
    #: Seconds from arrival to the first token. None if it never produced one.
    ttft_s: float | None
    #: Seconds from arrival to leaving the engine, however it left.
    total_s: float | None

    ttft_budget_s: float | None
    deadline_missed: bool

    prefix_cached_tokens: int
    prefix_hit_ratio: float
    declared_prefix_len: int
    declared_prefix_hit_ratio: float

    num_preemptions: int
    was_preempted: bool
    was_cancelled: bool
    finish_reason: str | None

    @classmethod
    def from_sequence(cls, sequence: Sequence) -> RequestMetrics:
        queue_s = (
            sequence.admitted_time - sequence.arrival_time
            if sequence.admitted_time is not None
            else None
        )
        total_s = (
            sequence.finished_time - sequence.arrival_time
            if sequence.finished_time is not None
            else None
        )
        return cls(
            request_id=sequence.request_id,
            prompt_tokens=sequence.prompt_len,
            generated_tokens=sequence.num_output_tokens,
            queue_s=queue_s,
            ttft_s=sequence.ttft_s,
            total_s=total_s,
            ttft_budget_s=sequence.ttft_budget_s,
            deadline_missed=sequence.deadline_missed,
            prefix_cached_tokens=sequence.num_cached_tokens,
            prefix_hit_ratio=sequence.prefix_hit_ratio,
            declared_prefix_len=sequence.declared_prefix_len,
            declared_prefix_hit_ratio=sequence.declared_prefix_hit_ratio,
            num_preemptions=sequence.num_preemptions,
            was_preempted=sequence.was_preempted,
            was_cancelled=sequence.finish_reason == "cancelled",
            finish_reason=sequence.finish_reason,
        )

    def to_row(self) -> dict[str, Any]:
        return {"type": "request", **self.__dict__}


@dataclass
class IterationMetrics:
    tick: int
    #: Seconds since the engine was created.
    elapsed_s: float
    #: Seconds spent inside the model forward pass.
    forward_s: float

    batch_size: int
    query_tokens: int
    num_prefill_chunks: int
    num_decodes: int

    blocks_used: int
    blocks_free: int
    block_utilization: float
    cached_blocks: int

    preempted: int
    urgently_admitted: int

    def to_row(self) -> dict[str, Any]:
        return {"type": "iteration", **self.__dict__}


@dataclass
class CancellationRecord:
    """What it cost to abandon one generation. The headline metric.

    ``ticks_to_free`` is the number of engine iterations between ``cancel()``
    being called and the blocks being back in the pool. It is zero by
    construction -- cancellation does not go through the loop -- and the test
    that asserts so is the one claim this project is really making.

    ``blocks_retained`` is the other half: blocks the sequence was using that did
    *not* come back, because something else still references them. Freeing a
    barge-in must not take the shared system prompt with it.
    """

    request_id: str
    tick_at_cancel: int
    tick_at_free: int
    wall_seconds: float
    blocks_held: int
    blocks_released: int
    tokens_generated: int
    had_first_token: bool

    @property
    def ticks_to_free(self) -> int:
        return self.tick_at_free - self.tick_at_cancel

    @property
    def blocks_retained(self) -> int:
        return self.blocks_held - self.blocks_released

    def to_row(self) -> dict[str, Any]:
        return {
            "type": "cancellation",
            **self.__dict__,
            "ticks_to_free": self.ticks_to_free,
            "blocks_retained": self.blocks_retained,
        }


class MetricsRecorder:
    """Collects records and, optionally, appends them to a JSONL file.

    Writing happens as records arrive rather than at the end, so a run that is
    interrupted still leaves usable data. The cost is one ``json.dumps`` and a
    buffered write per record -- against a forward pass, nothing.

    With no ``path`` it only collects in memory, which is what the tests and the
    engine's own default do.
    """

    def __init__(self, path: str | Path | None = None, *, keep_in_memory: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.keep_in_memory = keep_in_memory
        self.requests: list[RequestMetrics] = []
        self.iterations: list[IterationMetrics] = []
        self.cancellations: list[CancellationRecord] = []
        self._file: TextIO | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")

    # -- recording --------------------------------------------------------

    def record_request(self, metrics: RequestMetrics) -> None:
        if self.keep_in_memory:
            self.requests.append(metrics)
        self._write(metrics.to_row())

    def record_iteration(self, metrics: IterationMetrics) -> None:
        if self.keep_in_memory:
            self.iterations.append(metrics)
        self._write(metrics.to_row())

    def record_cancellation(self, record: CancellationRecord) -> None:
        if self.keep_in_memory:
            self.cancellations.append(record)
        self._write(record.to_row())

    def _write(self, row: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(row) + "\n")

    # -- lifecycle --------------------------------------------------------

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> MetricsRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"MetricsRecorder(path={self.path}, {len(self.requests)} requests, "
            f"{len(self.iterations)} iterations, {len(self.cancellations)} cancellations)"
        )
