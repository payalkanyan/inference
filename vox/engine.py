"""The engine loop: schedule, build tensors, forward, sample, postprocess.

``step()`` is one iteration. Nothing in it is specific to how many sequences are
in flight or what stage they are at -- that is entirely the scheduler's decision
and the batch builder's problem -- so the loop itself stays short enough to read
in one go.

``cancel()`` is the exception, and the reason this project exists. It is not part
of the loop at all: it runs synchronously, outside any iteration, and the blocks
are back in the pool before it returns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from vox.batch import build_batch
from vox.cache import PagedKVCache
from vox.model import Qwen2ForCausalLM
from vox.sampler import SamplingParams, sample
from vox.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from vox.sequence import Sequence, SequenceStatus
from vox.turn import Turn


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None
    #: How many times this request had its KV discarded under memory pressure.
    num_preemptions: int = 0
    #: Prompt tokens served from the prefix cache rather than prefilled.
    num_cached_tokens: int = 0
    prefix_hit_ratio: float = 0.0
    #: Share of a Turn's *declared* prefix that was reused. 0 without a Turn.
    declared_prefix_hit_ratio: float = 0.0

    #: Seconds from admission to the first token, and whether that was late.
    ttft_s: float | None = None
    deadline_missed: bool = False

    @property
    def cancelled(self) -> bool:
        return self.finish_reason == "cancelled"

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids)


def _snapshot(sequence: Sequence) -> RequestOutput:
    return RequestOutput(
        request_id=sequence.request_id,
        prompt_token_ids=sequence.token_ids[: sequence.prompt_len],
        token_ids=list(sequence.output_token_ids),
        finished=sequence.is_terminal,
        finish_reason=sequence.finish_reason,
        num_preemptions=sequence.num_preemptions,
        num_cached_tokens=sequence.num_cached_tokens,
        prefix_hit_ratio=sequence.prefix_hit_ratio,
        declared_prefix_hit_ratio=sequence.declared_prefix_hit_ratio,
        ttft_s=sequence.ttft_s,
        deadline_missed=sequence.deadline_missed,
    )


@dataclass
class CancellationRecord:
    """What it cost to abandon one generation. The headline metric.

    ``ticks_to_free`` is the number of engine iterations that elapsed between
    ``cancel()`` being called and the blocks being back in the pool. It is zero
    by construction -- cancellation does not go through the loop -- and the test
    that asserts so is the one claim this project is really making.

    ``blocks_retained`` is the other half: blocks the sequence was using that did
    *not* come back, because another turn still references them. Freeing a
    barge-in must not take the system prompt with it.
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


class Engine:
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        cache: PagedKVCache,
        config: SchedulerConfig | None = None,
        *,
        generator: torch.Generator | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.cache = cache
        self.clock = clock
        self.scheduler = Scheduler(cache, config, clock=clock)
        self.generator = generator

        #: Every request the engine has seen, by id, so that cancel() can find
        #: one and so stage 8 can report on requests that have already left.
        self.requests: dict[str, Sequence] = {}
        self.cancellations: list[CancellationRecord] = []
        #: Prefix tokens -> a sequence that exists only to hold references.
        self._retained: dict[tuple[int, ...], Sequence] = {}

        # Cheap running counters. Stage 8 builds proper per-request
        # instrumentation on top; these exist so tests can assert that a
        # mechanism actually engaged rather than assuming it did.
        self.num_iterations = 0
        self.num_preemptions = 0
        self.num_catch_up_chunks = 0
        self.num_blocks_published = 0
        self.num_cached_tokens_reused = 0
        self.num_deadline_misses = 0
        self.num_urgent_admissions = 0

    # -- submitting work ---------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        params: SamplingParams | None = None,
        stop_token_ids: frozenset[int] | set[int] | list[int] = (),
        *,
        ttft_budget_s: float | None = None,
        declared_prefix_len: int = 0,
    ) -> Sequence:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids is empty")
        if request_id in self.requests and not self.requests[request_id].is_terminal:
            raise ValueError(f"request {request_id!r} is already in flight")

        sequence = Sequence(
            token_ids=list(prompt_token_ids),
            params=params or SamplingParams(),
            request_id=request_id,
            stop_token_ids=frozenset(stop_token_ids),
            arrival_time=self.clock(),
            ttft_budget_s=ttft_budget_s,
            declared_prefix_len=declared_prefix_len,
        )
        self.requests[request_id] = sequence
        self.scheduler.add(sequence)
        return sequence

    def add_turn(self, turn: Turn) -> Sequence:
        """Submit a conversation turn, declaring which part of it is reusable."""
        return self.add_request(
            turn.request_id,
            turn.token_ids,
            turn.params,
            turn.stop_token_ids,
            ttft_budget_s=turn.ttft_budget_s,
            declared_prefix_len=turn.prefix_len,
        )

    # -- barge-in ----------------------------------------------------------

    def cancel(self, request_id: str) -> CancellationRecord | None:
        """Abandon a generation and reclaim its memory immediately.

        Synchronous on purpose. Nothing is queued, no flag is set for the next
        scheduling pass to notice, and no iteration runs in between: by the time
        this returns, the blocks are in the pool and a new turn admitted on the
        same tick can use them.

        Returns None if the request has already finished or was already
        cancelled, because a barge-in racing a completion is normal and not an
        error. An id that was never submitted is a KeyError -- that is a bug.
        """
        sequence = self.requests[request_id]
        if sequence.is_terminal:
            return None

        started = self.clock()
        tick = self.num_iterations
        blocks_held = len(sequence.block_table)

        released = self.scheduler.cancel(sequence)

        record = CancellationRecord(
            request_id=request_id,
            tick_at_cancel=tick,
            tick_at_free=self.num_iterations,
            wall_seconds=self.clock() - started,
            blocks_held=blocks_held,
            blocks_released=len(released),
            tokens_generated=sequence.num_output_tokens,
            had_first_token=sequence.has_first_token,
        )
        self.cancellations.append(record)
        return record

    # -- keeping a declared prefix resident --------------------------------

    def retain_prefix(self, token_ids: list[int]) -> int:
        """Pin the cached blocks covering ``token_ids`` against eviction.

        A retention is just a sequence that holds references and never runs, so
        it needs no new machinery -- the refcount already means "somebody is
        using this". Without it, a system prompt can be evicted while other
        conversations churn through the pool, and the next turn pays full price
        for a prefix it declared as reusable.

        Returns how many tokens are now pinned; 0 if none of it was cached yet.
        """
        key = tuple(token_ids)
        existing = self._retained.get(key)
        if existing is not None:
            return existing.num_computed_tokens

        holder = Sequence(token_ids=list(token_ids))
        pinned = self.cache.attach_cached_prefix(holder, leave_one=False)
        if pinned == 0:
            return 0
        self._retained[key] = holder
        return pinned

    def release_prefix(self, token_ids: list[int]) -> None:
        holder = self._retained.pop(tuple(token_ids), None)
        if holder is not None:
            self.cache.free(holder)

    @property
    def num_retained_blocks(self) -> int:
        return sum(len(h.block_table) for h in self._retained.values())

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    # -- one iteration -----------------------------------------------------

    def step(self) -> list[RequestOutput]:
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty:
            return []
        return self.run_scheduled(scheduled)

    def run_scheduled(self, scheduled: SchedulerOutput) -> list[RequestOutput]:
        """Execute a decision the scheduler has already made.

        Split out of ``step`` so a caller that wants to inspect or log the batch
        composition can do so without re-scheduling, which would double-allocate.
        """
        self.num_iterations += 1
        self.num_preemptions += len(scheduled.preempted)
        self.num_catch_up_chunks += scheduled.num_prefills
        self.num_cached_tokens_reused += sum(
            s.num_cached_tokens for s in scheduled.newly_admitted
        )
        self.num_urgent_admissions += len(scheduled.admitted_urgently)

        batch = build_batch(
            self.cache,
            scheduled.sequences,
            scheduled.query_lens,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        with torch.inference_mode():
            logits = self.model(batch.input_ids, batch.position_ids, state=batch.state)

        # The KV for these tokens is now in the cache.
        for sequence, query_len in zip(scheduled.sequences, scheduled.query_lens):
            sequence.num_computed_tokens += query_len
            # Any block that just became full is now a prefix somebody else can
            # adopt. Has to happen after num_computed_tokens moves, since that
            # is what says the block's contents are final.
            self.num_blocks_published += len(self.cache.register_computed_blocks(sequence))

        self._sample_and_postprocess(batch, logits)
        return [_snapshot(s) for s in scheduled.sequences]

    def _sample_and_postprocess(self, batch, logits: torch.Tensor) -> None:
        # Only sequences whose every token is now computed get to sample. A
        # sequence still mid-prefill has logits that do not predict its next
        # token. Always true in stage 4; chunked prefill makes it matter.
        ready = [i for i, s in enumerate(batch.sequences) if s.is_ready_to_sample]
        if not ready:
            return

        rows = logits[0, batch.logit_indices[ready], :]
        params = [batch.sequences[i].params for i in ready]
        next_ids = sample(rows, params, self.generator).tolist()

        now = self.clock()
        for index, token_id in zip(ready, next_ids):
            sequence = batch.sequences[index]
            sequence.append_token(token_id)

            if not sequence.has_first_token:
                sequence.first_token_time = now
                deadline = sequence.deadline
                if deadline is not None and now > deadline:
                    # Recorded, not acted on: the request still runs to the end.
                    sequence.deadline_missed = True
                    self.num_deadline_misses += 1

            if token_id in sequence.stop_token_ids:
                self.scheduler.finish(sequence, "stop")
            elif sequence.num_output_tokens >= sequence.params.max_new_tokens:
                self.scheduler.finish(sequence, "length")

    # -- driving it to completion -----------------------------------------

    def run_until_done(self, max_steps: int | None = None) -> dict[str, RequestOutput]:
        """Step until nothing is left. Returns the final state of every request."""
        outputs: dict[str, RequestOutput] = {}
        steps = 0
        while self.has_work:
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError(f"still working after {max_steps} steps")
            for output in self.step():
                outputs[output.request_id] = output
            steps += 1
        return outputs
