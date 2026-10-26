"""Iteration-level scheduling, chunked prefill, and recompute-based preemption.

Every iteration the scheduler decides which sequences run and how many tokens
each contributes, under a token budget. Requests join between iterations and
finished ones leave, so the batch is rebuilt from scratch each time rather than
having to drain before the next one starts -- that is what "continuous" means.

Three policies, in the order they matter:

**Decodes are reserved for first.** A decoding sequence contributes exactly one
token, so the whole running set costs at most ``max_num_seqs`` tokens of the
budget. Admitting prefills first would let one long prompt consume an entire
iteration and stall every conversation already in flight.

**Prefills are chunked.** Whatever budget is left is spent advancing sequences
that are catching up -- a first prefill, a later chunk of one, or a recompute
after preemption, which are all the same operation. A long prompt therefore
overlaps with everyone else's decoding instead of blocking it for an iteration,
which is the entire point for a workload with a long shared prefix and hard TTFT
deadlines.

**A prefix somebody already computed is adopted, not recomputed.** Admission
consults the prefix cache first, which for voice traffic -- where every turn
re-sends the same system prompt and history -- is usually most of the prompt.

**Deadlines outrank throughput.** Each request may carry a TTFT budget. Waiting
requests are ordered by deadline rather than by arrival, and a request whose
deadline is close enough to be in danger is admitted *ahead of the decode pass*,
which is the only place in this scheduler where latency is deliberately bought at
the cost of throughput. Decodes keep a reservation of one token each so they can
never be starved outright -- an urgent new turn should not freeze the
conversations already in progress.

A deadline that has already passed still counts as urgent. The engine records the
miss and runs the request anyway; shedding doomed work would be defensible for a
voice agent ("late is useless") but it is a product decision, not this layer's.

**Preemption covers what chunking cannot.** Because a chunk can shrink to fit
whatever memory is free, a sequence that is catching up can nearly always make
*some* progress. A decode cannot: its minimum unit is one token, and if the block
for it does not exist, something has to give. So before anything is scheduled,
the pre-pass preempts the most recently admitted sequences until every running
sequence can advance by at least one token. Preemption is recompute-based -- the
blocks are released and the KV is simply forgotten, to be rebuilt from tokens the
sequence still holds. Nothing is copied anywhere.

Evicting the *newest* is what makes progress guaranteed: the oldest running
sequence is never the victim, so it always advances and eventually finishes,
which frees memory for everybody behind it.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from vox.blocks import BlockPoolExhausted
from vox.cache import PagedKVCache
from vox.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    #: Most sequences in one batch.
    max_num_seqs: int = 16
    #: Most query tokens in one iteration, across the whole batch. This also
    #: bounds a prefill chunk, and with it the size of the T x K attention mask.
    max_num_batched_tokens: int = 2048
    #: A waiting request whose deadline is this close is admitted ahead of the
    #: decode pass. A flat margin rather than an estimate of prefill time:
    #: estimating that properly needs a calibrated token rate, and measuring one
    #: is explicitly out of scope for phase 1.
    at_risk_margin_s: float = 0.15


@dataclass
class SchedulerOutput:
    sequences: list[Sequence] = field(default_factory=list)
    #: How many tokens each sequence contributes this iteration.
    query_lens: list[int] = field(default_factory=list)
    #: "prefill" | "recompute" | "decode", parallel to ``sequences``. Reporting only.
    kinds: list[str] = field(default_factory=list)
    #: Sequences admitted from the waiting queue this iteration.
    newly_admitted: list[Sequence] = field(default_factory=list)
    #: Sequences whose KV was discarded this iteration to make room.
    preempted: list[Sequence] = field(default_factory=list)
    #: Sequences admitted because their deadline was in danger, ahead of decodes.
    admitted_urgently: list[Sequence] = field(default_factory=list)

    def add(self, sequence: Sequence, query_len: int) -> None:
        self.kinds.append(sequence.work_kind)
        self.sequences.append(sequence)
        self.query_lens.append(query_len)

    @property
    def is_empty(self) -> bool:
        return not self.sequences

    @property
    def num_query_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_prefills(self) -> int:
        return sum(1 for kind in self.kinds if kind != "decode")


class Scheduler:
    def __init__(
        self,
        cache: PagedKVCache,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache = cache
        self.config = config or SchedulerConfig()
        self.clock = clock
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    # -- queue management -------------------------------------------------

    def add(self, sequence: Sequence) -> None:
        """Admit a request to the waiting queue.

        A prompt whose KV could not fit in the pool even with the pool empty is
        rejected here. Letting it in would mean chunk-prefilling it until memory
        ran out and then preempting it forever, so it is better to say so now.
        """
        blocks = self.cache.pool.blocks_needed(sequence.num_tokens)
        if blocks > self.cache.num_blocks:
            raise ValueError(
                f"prompt of {sequence.num_tokens} tokens needs {blocks} blocks; "
                f"the pool only has {self.cache.num_blocks}"
            )
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def finish(self, sequence: Sequence, reason: str) -> None:
        """Retire a sequence and give its blocks back immediately."""
        sequence.status = SequenceStatus.FINISHED
        sequence.finish_reason = reason
        if sequence in self.running:
            self.running.remove(sequence)
        self.cache.free(sequence)

    def cancel(self, sequence: Sequence) -> list[int]:
        """Stop a sequence and release its memory now, not at a boundary.

        Called straight through from ``Engine.cancel`` with no iteration in
        between, which is the entire point: a barge-in must not have to wait for
        the current decode to finish or for a scheduling pass to notice.

        Returns the blocks that actually went back to the pool. Blocks still
        referenced by somebody else -- a system prompt shared with the next turn
        -- are not among them, and that is the refcount doing its job.
        """
        sequence.status = SequenceStatus.CANCELLED
        sequence.finish_reason = "cancelled"
        if sequence in self.running:
            self.running.remove(sequence)
        if sequence in self.waiting:
            self.waiting.remove(sequence)
        return self.cache.free(sequence)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # -- ordering ----------------------------------------------------------

    def _waiting_in_priority_order(self) -> list[Sequence]:
        """Waiting requests, most deserving first.

        Three tiers. Sequences that have already produced a token and were
        preempted come first: they are committed work whose memory was taken
        away, and resuming them is what keeps the progress guarantee from stage
        5 intact. Then requests with a deadline, earliest first. Then everything
        else, by arrival.
        """

        def key(sequence: Sequence) -> tuple[int, float, float]:
            resuming = 0 if sequence.has_first_token else 1
            deadline = sequence.deadline
            return (resuming, deadline if deadline is not None else math.inf,
                    sequence.arrival_time)

        return sorted(self.waiting, key=key)

    # -- memory -----------------------------------------------------------

    def _needs_new_block(self, sequence: Sequence) -> bool:
        """Whether advancing by one token would require another block."""
        wanted = self.cache.pool.blocks_needed(sequence.num_computed_tokens + 1)
        return wanted > len(sequence.block_table)

    def _fit_chunk(self, sequence: Sequence, budget: int, reserved_blocks: int = 0) -> int:
        """Largest number of tokens ``sequence`` can advance by right now.

        Bounded by what it has left to compute, by the iteration's token budget,
        and by the memory actually available. The last bound is what lets a
        prefill shrink instead of being preempted.

        ``reserved_blocks`` are blocks already promised to running sequences that
        have not been scheduled yet. ``_make_room`` guarantees every running
        sequence can advance one token, and an admission happening before the
        decode pass must not spend that guarantee -- otherwise the decode it was
        reserved for fails to allocate.
        """
        want = min(sequence.num_uncomputed_tokens, budget)
        if want <= 0:
            return 0
        block_size = self.cache.block_size
        spare_blocks = max(0, self.cache.pool.num_free_blocks - reserved_blocks)
        room = (
            len(sequence.block_table) * block_size
            + spare_blocks * block_size
            - sequence.num_computed_tokens
        )
        return max(0, min(want, room))

    def _blocks_promised_to_decodes(self, out: SchedulerOutput) -> int:
        """Blocks running sequences still need in order to advance one token."""
        return sum(
            1
            for sequence in self.running
            if sequence not in out.sequences
            and not sequence.is_catching_up
            and self._needs_new_block(sequence)
        )

    def _preempt(self, sequence: Sequence) -> None:
        self.cache.free(sequence)
        self.running.remove(sequence)
        sequence.reset_for_recompute()
        #: To the front: it has already waited once and holds no memory now.
        self.waiting.appendleft(sequence)

    def _make_room(self) -> list[Sequence]:
        """Preempt until every running sequence can advance by one token."""
        preempted: list[Sequence] = []
        while self.running:
            needed = sum(1 for s in self.running if self._needs_new_block(s))
            if needed <= self.cache.pool.num_free_blocks:
                break
            if len(self.running) == 1:
                # Nobody left to evict. This sequence's context has outgrown the
                # pool, and preempting it would only livelock on the recompute.
                lone = self.running[0]
                raise BlockPoolExhausted(
                    f"sequence {lone.seq_id} has reached {lone.num_tokens} tokens, "
                    f"which no longer fits in {self.cache.num_blocks} blocks of "
                    f"{self.cache.block_size}"
                )
            victim = self.running[-1]       # newest first: the oldest always progresses
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    # -- the decision ------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        now = self.clock()
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        out.preempted = self._make_room()

        # 1. Requests in danger of missing their deadline, ahead of the decode
        #    pass. Decodes keep a reservation of one token each, so buying
        #    latency here can never freeze a conversation already in progress.
        decoders = sum(1 for s in self.running if not s.is_catching_up)
        reserved = min(decoders, self.config.max_num_seqs)
        budget = self._admit(out, budget, now, reserve=reserved, urgent_only=True)

        # 2. Decodes. One token each, so this can never crowd out everything else.
        for sequence in list(self.running):
            # The urgent pass above has already put its admissions into `running`,
            # and a sequence must contribute exactly one slice per iteration:
            # scheduling it twice would advance num_computed_tokens twice for a
            # single forward pass.
            if sequence in out.sequences:
                continue
            if sequence.is_catching_up:
                continue
            if len(out.sequences) >= self.config.max_num_seqs or budget < 1:
                break
            self.cache.allocate_for(sequence, 1)
            out.add(sequence, 1)
            budget -= 1

        # 3. Sequences catching up: prefill chunks and post-preemption recomputes.
        for sequence in list(self.running):
            if sequence in out.sequences:
                continue
            if not sequence.is_catching_up:
                continue
            if len(out.sequences) >= self.config.max_num_seqs or budget < 1:
                break
            chunk = self._fit_chunk(sequence, budget)
            if chunk == 0:
                continue
            self.cache.allocate_for(sequence, chunk)
            out.add(sequence, chunk)
            budget -= chunk

        # 4. Everything else still waiting, with whatever budget is left.
        budget = self._admit(out, budget, now, reserve=0, urgent_only=False)
        return out

    def _deserves_priority(self, sequence: Sequence, now: float) -> bool:
        """Whether this request is admitted ahead of the decode pass.

        Two kinds qualify. A request whose TTFT deadline is in danger, which is
        latency deliberately bought at throughput's expense. And a sequence that
        was preempted after already producing a token: that is committed work
        whose memory was taken away, and leaving it queued behind fresh arrivals
        would reopen precisely the starvation that newest-first eviction exists
        to prevent. Its TTFT is already settled, so it is not "at risk" -- it is
        owed.
        """
        return sequence.has_first_token or sequence.is_at_risk(
            now, self.config.at_risk_margin_s
        )

    def _admit(
        self,
        out: SchedulerOutput,
        budget: int,
        now: float,
        *,
        reserve: int,
        urgent_only: bool,
    ) -> int:
        """Admit waiting requests in priority order. Returns the budget left."""
        reserved_blocks = self._blocks_promised_to_decodes(out)
        for candidate in self._waiting_in_priority_order():
            if len(out.sequences) >= self.config.max_num_seqs:
                break
            if budget - reserve < 1:
                break
            # Something evicted a moment ago would only be evicted again.
            if candidate in out.preempted:
                continue
            if urgent_only and not self._deserves_priority(candidate, now):
                continue

            # Adopt whatever of this prompt somebody has already computed. This
            # happens before sizing the chunk, because a hit shrinks the work
            # left to do and can turn a request that would not have fitted into
            # one that does.
            reused = self.cache.attach_cached_prefix(candidate)

            chunk = self._fit_chunk(candidate, budget - reserve, reserved_blocks)
            if chunk == 0:
                # Not admissible after all; do not sit on the blocks we just
                # adopted, so that only running sequences ever hold memory.
                if reused:
                    self.cache.detach_prefix(candidate)
                continue
            self.cache.allocate_for(candidate, chunk)
            self.waiting.remove(candidate)
            candidate.status = SequenceStatus.RUNNING
            self.running.append(candidate)
            out.add(candidate, chunk)
            out.newly_admitted.append(candidate)
            if urgent_only:
                out.admitted_urgently.append(candidate)
            budget -= chunk

        return budget
