"""Per-sequence state.

A ``Sequence`` is one in-flight generation: its tokens, where its KV lives, and
the parameters it was admitted with. The scheduler reasons about these.

Request-level fields (``request_id``, ``params``, ``stop_token_ids``) live here
rather than in a separate Request object, because every one of them is per
generation and a second object would only add indirection. Stage 7's Turn is a
*constructor* for one of these, not a parallel type.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from vox.sampler import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"      # admitted, no KV computed yet
    RUNNING = "running"      # in the running batch
    PREEMPTED = "preempted"  # evicted under memory pressure, will be recomputed
    FINISHED = "finished"    # hit a stop token or its token budget
    CANCELLED = "cancelled"  # barge-in: abandoned mid-decode


#: Terminal statuses. Blocks are released on entering any of these.
TERMINAL = frozenset({SequenceStatus.FINISHED, SequenceStatus.CANCELLED})

_next_id = itertools.count().__next__


# eq=False so that `==` and `in` compare identity, not field values. The
# scheduler does `self.running.remove(sequence)`, and with value equality two
# sequences that happened to hold the same tokens would be interchangeable --
# which, for two turns of the same conversation replaying the same prompt, is not
# a hypothetical.
@dataclass(eq=False)
class Sequence:
    """Token ids, how much of them is in the cache, and where that cache lives."""

    token_ids: list[int]

    params: SamplingParams = field(default_factory=SamplingParams)
    #: Caller-supplied identifier. ``seq_id`` is the engine's internal one.
    request_id: str = ""
    stop_token_ids: frozenset[int] = frozenset()

    seq_id: int = field(default_factory=_next_id)

    #: Tokens whose keys and values are written to the cache. Everything from
    #: here to ``num_tokens`` still has to be prefilled. This is what makes
    #: chunked prefill and recompute-based preemption expressible: preemption
    #: just resets it to 0.
    num_computed_tokens: int = 0

    #: Logical block index -> physical block id in the pool.
    block_table: list[int] = field(default_factory=list)

    #: Chained hash of each full block, parallel to the start of block_table.
    #: Carried along so the chain is extended rather than recomputed from
    #: scratch every iteration.
    block_hashes: list[int] = field(default_factory=list)

    #: Prompt tokens whose KV came from the prefix cache instead of a forward
    #: pass. Stage 8 reports this as the hit ratio.
    num_cached_tokens: int = 0

    # -- deadline accounting ------------------------------------------------
    #: Engine clock reading when this request was admitted to the queue.
    arrival_time: float = 0.0
    #: Seconds from arrival within which the first token must appear. None means
    #: no deadline, and such a request yields to those that have one.
    ttft_budget_s: float | None = None
    first_token_time: float | None = None
    #: Set once, when the first token turns out to be late. Recorded rather than
    #: acted on: the request still runs to completion.
    deadline_missed: bool = False

    #: Tokens the caller declared as a reusable prefix via a Turn. 0 when the
    #: request was submitted without one.
    declared_prefix_len: int = 0

    status: SequenceStatus = SequenceStatus.WAITING

    #: "stop" | "length" | "cancelled", set when the sequence becomes terminal.
    finish_reason: str | None = None

    #: How many times this sequence has had its KV thrown away under memory
    #: pressure. Stage 8 reports it per request.
    num_preemptions: int = 0

    #: Length of the original prompt, so output tokens are just the tail. Set
    #: automatically; pass it only when reconstructing a sequence.
    prompt_len: int = -1

    def __post_init__(self) -> None:
        if self.prompt_len < 0:
            self.prompt_len = len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.prompt_len

    @property
    def needs_prefill(self) -> bool:
        """True while any prompt token still has no KV in the cache."""
        return self.num_computed_tokens < self.prompt_len

    @property
    def is_catching_up(self) -> bool:
        """More than one token has no KV yet.

        A first prefill, a later prefill chunk, and a recompute after preemption
        are all the same operation -- compute a run of tokens whose KV is
        missing -- so the scheduler treats them identically and only this
        distinction matters to it.
        """
        return self.num_uncomputed_tokens > 1

    @property
    def work_kind(self) -> str:
        """What this sequence would be doing next: prefill, recompute or decode.

        Only for reporting. "recompute" is a sequence that had been generating,
        was preempted, and is now rebuilding KV for tokens it already produced.
        """
        if self.num_computed_tokens < self.prompt_len:
            return "prefill"
        return "recompute" if self.is_catching_up else "decode"

    @property
    def was_preempted(self) -> bool:
        return self.num_preemptions > 0

    @property
    def prefix_hit_ratio(self) -> float:
        """Share of the prompt that did not have to be prefilled."""
        return self.num_cached_tokens / self.prompt_len if self.prompt_len else 0.0

    @property
    def declared_prefix_hit_ratio(self) -> float:
        """Share of the *declared* prefix that was reused.

        The denominator the caller actually cares about. A turn that declared a
        long reusable prefix and reused little of it is a problem worth seeing,
        and the overall prompt ratio hides it behind the new segment's length.
        """
        if not self.declared_prefix_len:
            return 0.0
        return min(self.num_cached_tokens, self.declared_prefix_len) / self.declared_prefix_len

    # -- deadlines ----------------------------------------------------------

    @property
    def deadline(self) -> float | None:
        if self.ttft_budget_s is None:
            return None
        return self.arrival_time + self.ttft_budget_s

    @property
    def has_first_token(self) -> bool:
        return self.first_token_time is not None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def is_at_risk(self, now: float, margin_s: float) -> bool:
        """Whether this request's first token is in danger of being late.

        Only meaningful before the first token: afterwards the TTFT deadline is
        settled either way. A request already past its deadline still counts as
        at risk, so that it is scheduled rather than quietly abandoned -- the
        engine records the miss, it does not drop the work.
        """
        deadline = self.deadline
        if deadline is None or self.has_first_token:
            return False
        return deadline - now <= margin_s

    @property
    def is_ready_to_sample(self) -> bool:
        """True once every token it holds has been computed.

        A sequence mid-prefill has computed only part of its prompt, and the
        logits at that point do not predict its next token. Stage 4 always
        finishes a prefill in one iteration so this is always true after a
        forward; chunked prefill in stage 5 is where it starts mattering.
        """
        return self.num_computed_tokens >= self.num_tokens

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Throw away the KV but keep the tokens.

        This is what makes preemption cheap to implement: nothing is copied out
        to host memory, the sequence simply forgets that its keys and values were
        ever computed and rebuilds them from tokens it still has. The cost is
        paid in recomputed FLOPs when it resumes, and the caller is responsible
        for having released the blocks first.
        """
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.block_hashes.clear()
        self.num_preemptions += 1
        self.status = SequenceStatus.PREEMPTED

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.value}, "
            f"{self.num_computed_tokens}/{self.num_tokens} computed, "
            f"{len(self.block_table)} blocks)"
        )
