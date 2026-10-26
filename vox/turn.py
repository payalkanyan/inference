"""A conversation turn: a declared reusable prefix plus a short new segment.

Prefix caching already discovers sharing by hashing, so why declare it?

Because hashing can only tell you that two prompts *happened* to agree. It cannot
tell you that they were *meant* to. For voice traffic that difference is
operational:

* **A miss becomes visible.** A turn that declares a 2000-token prefix and gets
  no reuse has a problem -- the pool is too small, or the blocks were evicted
  between turns. Without the declaration there is nothing to compare the hit
  against, and the engine silently does the expensive thing.
* **The prefix can be kept resident.** ``Engine.retain_prefix`` pins the blocks
  covering a declared prefix so eviction cannot take them while other
  conversations churn through memory. Discovered sharing has no handle to pin.
* **Reuse stops at the boundary.** The lookup is bounded by the declared prefix
  rather than speculating into the new user segment, which by construction is the
  part that differs.

A Turn is a constructor for a Sequence, not a parallel runtime type. Nothing in
the scheduler knows about turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vox.sampler import SamplingParams


@dataclass(frozen=True)
class Turn:
    request_id: str

    #: System prompt, tool definitions, conversation history -- the part that is
    #: the same as last turn.
    prefix_token_ids: tuple[int, ...]

    #: The new user utterance, plus whatever generation prompt follows it.
    segment_token_ids: tuple[int, ...]

    params: SamplingParams = field(default_factory=SamplingParams)
    stop_token_ids: frozenset[int] = frozenset()

    #: Seconds from arrival within which the first token must be produced.
    #: ``None`` means this turn has no deadline and yields to those that do.
    ttft_budget_s: float | None = None

    def __post_init__(self) -> None:
        if not self.segment_token_ids:
            raise ValueError("a turn needs a new segment; a bare prefix has nothing to answer")

    @property
    def token_ids(self) -> list[int]:
        return [*self.prefix_token_ids, *self.segment_token_ids]

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prefix_token_ids) + len(self.segment_token_ids)

    def __repr__(self) -> str:
        return (
            f"Turn({self.request_id!r}, prefix={self.prefix_len}, "
            f"segment={len(self.segment_token_ids)}, ttft_budget={self.ttft_budget_s})"
        )
