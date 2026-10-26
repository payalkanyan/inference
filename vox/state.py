"""What attention needs to know beyond the hidden states.

``AttentionState`` is built once per forward pass -- all 24 layers read the same
cache bounds, the same mask and the same slot indices, so computing it per layer
would be 24x the work for identical answers.

It lives in its own module so that both the cache implementations and the
attention layer can depend on it without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from vox.cache import KVCache


@dataclass(frozen=True)
class AttentionState:
    cache: KVCache | None = None

    #: Logical sequence bounds: this forward computes tokens [start, end).
    start: int = 0
    end: int = 0

    mask: Tensor | None = None
    is_causal: bool = True

    #: Paged cache only: physical slots to write the new tokens into, and the
    #: slots to read the whole prefix back from. A slot is
    #: ``block_id * block_size + offset_in_block``.
    write_slots: Tensor | None = None
    read_slots: Tensor | None = None

    @classmethod
    def uncached(cls, q_len: int) -> AttentionState:
        """Stage-1 behaviour: the whole sequence in one go, plain causal mask."""
        return cls(None, 0, q_len, None, True)


def build_causal_mask(q_len: int, kv_len: int, *, device, dtype: torch.dtype) -> Tensor:
    """Additive mask for ``q_len`` queries sitting at the end of ``kv_len`` keys.

    Query ``i`` is at sequence position ``kv_len - q_len + i`` and may attend to
    keys ``0..kv_len - q_len + i``.
    """
    offset = kv_len - q_len
    keys = torch.arange(kv_len, device=device)
    queries = torch.arange(q_len, device=device).unsqueeze(1)
    blocked = keys > queries + offset
    return torch.zeros(q_len, kv_len, device=device, dtype=dtype).masked_fill_(
        blocked, float("-inf")
    )


def mask_for(
    start: int, end: int, q_len: int, *, device, dtype: torch.dtype
) -> tuple[Tensor | None, bool]:
    """Decide how to mask ``q_len`` queries ending at ``end`` with ``start`` cached.

    Returns ``(mask, is_causal)``. Three cases, and only one of them needs a
    materialised mask:

    * **prefill from an empty cache** (``start == 0``): ordinary aligned causal
      attention, so ``is_causal=True`` covers it and the arithmetic stays
      bit-identical to the uncached path.
    * **decode** (``q_len == 1``): the single query sits at the end and may attend
      to every cached key, so *no* mask is correct. ``is_causal=True`` would be
      actively wrong -- SDPA aligns its implicit mask to the top left, so one
      query over n keys would be restricted to key 0 alone.
    * **chunk continuation**: queries sit partway through the sequence, which no
      flag expresses, so this is the only case that builds a mask.

    Both cache implementations route through here, which is what guarantees they
    make identical masking decisions.
    """
    if start == 0:
        return None, True
    if q_len == 1:
        return None, False
    return build_causal_mask(q_len, end, device=device, dtype=dtype), False
