"""Turning a set of scheduled sequences into one forward pass.

Sequences in a batch disagree about everything that matters: how many tokens they
contribute this iteration, how long their prefix already is, and which blocks hold
it. There are two usual ways to reconcile that.

**Padding** gives every sequence the same shape and masks the slack. It is simple,
but the cost is the *maximum* rather than the sum: one 500-token prefill sharing a
batch with eight 1-token decodes would compute 4500 query positions to produce 9
tokens.

**Flattening**, used here, concatenates instead. Queries become one row of
``T = sum(query_len)`` tokens, keys become one run of ``K = sum(kv_len)``, and
sequences are kept apart by the mask rather than by the shape. Nothing is
computed that is not wanted.

The whole separation is this::

    allowed[t, k] = (seq_of_query[t] == seq_of_key[k]) & (key_pos[k] <= query_pos[t])

one term for "same sequence", one for causality. That covers prefill, decode,
mixed batches and chunk continuation without special cases, which is why neither
``layers.py`` nor ``cache.py`` needed changing for this stage: the tensor batch
dimension stays 1, so batching happens along the token dimension, and the paged
cache already addresses memory by slot rather than by offset.

The cost of flattening without custom kernels is that the ``T x K`` mask is
materialised. A varlen attention kernel would read the sequence boundaries
directly and never build it. That is a deliberate phase-1 trade: see the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vox.cache import PagedKVCache
from vox.sequence import Sequence
from vox.state import AttentionState


@dataclass
class BatchInput:
    """Flattened tensors for one forward pass, plus how to read the result back."""

    input_ids: Tensor        # [1, T]
    position_ids: Tensor     # [1, T]
    state: AttentionState    # write_slots [T], read_slots [K], mask [T, K]

    #: Index into the T axis of each sequence's final query position -- the row of
    #: the logits that predicts that sequence's next token.
    logit_indices: Tensor    # [B]

    sequences: list[Sequence]
    query_lens: list[int]

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def num_query_tokens(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def num_key_tokens(self) -> int:
        return int(self.state.read_slots.shape[0])


def build_ragged_mask(
    query_seq: Tensor,
    query_pos: Tensor,
    key_seq: Tensor,
    key_pos: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Additive [T, K] mask: same-sequence AND causal.

    ``query_pos`` and ``key_pos`` are positions within each token's own sequence,
    so causality is a comparison of sequence-local positions and needs no
    knowledge of where either token landed in the flattened batch.
    """
    same_sequence = query_seq.unsqueeze(1) == key_seq.unsqueeze(0)
    causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
    allowed = same_sequence & causal
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill_(
        ~allowed, float("-inf")
    )


def build_batch(
    cache: PagedKVCache,
    sequences: list[Sequence],
    query_lens: list[int],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BatchInput:
    """Flatten ``sequences`` into one forward pass.

    ``query_lens[i]`` is how many of sequence ``i``'s uncomputed tokens to compute
    now: its whole remaining prompt for a prefill, 1 for a decode, a chunk of it
    once stage 5 arrives. Blocks must already be allocated -- the scheduler does
    that, because it is the thing that can choose to admit fewer sequences.
    """
    if not sequences:
        raise ValueError("cannot build a batch with no sequences")
    if len(sequences) != len(query_lens):
        raise ValueError(f"{len(sequences)} sequences but {len(query_lens)} query lengths")

    input_ids: list[int] = []
    position_ids: list[int] = []
    query_seq: list[int] = []
    query_pos: list[int] = []
    key_seq: list[int] = []
    key_pos: list[int] = []
    write_slots: list[Tensor] = []
    read_slots: list[Tensor] = []
    logit_indices: list[int] = []

    for index, (sequence, query_len) in enumerate(zip(sequences, query_lens)):
        if query_len <= 0:
            raise ValueError(f"sequence {sequence.seq_id} was scheduled for {query_len} tokens")
        start = sequence.num_computed_tokens
        end = start + query_len
        if end > sequence.num_tokens:
            raise ValueError(
                f"sequence {sequence.seq_id} has {sequence.num_tokens} tokens; "
                f"cannot compute up to {end}"
            )

        input_ids.extend(sequence.token_ids[start:end])
        position_ids.extend(range(start, end))

        # Queries this iteration, tagged with the sequence they belong to.
        query_seq.extend([index] * query_len)
        query_pos.extend(range(start, end))

        # Keys to attend over: the whole prefix including what is being written now.
        key_seq.extend([index] * end)
        key_pos.extend(range(end))

        write_slots.append(cache.slots(sequence.block_table, start, end, device=device))
        read_slots.append(cache.slots(sequence.block_table, 0, end, device=device))

        logit_indices.append(len(input_ids) - 1)

    long = dict(dtype=torch.long, device=device)
    mask = build_ragged_mask(
        torch.tensor(query_seq, **long),
        torch.tensor(query_pos, **long),
        torch.tensor(key_seq, **long),
        torch.tensor(key_pos, **long),
        dtype=dtype,
    )

    state = AttentionState(
        cache=cache,
        # start/end describe a single sequence and mean nothing for a batch; the
        # mask carries the structure instead. position_ids are always explicit
        # here, so nothing downstream derives positions from `start`.
        start=0,
        end=len(key_seq),
        mask=mask,
        is_causal=False,
        write_slots=torch.cat(write_slots),
        read_slots=torch.cat(read_slots),
    )

    return BatchInput(
        input_ids=torch.tensor([input_ids], **long),
        position_ids=torch.tensor([position_ids], **long),
        state=state,
        logit_indices=torch.tensor(logit_indices, **long),
        sequences=list(sequences),
        query_lens=list(query_lens),
    )
