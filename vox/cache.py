"""KV cache implementations.

Two of them, both still live:

``ContiguousKVCache`` (stage 2)
    One preallocated buffer per sequence, written left to right. Kept as a
    reference implementation -- it is the simplest thing that works, so the paged
    cache is checked against it.

``PagedKVCache`` (stage 3)
    One fixed-size pool of blocks shared by every sequence, addressed through a
    per-sequence block table. No per-request preallocation: a sequence holds
    blocks for the tokens it has, not for the tokens it might generate.

Both store keys *after* RoPE has been applied, which is what makes them reusable:
RoPE depends only on relative position, so a key rotated at absolute position
``n`` stays correct for every later query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from vox.blocks import BlockPool
from vox.config import Qwen2Config
from vox.prefix import hash_block
from vox.sequence import Sequence
from vox.state import AttentionState, mask_for


def _empty(shape, *, device, dtype) -> Tensor:
    """Allocate with inference mode explicitly off.

    A buffer created inside ``inference_mode`` becomes an "inference tensor" and
    then cannot be mutated outside it -- a confusing failure at a distance for
    something meant to live as long as the process, so it is ruled out here
    rather than documented.
    """
    with torch.inference_mode(False):
        return torch.zeros(shape, device=device, dtype=dtype)


class KVCache(ABC):
    """Attention talks to a cache through exactly these two calls.

    Both take the ``AttentionState``, so attention never has to know whether the
    memory behind it is one contiguous run or a scatter of blocks.
    """

    @abstractmethod
    def write(self, layer_idx: int, k: Tensor, v: Tensor, state: AttentionState) -> None:
        """Store ``k``/``v`` of shape [1, num_kv_heads, n, head_dim]."""

    @abstractmethod
    def read(self, layer_idx: int, state: AttentionState) -> tuple[Tensor, Tensor]:
        """Return the prefix as [1, num_kv_heads, state.end, head_dim]."""


class ContiguousKVCache(KVCache):
    """One sequence's keys and values, in two contiguous buffers.

    Layout is ``[num_layers, num_kv_heads, max_seq_len, head_dim]``.

    This is deliberately the wrong shape for voice traffic, which is what
    motivates the paged cache: it reserves ``max_seq_len`` per sequence whether or
    not those tokens are used, and two turns sharing a system prompt cannot share
    a single byte of it.
    """

    def __init__(
        self,
        cfg: Qwen2Config,
        max_seq_len: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, max_seq_len, cfg.head_dim)
        self.key = _empty(shape, device=device, dtype=dtype)
        self.value = _empty(shape, device=device, dtype=dtype)
        #: Number of tokens written so far, i.e. the next free slot.
        self.length = 0

    def build_state(self, q_len: int, *, device, dtype: torch.dtype) -> AttentionState:
        start, end = self.length, self.length + q_len
        if end > self.max_seq_len:
            raise ValueError(f"sequence would reach {end} tokens, cache holds {self.max_seq_len}")
        mask, is_causal = mask_for(start, end, q_len, device=device, dtype=dtype)
        return AttentionState(self, start, end, mask, is_causal)

    def write(self, layer_idx: int, k: Tensor, v: Tensor, state: AttentionState) -> None:
        n = k.shape[2]
        start = state.start
        if start + n > self.max_seq_len:
            raise ValueError(
                f"cache overflow: writing {n} tokens at {start} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        self.key[layer_idx, :, start : start + n] = k[0]
        self.value[layer_idx, :, start : start + n] = v[0]

    def read(self, layer_idx: int, state: AttentionState) -> tuple[Tensor, Tensor]:
        end = state.end
        return (
            self.key[layer_idx, :, :end].unsqueeze(0),
            self.value[layer_idx, :, :end].unsqueeze(0),
        )

    def advance(self, n: int) -> None:
        """Commit ``n`` newly written tokens. Called once per forward, not per layer."""
        if self.length + n > self.max_seq_len:
            raise ValueError(
                f"cache overflow: {self.length} + {n} exceeds max_seq_len={self.max_seq_len}"
            )
        self.length += n

    def reset(self) -> None:
        """Forget the contents without reallocating. Stale bytes are never read,
        because reads are always bounded by ``length``."""
        self.length = 0

    @property
    def num_bytes(self) -> int:
        return self.key.numel() * self.key.element_size() * 2

    def __repr__(self) -> str:
        return (
            f"ContiguousKVCache(length={self.length}/{self.max_seq_len}, "
            f"{self.num_bytes / 1e6:.1f}MB)"
        )


class PagedKVCache(KVCache):
    """A fixed pool of equally sized blocks, shared by every sequence.

    Storage is ``[num_layers, num_blocks, block_size, num_kv_heads, head_dim]``.
    The block dimensions are kept in the shape because they are the thing that
    makes this paged, but all addressing goes through a flat *slot* index::

        slot = block_id * block_size + offset_within_block

    which is just a free view of the same memory. One slot vector is computed per
    forward pass and reused by all 24 layers: ``write_slots`` for the new tokens,
    ``read_slots`` for the prefix to attend over. Reads gather and writes scatter,
    because without custom kernels that is how you make a discontiguous sequence
    look contiguous to SDPA.
    """

    def __init__(
        self,
        cfg: Qwen2Config,
        num_blocks: int,
        block_size: int = 16,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, block_size)
        self.enable_prefix_caching = enable_prefix_caching

        shape = (
            cfg.num_hidden_layers,
            num_blocks,
            block_size,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        self.key = _empty(shape, device=device, dtype=dtype)
        self.value = _empty(shape, device=device, dtype=dtype)

        # Slot-addressed views over the same storage. Free: no copy, no extra memory.
        flat = (cfg.num_hidden_layers, num_blocks * block_size, cfg.num_key_value_heads, cfg.head_dim)
        self._key_slots = self.key.view(flat)
        self._value_slots = self.value.view(flat)

    # -- slot arithmetic --------------------------------------------------

    def slots(self, block_table: list[int], start: int, end: int, *, device) -> Tensor:
        """Physical slots for logical positions [start, end) of a block table."""
        if end > len(block_table) * self.block_size:
            raise ValueError(
                f"position {end - 1} is past the {len(block_table)} block(s) "
                f"allocated to this sequence"
            )
        positions = torch.arange(start, end, device=device)
        table = torch.tensor(block_table, dtype=torch.long, device=device)
        return table[positions // self.block_size] * self.block_size + positions % self.block_size

    def build_state(
        self, sequence: Sequence, q_len: int, *, device, dtype: torch.dtype
    ) -> AttentionState:
        start = sequence.num_computed_tokens
        end = start + q_len
        mask, is_causal = mask_for(start, end, q_len, device=device, dtype=dtype)
        return AttentionState(
            cache=self,
            start=start,
            end=end,
            mask=mask,
            is_causal=is_causal,
            write_slots=self.slots(sequence.block_table, start, end, device=device),
            read_slots=self.slots(sequence.block_table, 0, end, device=device),
        )

    # -- the KVCache contract ---------------------------------------------

    def write(self, layer_idx: int, k: Tensor, v: Tensor, state: AttentionState) -> None:
        # [1, num_kv_heads, n, head_dim] -> [n, num_kv_heads, head_dim], one row per slot.
        self._key_slots[layer_idx].index_copy_(
            0, state.write_slots, k[0].transpose(0, 1).contiguous()
        )
        self._value_slots[layer_idx].index_copy_(
            0, state.write_slots, v[0].transpose(0, 1).contiguous()
        )

    def read(self, layer_idx: int, state: AttentionState) -> tuple[Tensor, Tensor]:
        k = self._key_slots[layer_idx].index_select(0, state.read_slots)
        v = self._value_slots[layer_idx].index_select(0, state.read_slots)
        # [end, num_kv_heads, head_dim] -> [1, num_kv_heads, end, head_dim]
        return k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)

    # -- allocation, delegated to the pool --------------------------------

    def allocate_for(self, sequence: Sequence, num_new_tokens: int) -> list[int]:
        return self.pool.allocate_for(sequence, num_new_tokens)

    def free(self, sequence: Sequence) -> list[int]:
        return self.pool.free(sequence)

    # -- prefix caching ---------------------------------------------------

    def attach_cached_prefix(self, sequence: Sequence, *, leave_one: bool = True) -> int:
        """Adopt already-computed blocks for as much of the prompt as possible.

        Walks the block chain from position 0 and stops at the first miss,
        because a hit further along would be meaningless: the chain is what makes
        a block's identity depend on its whole prefix.

        Returns the number of tokens that will not need prefilling. Those blocks
        are now referenced by this sequence, so ``num_computed_tokens`` moves
        forward and the forward pass simply starts later -- the write slots cover
        only the new tokens while the read slots still cover the whole prefix,
        which is why nothing in the attention path had to change for this.

        ``leave_one`` is false only when pinning a prefix for later reuse rather
        than preparing a sequence to generate: there is no forward pass to leave
        a token for.
        """
        if not self.enable_prefix_caching or sequence.num_computed_tokens:
            return 0

        # One token must always be left to compute. A fully cached prompt would
        # give the forward pass nothing to produce logits from, and there would
        # be no next token to sample.
        reusable = sequence.num_tokens - 1 if leave_one else sequence.num_tokens

        # A declared prefix bounds the search. The new user segment is by
        # construction the part that differs, so speculating into it is wasted
        # hashing, and a hit there would mean the caller mislabelled the split.
        if sequence.declared_prefix_len:
            reusable = min(reusable, sequence.declared_prefix_len)

        parent_hash: int | None = None
        index = 0
        while (index + 1) * self.block_size <= reusable:
            tokens = tuple(
                sequence.token_ids[index * self.block_size : (index + 1) * self.block_size]
            )
            block_hash = hash_block(parent_hash, tokens)
            block_id = self.pool.find_cached(block_hash, parent_hash, tokens)
            if block_id is None:
                break
            self.pool.claim_cached(block_id)
            sequence.block_table.append(block_id)
            sequence.block_hashes.append(block_hash)
            parent_hash = block_hash
            index += 1

        reused = index * self.block_size
        sequence.num_computed_tokens = reused
        sequence.num_cached_tokens = reused
        return reused

    def detach_prefix(self, sequence: Sequence) -> None:
        """Give back blocks adopted by ``attach_cached_prefix``.

        Used when a sequence turns out not to be admissible after all, so that
        only running sequences ever hold memory.
        """
        self.free(sequence)
        sequence.num_computed_tokens = 0
        sequence.num_cached_tokens = 0
        sequence.block_hashes.clear()

    def register_computed_blocks(self, sequence: Sequence) -> list[int]:
        """Publish any block that has just become full and fully computed.

        Only full blocks: a partial tail can still gain tokens, so its contents
        are not final and nothing may be keyed on them. Extends the sequence's
        existing hash chain rather than rebuilding it, so the cost is in the
        blocks that are new this iteration.
        """
        if not self.enable_prefix_caching:
            return []

        full_blocks = sequence.num_computed_tokens // self.block_size
        registered: list[int] = []
        while len(sequence.block_hashes) < full_blocks:
            index = len(sequence.block_hashes)
            parent_hash = sequence.block_hashes[-1] if sequence.block_hashes else None
            tokens = tuple(
                sequence.token_ids[index * self.block_size : (index + 1) * self.block_size]
            )
            block_hash = hash_block(parent_hash, tokens)
            sequence.block_hashes.append(block_hash)

            block_id = sequence.block_table[index]
            if not self.pool.blocks[block_id].is_cached:
                self.pool.register(block_id, block_hash, parent_hash, tokens)
                registered.append(block_id)
        return registered

    @property
    def bytes_per_block(self) -> int:
        per_block = self.key.numel() // self.num_blocks
        return per_block * self.key.element_size() * 2

    @property
    def num_bytes(self) -> int:
        return self.key.numel() * self.key.element_size() * 2

    def __repr__(self) -> str:
        return (
            f"PagedKVCache({self.pool.num_used_blocks}/{self.num_blocks} blocks used, "
            f"block_size={self.block_size}, {self.num_bytes / 1e6:.1f}MB)"
        )
