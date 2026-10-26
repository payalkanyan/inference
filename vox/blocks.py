"""The block pool: who owns which block of KV memory.

This module is pure bookkeeping -- it never touches a tensor. The bytes live in
``PagedKVCache``; this decides which sequence may use which block, and when a
block goes back on the free list.

Three properties here carry the project:

* **Reference counting.** A block can belong to more than one sequence. Freeing a
  sequence decrements; a block returns to the pool only when its last owner lets
  go. This is how two turns of a conversation share one system prompt, and how
  cancellation in stage 7 releases exactly one sequence's own blocks while a
  shared prefix stays standing.
* **LRU order.** The free list hands out the block that has been free *longest*.
  A freed block keeps its contents and its hash, so it can still be reused by a
  later request; "oldest first" is what makes a recently released prefix likely
  to still be there when the next turn arrives.
* **The prefix registry.** A map from chained block hash to physical block, so a
  new request can find a prefix somebody else already computed. Every hit is
  verified against the stored parent hash and token ids, because the hash is an
  index and not proof -- see ``vox/prefix.py``.

Freeing and eviction are therefore different things. Freeing drops a reference
and leaves the contents intact and findable. Eviction happens later, when that
block is actually handed out for new data, and *that* is the point at which it
must leave the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from vox.sequence import Sequence


class BlockPoolExhausted(RuntimeError):
    """No free blocks. Stage 5 catches this and preempts instead of failing."""


class LRUFreeList:
    """Free block ids, least recently freed first.

    A deque would cover handing out the oldest, but prefix caching also needs to
    pull one specific block back out of the middle when a cache hit revives it.
    A dict preserves insertion order and deletes in constant time, so it serves
    as an ordered set.
    """

    def __init__(self, block_ids: range | list[int]) -> None:
        self._ids: dict[int, None] = dict.fromkeys(block_ids)

    def push(self, block_id: int) -> None:
        self._ids[block_id] = None

    def pop_oldest(self) -> int:
        block_id = next(iter(self._ids))
        del self._ids[block_id]
        return block_id

    def remove(self, block_id: int) -> None:
        del self._ids[block_id]

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return bool(self._ids)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._ids

    def __iter__(self) -> Iterator[int]:
        return iter(self._ids)

    def __repr__(self) -> str:
        return f"LRUFreeList({list(self._ids)})"


@dataclass
class Block:
    block_id: int
    ref_count: int = 0

    #: Set once the block is full and computed. A free block keeps these, which
    #: is what lets a later request find it; they are cleared on eviction.
    block_hash: int | None = None
    parent_hash: int | None = None
    token_ids: tuple[int, ...] | None = None

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    @property
    def is_cached(self) -> bool:
        return self.block_hash is not None

    def forget(self) -> None:
        self.block_hash = None
        self.parent_hash = None
        self.token_ids = None


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError(f"need positive num_blocks/block_size, got {num_blocks}/{block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = LRUFreeList(range(num_blocks))
        #: Chained block hash -> physical block id, for blocks whose contents are
        #: known. Entries may point at blocks that are free but not yet reused.
        self.cached_blocks: dict[int, int] = {}

    # -- single blocks ----------------------------------------------------

    def allocate_block(self) -> int:
        """Hand out the least recently freed block, evicting it if it held a prefix.

        This is the only place eviction happens, and it has to deregister: the
        block is about to be overwritten with unrelated KV, and an entry left
        pointing at it would hand that KV to a later request as if it were their
        own prefix.
        """
        if not self.free_block_ids:
            raise BlockPoolExhausted(f"all {self.num_blocks} blocks are in use")
        block_id = self.free_block_ids.pop_oldest()
        block = self.blocks[block_id]
        if not block.is_free:
            raise RuntimeError(f"block {block_id} was on the free list with refcount {block.ref_count}")
        if block.is_cached:
            del self.cached_blocks[block.block_hash]
            block.forget()
        block.ref_count = 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Take a second reference, e.g. when a new sequence adopts a shared prefix."""
        block = self.blocks[block_id]
        if block.is_free:
            raise RuntimeError(f"cannot share free block {block_id}")
        block.ref_count += 1

    def decref(self, block_id: int) -> bool:
        """Drop one reference. Returns True if the block went back on the free list."""
        block = self.blocks[block_id]
        if block.is_free:
            raise RuntimeError(f"double free of block {block_id}")
        block.ref_count -= 1
        if block.is_free:
            # Contents and hash are kept: the block is reusable, not erased.
            self.free_block_ids.push(block_id)
            return True
        return False

    # -- the prefix registry ----------------------------------------------

    def find_cached(
        self, block_hash: int, parent_hash: int | None, token_ids: tuple[int, ...]
    ) -> int | None:
        """Look up a block holding exactly these tokens after exactly this parent.

        The hash is only an index. A match on it is confirmed against the stored
        parent and tokens, so a collision degrades to a miss instead of serving
        another conversation's KV.
        """
        block_id = self.cached_blocks.get(block_hash)
        if block_id is None:
            return None
        block = self.blocks[block_id]
        if block.parent_hash != parent_hash or block.token_ids != token_ids:
            return None
        return block_id

    def register(
        self, block_id: int, block_hash: int, parent_hash: int | None,
        token_ids: tuple[int, ...],
    ) -> None:
        """Publish a full, computed block so later requests can find it."""
        block = self.blocks[block_id]
        if block.is_free:
            raise RuntimeError(f"cannot register free block {block_id}")
        block.block_hash = block_hash
        block.parent_hash = parent_hash
        block.token_ids = token_ids
        self.cached_blocks[block_hash] = block_id

    def claim_cached(self, block_id: int) -> None:
        """Take a reference to a block found in the registry.

        It may be in use by another sequence, in which case this is plain
        sharing, or free but not yet reused, in which case it has to come back
        off the free list first.
        """
        block = self.blocks[block_id]
        if block.is_free:
            self.free_block_ids.remove(block_id)
            block.ref_count = 1
        else:
            block.ref_count += 1

    @property
    def num_cached_blocks(self) -> int:
        return len(self.cached_blocks)

    # -- whole sequences --------------------------------------------------

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate_for(self, sequence: Sequence, num_new_tokens: int) -> list[int]:
        """Grow ``sequence``'s block table to cover ``num_new_tokens`` more tokens.

        Allocation is on demand: a sequence holds blocks for the tokens it has,
        not for the tokens it might eventually generate. Capacity is checked
        before anything is mutated, so a failure leaves the block table untouched
        and stage 5 can preempt and retry.
        """
        needed = self.blocks_needed(sequence.num_computed_tokens + num_new_tokens)
        shortfall = needed - len(sequence.block_table)
        if shortfall <= 0:
            return []
        if shortfall > self.num_free_blocks:
            raise BlockPoolExhausted(
                f"sequence {sequence.seq_id} needs {shortfall} more block(s), "
                f"{self.num_free_blocks} free"
            )
        allocated = [self.allocate_block() for _ in range(shortfall)]
        sequence.block_table.extend(allocated)
        return allocated

    def free(self, sequence: Sequence) -> list[int]:
        """Release the sequence's claim on its blocks; return those actually freed.

        Blocks still referenced by someone else stay allocated, which is the
        whole point of the refcount.
        """
        reclaimed = [b for b in sequence.block_table if self.decref(b)]
        sequence.block_table.clear()
        return reclaimed

    # -- introspection (instrumentation in stage 8 reads these) -----------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks

    def __repr__(self) -> str:
        return (
            f"BlockPool({self.num_used_blocks}/{self.num_blocks} used, "
            f"block_size={self.block_size})"
        )
