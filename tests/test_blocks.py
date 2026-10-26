"""The block pool: allocation on demand, reference counting, LRU order.

Pure bookkeeping, no tensors and no weights, so this whole file runs in
milliseconds. The reference-counting tests here are the precondition for stage 7:
cancelling a sequence must release exactly its own blocks and leave a shared
prefix standing.
"""

from __future__ import annotations

import pytest

from vox.blocks import BlockPool, BlockPoolExhausted
from vox.sequence import Sequence


def seq(**kwargs) -> Sequence:
    return Sequence(token_ids=[1, 2, 3], **kwargs)


# -- single blocks --------------------------------------------------------

def test_allocate_takes_a_block_and_sets_one_reference():
    pool = BlockPool(num_blocks=4, block_size=8)
    assert pool.num_free_blocks == 4

    block_id = pool.allocate_block()
    assert pool.num_free_blocks == 3
    assert pool.num_used_blocks == 1
    assert pool.blocks[block_id].ref_count == 1


def test_exhaustion_raises_its_own_error_type():
    """Stage 5 catches this specifically in order to preempt instead of failing."""
    pool = BlockPool(num_blocks=2, block_size=8)
    pool.allocate_block()
    pool.allocate_block()
    with pytest.raises(BlockPoolExhausted, match="all 2 blocks"):
        pool.allocate_block()


def test_decref_to_zero_returns_the_block():
    pool = BlockPool(num_blocks=2, block_size=8)
    block_id = pool.allocate_block()
    assert pool.decref(block_id) is True
    assert pool.num_free_blocks == 2


def test_a_shared_block_survives_one_owner_leaving():
    """The mechanism the whole barge-in story rests on."""
    pool = BlockPool(num_blocks=2, block_size=8)
    block_id = pool.allocate_block()
    pool.incref(block_id)
    assert pool.blocks[block_id].ref_count == 2

    assert pool.decref(block_id) is False      # first owner lets go
    assert pool.num_free_blocks == 1
    assert pool.blocks[block_id].ref_count == 1

    assert pool.decref(block_id) is True       # last owner lets go
    assert pool.num_free_blocks == 2


def test_sharing_a_free_block_is_refused():
    pool = BlockPool(num_blocks=2, block_size=8)
    with pytest.raises(RuntimeError, match="cannot share free block"):
        pool.incref(0)


def test_double_free_is_refused():
    pool = BlockPool(num_blocks=2, block_size=8)
    block_id = pool.allocate_block()
    pool.decref(block_id)
    with pytest.raises(RuntimeError, match="double free"):
        pool.decref(block_id)


def test_free_list_is_least_recently_freed_first():
    """LRU. Stage 6 relies on this: a block freed long ago is the one least
    likely to still hold a prefix anybody wants."""
    pool = BlockPool(num_blocks=3, block_size=8)
    a, b, c = pool.allocate_block(), pool.allocate_block(), pool.allocate_block()

    pool.decref(b)   # freed first -> handed out first
    pool.decref(a)
    pool.decref(c)

    assert [pool.allocate_block() for _ in range(3)] == [b, a, c]


# -- whole sequences ------------------------------------------------------

def test_blocks_needed_rounds_up():
    pool = BlockPool(num_blocks=8, block_size=4)
    assert pool.blocks_needed(0) == 0
    assert pool.blocks_needed(1) == 1
    assert pool.blocks_needed(4) == 1
    assert pool.blocks_needed(5) == 2


def test_allocation_is_on_demand_not_preallocated():
    """A sequence holds blocks for the tokens it has, not the ones it may generate."""
    pool = BlockPool(num_blocks=8, block_size=4)
    s = seq()

    pool.allocate_for(s, 4)                 # exactly one block's worth
    assert len(s.block_table) == 1
    s.num_computed_tokens = 4

    pool.allocate_for(s, 1)                 # crossing the boundary adds one
    assert len(s.block_table) == 2
    s.num_computed_tokens = 5

    pool.allocate_for(s, 3)                 # still inside block 2, adds nothing
    assert len(s.block_table) == 2


def test_allocate_for_is_a_no_op_when_capacity_already_exists():
    pool = BlockPool(num_blocks=8, block_size=16)
    s = seq()
    pool.allocate_for(s, 3)
    assert pool.allocate_for(s, 5) == []
    assert len(s.block_table) == 1


def test_allocate_for_leaves_the_block_table_untouched_when_it_cannot_fit():
    """Capacity is checked before anything is mutated, so stage 5 can preempt
    and retry rather than clean up a half-grown sequence."""
    pool = BlockPool(num_blocks=2, block_size=4)
    s = seq()
    with pytest.raises(BlockPoolExhausted, match="needs 3 more"):
        pool.allocate_for(s, 12)            # wants 3 blocks, pool has 2
    assert s.block_table == []
    assert pool.num_free_blocks == 2


def test_freeing_a_sequence_clears_its_table_and_returns_the_blocks():
    pool = BlockPool(num_blocks=8, block_size=4)
    s = seq()
    pool.allocate_for(s, 9)                 # 3 blocks
    held = list(s.block_table)

    reclaimed = pool.free(s)
    assert sorted(reclaimed) == sorted(held)
    assert s.block_table == []
    assert pool.num_free_blocks == 8


def test_freeing_one_sequence_leaves_a_shared_prefix_standing():
    """Stage 7's cancellation guarantee, expressed at the pool level.

    Two sequences, the first block shared. Freeing one must reclaim only the
    blocks it alone owns.
    """
    pool = BlockPool(num_blocks=8, block_size=4)
    a, b = seq(), seq()

    pool.allocate_for(a, 8)                 # a: 2 blocks
    shared = a.block_table[0]

    pool.incref(shared)                     # b adopts a's first block
    b.block_table.append(shared)
    b.num_computed_tokens = 4
    pool.allocate_for(b, 4)                 # b: shared + 1 of its own
    own_to_b = b.block_table[1]

    reclaimed = pool.free(b)
    assert reclaimed == [own_to_b]           # only b's own block came back
    assert pool.blocks[shared].ref_count == 1
    assert shared in a.block_table


def test_utilization_tracks_usage():
    pool = BlockPool(num_blocks=4, block_size=8)
    assert pool.utilization == 0.0
    pool.allocate_block()
    pool.allocate_block()
    assert pool.utilization == 0.5


def test_rejects_nonsense_geometry():
    with pytest.raises(ValueError):
        BlockPool(num_blocks=0, block_size=8)
    with pytest.raises(ValueError):
        BlockPool(num_blocks=4, block_size=0)
