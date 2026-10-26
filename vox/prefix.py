"""Block hashing for prefix sharing.

Two sequences can share a physical block only if that block holds the same
tokens *at the same place in the same prefix*. Content alone is not enough: the
keys and values in a block are a function of everything before it, so the same 16
tokens appearing at a different offset, or after a different history, produce
entirely different KV.

So block identity is **chained**::

    identity(0) = hash(None,            tokens[0:B])
    identity(i) = hash(identity(i - 1),  tokens[i*B:(i+1)*B])

which makes a block's identity depend on the whole prefix leading to it, while
staying a single integer to look up.

Two things this module does *not* try to do, both deliberate:

* **Partial blocks are never hashed.** A block that is not full can still gain
  tokens, so its contents are not yet final and nothing may be keyed on them.
* **The hash is an index, not proof.** A 64-bit collision is unlikely but its
  consequence here is the worst in the project -- silently serving one
  conversation's KV to another, producing fluent and completely wrong output. So
  the pool stores the parent hash and token ids alongside each cached block and
  verifies them on every hit. The hash narrows the search; equality decides.
"""

from __future__ import annotations


def hash_block(parent_hash: int | None, token_ids: tuple[int, ...]) -> int:
    """Identity of a full block of ``token_ids`` following ``parent_hash``.

    ``None`` as the parent means this is the first block of a sequence, which is
    what anchors the chain at position 0 -- and therefore why a reused block's KV
    is valid at exactly the positions it was computed for.
    """
    return hash((parent_hash, token_ids))
