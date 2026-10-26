# vox

A from-scratch LLM inference engine in Python, specialized for interruptible
voice-agent traffic. Phase 1: a correct, working engine with built-in
instrumentation.

**Status: stage 7 of 9.** Everything above, plus the specialisation this project
exists for: immediate cancellation, an explicit Turn abstraction, and
deadline-aware admission. Verified token-identical to HuggingFace transformers
under greedy decoding through all four paths; output is unchanged by batching, by
chunking, by having KV thrown away mid-decode, by reusing a prefix another request
computed, and by another request being cancelled alongside it. 240 tests pass.
The full README (architecture, component-by-component walkthrough, and an honest
"what this is not") lands in stage 9.

## Build order

| # | Stage | Status |
|---|-------|--------|
| 1 | Model forward pass + naive generation, token-identical to HF | done |
| 2 | Contiguous KV cache | done |
| 3 | Paged KV cache + block manager | done |
| 4 | Scheduler + continuous batching | done |
| 5 | Chunked prefill + preemption | done |
| 6 | Prefix caching | done |
| 7 | Cancellation, Turn abstraction, deadline-aware admission | done |
| 8 | Instrumentation + voice-conversation demo | |
| 9 | README | |

## Running stage 1

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python transformers safetensors pytest

.venv/bin/python scripts/download_model.py        # ~1 GB into the HF cache
.venv/bin/python -m pytest tests/ -q              # the correctness gate
.venv/bin/python scripts/run_reference.py "What is the capital of France?"
.venv/bin/python scripts/run_engine.py
.venv/bin/python scripts/run_prefix.py
.venv/bin/python scripts/run_bargein.py
```

`run_reference.py` runs every single-sequence path and fails if they disagree.
`run_bargein.py` runs a short conversation with two barge-ins and reports what
each cancellation cost. `run_prefix.py` runs a four-turn conversation that re-sends the same 99-token
system prompt, with prefix caching off and then on, and checks the two agree.
`run_engine.py` drives four requests through the engine, two of them arriving
mid-flight, prints the batch composition each iteration, and then checks every
answer against an unbatched run. Pass `squeeze` to run it with a tight budget and
a tight pool, which forces prompts to arrive in chunks and sequences to be
preempted:

```
iter  seqs   tok   blk  composition
   0     2     6     3  france:prefillx5, sky:prefill
   3     3     6     5  france:decode, sky:decode, count:prefillx4
   7     2     2     6  france:decode, sky:decode   PREEMPTED: code, count
  14     2     6     8  france:decode, sky:recomputex5
```

Set `VOX_MODEL_DIR` to use weights you already have on disk instead of the HF cache.

`tests/test_layers.py`, `tests/test_blocks.py` and the structural halves of
`tests/test_cache.py` and `tests/test_paged.py` need no weights and run in about a
second. The weight-backed tests load both models and take roughly two minutes on
CPU in total.

## The two reference paths

`generate_naive` keeps **no KV cache at all** -- every decode step recomputes the
whole sequence. That is deliberate: it is the simplest thing that can be correct,
so everything built later has an unambiguous reference to be checked against. It
stays in the tree and stays tested for exactly that reason.

`generate_cached` (stage 2) prefills the prompt once and then decodes one token at
a time against a `ContiguousKVCache`: one preallocated buffer per sequence.

`generate_paged` (stage 3) does the same against a `PagedKVCache`: one fixed pool
of equally sized blocks shared by every sequence, reached through a per-sequence
block table. Blocks are allocated as the sequence grows and released when it ends.

`generate_engine` (stage 4) is a thin adapter that pushes one request through the
real `Engine`, so the batching machinery is held to the same gate as the rest. It
is not a fourth implementation.

Only the engine is the real thing; the first three remain as test oracles, and the
HF gate is parametrised over all of them so each new path is re-checked rather
than self-certified once. The tok/s figures the scripts print are a sanity signal
that the cache is doing something, not a performance claim.

## Why paged

The contiguous cache is the wrong shape for the target workload. It reserves
`max_seq_len` per sequence whether or not those tokens are used, and two turns
sharing a system prompt cannot share a single byte of it. Voice traffic is a long
shared prefix, a short decode, and frequent cancellation -- close to its worst
case.

The paged cache addresses memory by *slot* rather than by offset:

```
slot = block_id * block_size + offset_within_block
```

One slot vector is computed per forward pass and reused by all 24 layers:
`write_slots` for the new tokens, `read_slots` for the prefix to attend over.
Reads gather and writes scatter, because without custom kernels that is how you
make a discontiguous sequence look contiguous to SDPA.

Reference counting is in place from the start even though nothing shares blocks
yet. Prefix sharing arrives in stage 6, and cancellation in stage 7 has to release
exactly one sequence's own blocks while a shared system prompt stays standing --
that is the refcount. The free list is LRU-ordered for the same reason: once a
freed block keeps its contents, "oldest first" is what makes a recently released
prefix likely to still be there for the next turn.

## Batching

A batch's sequences disagree about everything that matters: how many tokens each
contributes this iteration, how long its prefix already is, and which blocks hold
it. Padding to a common shape would make the cost the *maximum* rather than the
sum -- one 500-token prefill sharing a batch with eight 1-token decodes would
compute 4500 query positions to produce 9 tokens.

So `vox/batch.py` flattens instead. Queries become one row of `T = sum(query_len)`
tokens, keys one run of `K = sum(kv_len)`, and sequences are held apart by the
mask rather than by the shape:

```python
allowed[t, k] = (seq_of_query[t] == seq_of_key[k]) & (key_pos[k] <= query_pos[t])
```

One term for "same sequence", one for causality. That covers prefill, decode,
mixed batches and chunk continuation with no special cases, which is why stage 4
needed no changes at all to `layers.py` or `cache.py`: the tensor batch dimension
stays 1, so batching happens along the token dimension, and the paged cache
already addresses memory by slot rather than by offset.

The cost of flattening without custom kernels is that the `T x K` mask is
materialised. A varlen attention kernel would read the sequence boundaries
directly and never build it. That is a deliberate phase-1 trade, and it is the
main reason the token budget exists.

### Scheduling

Each iteration the scheduler reserves budget for **decodes first**. A running
sequence contributes exactly one token, so the whole running set costs at most
`max_num_seqs` tokens; admitting prefills first would let one long prompt consume
an entire iteration and stall every conversation already in flight. Whatever
budget is left admits new requests, FIFO.

Whatever budget is left then goes to sequences that are **catching up** -- a
first prefill, a later chunk of one, or a recompute after preemption, which are
all the same operation: compute a run of tokens whose KV is missing. A long prompt
therefore overlaps with everyone else's decoding instead of blocking it for an
iteration, which is the whole point for a workload with a long shared prefix and
hard TTFT deadlines. Anything still left admits new requests, FIFO.

### Preemption

Chunking and preemption belong together, because chunking is what makes
preemption rare: **a prefill can shrink to fit whatever memory is free, and a
decode cannot.** A decode's minimum unit is one token, and if the block for it
does not exist, something has to give.

So before anything is scheduled, a pre-pass preempts until every running sequence
can advance by at least one token. Preemption is recompute-based: the blocks are
released and the KV is simply forgotten, to be rebuilt later from tokens the
sequence still holds. Nothing is copied anywhere, so preempting is as cheap as
freeing -- the cost is paid in recomputed FLOPs on resume.

The victim is always the **most recently admitted** sequence. That is what makes
progress guaranteed: the oldest running sequence is never evicted, so it always
advances and eventually finishes, freeing memory for everyone behind it.

It also means the newest sequence bears the whole cost. Under sustained pressure
it can be evicted and recomputed repeatedly while an older one drains -- visible
in `run_engine.py squeeze`, where a deliberately undersized pool produces 12
preemptions and 60 iterations for work that takes 26 when memory is adequate. That
is a real characteristic of newest-first eviction, not a bug, and the honest fix is
a larger pool rather than a cleverer policy.

A request whose prompt could not fit in the pool even when empty is rejected at
admission, because admitting it would mean chunk-prefilling it until memory ran
out and then preempting it forever. A sequence that *outgrows* the pool while
generating raises instead, which is a loud failure rather than a livelock.

### What batching does and does not change

Batched output is token-identical to running each request alone, and that is the
gate. It is *not* bit-identical, and cannot be: attention reduces over every key
in the batch, so a sequence batched with others is reduced over a wider tensor
than it would be alone. Same values, different accumulation -- measured at around
3e-5 on the logits, orders of magnitude below anything that could flip an argmax.
A single sequence through the flattened path **is** bit-identical, which is what
separates "the mask is wrong" from "batching moved the arithmetic".

Chunked prefill and preemption are held to the same standard, and the tests that
check them assert that the mechanism actually engaged -- `num_catch_up_chunks` and
`num_preemptions` on the engine. A "preemption changes nothing" test passes
trivially if the pool was quietly large enough, so it is not allowed to.

## Prefix caching

Every turn of a voice conversation re-sends the same system prompt, tool
definitions and history, and changes only a short tail. Recomputing that prefix
each turn is the single largest avoidable cost in the workload, so blocks are
shared between separate requests rather than only within one.

Sharing is keyed on a **chained** hash:

```
identity(0) = hash(None,            tokens[0:B])
identity(i) = hash(identity(i-1),   tokens[i*B:(i+1)*B])
```

Chaining is not an optimisation, it is the correctness condition. A block's keys
and values are a function of everything before it, so the same 16 tokens at a
different offset or after a different history hold entirely different KV. Keying
on content alone would share them and produce fluent, wrong output.

On admission the scheduler walks the chain from position 0 and stops at the first
miss -- a hit further along would be meaningless. Blocks that hit are adopted with
an `incref`, `num_computed_tokens` jumps forward, and the forward pass simply
starts later. Nothing in the attention path needed to change for this: write slots
already covered only the new tokens while read slots covered the whole prefix.

Measured on `run_prefix.py`, with a 99-token system prompt and 3-6 new tokens per
turn:

```
 turn  prompt  prefilled  reused    hit
    0     105        105       0    0%
    1     102          6      96   94%
    2     102          6      96   94%
    3     102          6      96   94%
```

This saves prefill *work*, not decode iterations -- the iteration count is
unchanged. The win is in time-to-first-token, which stage 7 starts scheduling
against.

### Three ways this can be silently wrong

All three produce fluent, incorrect text rather than a crash, so each has a test
named after it.

**Sharing across different prefixes.** Handled by chaining, above.

**Accepting a hash collision.** The registry is a 64-bit index, and a collision
would serve one conversation's KV to another. So the pool stores each cached
block's parent hash and token ids and verifies them on every hit; the hash
narrows the search, equality decides, and a collision degrades to a miss.

**Leaving an evicted block registered.** Freeing and eviction are different
events. Freeing drops a reference and deliberately leaves the contents intact and
findable, which is what lets the next turn pick up a prefix the previous turn
released. Eviction happens later, when that block is actually handed out for new
data, and `allocate_block` is the one place it happens -- so that is where the
registry entry is removed. A block left registered after being overwritten would
hand its new occupant's KV to whoever looked up the old prefix.

### Two deliberate limits

**Partial blocks are never shared.** A block that is not full can still gain
tokens, so its contents are not final. Reuse is therefore rounded down to a block
boundary, which is why turn 1 above reuses 96 rather than all 99 shared tokens.

**One token is always left to compute.** A fully cached prompt would give the
forward pass nothing to produce logits from, and there would be no next token to
sample.

## Barge-in

When a user interrupts, the in-flight generation is worthless and its memory is
needed by the utterance that interrupted it. So `Engine.cancel(request_id)` does
not set a flag for the next scheduling pass to notice: it runs synchronously,
outside the loop, and the blocks are in the pool before it returns.

That is what `ticks_to_free == 0` on every `CancellationRecord` means, and it is
asserted on the tick counter rather than on wall clock, because the claim is
structural rather than a timing measurement. Measured on `run_bargein.py`:

```
cancellations: 2
  t1: 0 ticks, 17 us, 2 released / 5 kept
  t3: 0 ticks, 15 us, 2 released / 5 kept
```

The second half of that line matters as much as the first. A barge-in must
release exactly the blocks the abandoned generation alone owned and leave the
shared system prompt standing -- `2 released / 5 kept` above. That is the
reference count built in stage 3, finally doing the job it was built for, and
`test_cancelling_leaves_a_shared_prefix_standing` asserts it against two
concurrent turns rather than against the demo's retained prefix alone.

A cancel racing a completion returns `None` rather than raising, because for a
voice agent that race is ordinary traffic. An id that was never submitted is a
`KeyError`, because that is a bug.

## Turns

`Turn` declares which part of a request is reusable prefix (system prompt, tools,
history) and which part is the new user segment. Prefix caching already discovers
sharing by hashing, so the declaration buys three things hashing cannot:

- **A miss becomes visible.** `declared_prefix_hit_ratio` reports reuse against
  the denominator the caller cares about. A turn that declared a 2000-token
  prefix and reused none of it has a problem; the whole-prompt ratio hides that
  behind the new segment's length.
- **The prefix can be kept resident.** `retain_prefix` pins the blocks covering a
  declared prefix so eviction cannot take them while other conversations churn
  through the pool. A retention is just a sequence that holds references and
  never runs, so it needed no new machinery -- the refcount already means
  "somebody is using this".
- **Reuse stops at the boundary** rather than speculating into the segment, which
  by construction is the part that differs.

A Turn is a constructor for a `Sequence`, not a parallel runtime type. Nothing in
the scheduler knows about turns.

## Deadlines

Each request may carry a TTFT budget. Waiting requests are ordered by deadline
rather than arrival, and a request whose deadline is close enough to be in danger
is admitted *ahead of the decode pass* -- the one place in this scheduler where
latency is deliberately bought at the cost of throughput.

Decodes keep a reservation of one token each, so however many urgent turns
arrive, the conversations already in progress still advance. A sequence that was
preempted after producing a token is admitted in the same priority pass: its TTFT
is settled, but it is committed work whose memory was taken away, and queueing it
behind fresh arrivals would reopen the starvation that newest-first eviction
exists to prevent.

A miss is **recorded, not acted on**. `deadline_missed` and `ttft_s` are set when
the first token turns out to be late, and the request still runs to completion. A
deadline already in the past still counts as urgent. Shedding doomed work would
be defensible for a voice agent -- late really is useless -- but that is a product
decision rather than this layer's.

`at_risk_margin_s` is a flat margin rather than an estimate of remaining prefill
time. Estimating that properly needs a calibrated token rate, and measuring one is
explicitly out of scope for phase 1.

Prefix caching turns out to be the main TTFT mechanism rather than a throughput
one. In `run_bargein.py` the first turn misses a 300 ms budget at 370 ms with a
cold prefix, while later turns reusing 91% of it land at around 100 ms.
