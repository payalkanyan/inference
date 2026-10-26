# vox

A from-scratch LLM inference engine in Python, specialised for interruptible
voice-agent traffic.

Most from-scratch engines optimise throughput on independent completion requests.
This one optimises three things that voice agents need and general text serving
does not:

1. **Hard time-to-first-token deadlines.** A few hundred milliseconds. Late is
   useless, so the scheduler is willing to spend throughput to hit a deadline.
2. **Short decodes on a long, mostly-unchanging prefix.** System prompt, tool
   definitions and history are re-sent every turn, so the prefix is shared
   explicitly rather than discovered by accident.
3. **Frequent mid-decode cancellation.** When the user barges in, the in-flight
   generation is worthless and its memory is needed *now* by the utterance that
   interrupted it.

The standard machinery — paged KV cache, continuous batching, chunked prefill,
preemption, prefix caching — is built properly first, and the specialisation sits
on top of it.

**Model:** Qwen2.5-0.5B-Instruct (494M parameters), HuggingFace weights loaded
manually. The forward pass is implemented here: RMSNorm, RoPE, grouped-query
attention and SwiGLU, in plain PyTorch. `transformers` is a dependency only for
its tokenizer and as the correctness reference.

**Phase 1 scope.** A correct, working engine with built-in instrumentation.
Benchmarking and comparison studies are explicitly out of scope — see
[What this is not](#what-this-is-not).

---

## Status

257 tests pass. The forward pass is **token-identical to HuggingFace
transformers** under greedy decoding, and on float32 CPU the prefill logits are
bit-identical. Output is unchanged by every layer of machinery added on top:
caching, paging, batching, chunking, preemption, prefix reuse, and another
request being cancelled alongside it.

| Component | Status |
|---|---|
| Model forward pass + naive generation, token-identical to HF | done |
| Contiguous KV cache | done |
| Paged KV cache + block pool, refcounting, LRU eviction | done |
| Scheduler + continuous batching | done |
| Chunked prefill + recompute-based preemption | done |
| Prefix caching across separate requests | done |
| Cancellation, Turn abstraction, deadline-aware admission | done |
| Instrumentation + voice demo | done |

Verified on Python 3.12.3, torch 2.14.1+cpu, transformers 5.20.2, CPU float32.

---

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python transformers safetensors pytest

.venv/bin/python scripts/download_model.py        # ~1 GB into the HF cache
.venv/bin/python -m pytest tests/ -q              # the correctness gates
```

Set `VOX_MODEL_DIR` to use weights you already have instead of the HF cache.

CUDA is an optional fast path, not a requirement: everything runs on CPU in
float32, which is the configuration all the numbers here come from.

### Demos

```bash
.venv/bin/python scripts/demo_voice.py            # the one to run first
.venv/bin/python scripts/run_bargein.py           # cancellation, in isolation
.venv/bin/python scripts/run_prefix.py            # prefix reuse, with and without
.venv/bin/python scripts/run_engine.py [squeeze]  # continuous batching, iteration by iteration
.venv/bin/python scripts/run_reference.py "..."   # all four generation paths agree
```

`demo_voice.py` simulates a dental-practice voice agent: a 234-token system
prompt, six turns with accumulating history, and two barge-ins. It prints the
per-request metrics and writes every record to `voice_run.jsonl`.

```
caller: Tell me everything about how root canals work.
  ada:  "I'm sorry, but"
  *** BARGE-IN *** cancelled after 5 tokens; freed in 0 ticks / 19 us
                   (7 blocks released, 14 kept)

request  prompt  gen   queue     ttft   total  reuse  pre  canc  outcome
turn0       247   20       0   784!    2027    0%    0 False  length
turn1       282   20       0   123     1373   96%    0 False  length
turn2       317    5       0   130      424   95%    0  True  cancelled
turn3       318   20       0   143     1545   95%    0 False  length
turn4       352    4       0   107      328   99%    0  True  cancelled
turn5       350   20       0   103     1515   99%    0 False  length
```

The model's actual answers are poor — it is a 0.5B model being fed a raw prompt
rather than its chat template. The demo exercises the engine, not the model.

---

## Architecture

```
vox/
  config.py      Qwen2Config, parsed from HF config.json
  weights.py     safetensors -> load_state_dict
  hub.py         locating weights on disk
  layers.py      RMSNorm, RoPE, GQA attention, SwiGLU
  model.py       the 24-layer decoder stack
  state.py       AttentionState: what attention needs beyond hidden states
  cache.py       ContiguousKVCache and PagedKVCache
  blocks.py      block pool: refcounts, LRU free list, prefix registry
  prefix.py      chained block hashing
  sequence.py    per-sequence state: tokens, block table, timings, deadline
  turn.py        Turn: a declared reusable prefix plus a new segment
  batch.py       flattening several sequences into one forward pass
  scheduler.py   iteration-level scheduling, chunking, preemption, deadlines
  engine.py      the loop, and cancel()
  sampler.py     greedy and temperature sampling
  metrics.py     the three record types and the JSONL writer
  generate.py    four single-sequence reference paths, used as test oracles
```

2,900 lines of engine, 4,000 lines of tests.

### One iteration

`Engine.step()` is the whole loop:

1. **Schedule.** `Scheduler.schedule()` returns a list of sequences and how many
   tokens each contributes. It preempts if it must, admits urgent requests,
   allocates blocks, and consults the prefix cache. Nothing downstream makes a
   policy decision.
2. **Build.** `build_batch()` flattens those sequences into one `[1, T]` row of
   tokens, one `[K]` run of key slots, and a `[T, K]` mask.
3. **Forward.** One call into the model. Attention writes the new KV through the
   slot indices and reads the whole prefix back through them.
4. **Sample.** One row of logits per sequence whose tokens are now all computed.
   A sequence mid-prefill-chunk produces nothing.
5. **Post-process.** Append tokens, publish any block that just became full,
   retire anything that hit a stop token or its budget, record the iteration.

`cancel()` is deliberately *not* part of this loop. See
[Barge-in](#barge-in).

---

## Components

### The forward pass

`layers.py` and `model.py` implement Qwen2 directly. Module and parameter names
mirror HuggingFace's, so weight loading is a plain `load_state_dict` rather than
a hand-maintained mapping table — one fewer place for a silent transposition bug.
`weights.py` raises on any missing or unexpected key rather than loading
partially.

Three details decide whether this is correct, and all three produce *fluent,
wrong* output when they are not:

- **RoPE uses the half-split (`rotate_half`) convention**, not pairwise
  interleaving, with `theta = 1e6`.
- **q/k/v projections carry a bias; o_proj does not.** This is Qwen2-specific.
- **GQA maps query head `i` to KV head `i // 14`** — 14 query heads over 2 KV
  heads.

`tests/test_layers.py` pins each of them directly, including attention against a
dense per-head reference with an explicit causal mask, and the property the whole
cache design rests on: that `q·k` after RoPE depends only on *relative* position.

### Paged KV cache

The contiguous cache (`ContiguousKVCache`) is kept as a reference. It is also
deliberately the wrong shape for this workload: it reserves `max_seq_len` per
sequence whether or not those tokens are used, and two turns sharing a system
prompt cannot share a single byte.

`PagedKVCache` is one fixed pool of equally sized blocks shared by every
sequence. Storage is `[layers, blocks, block_size, kv_heads, head_dim]`, but all
addressing goes through a flat *slot* index over the same memory:

```
slot = block_id * block_size + offset_within_block
```

One slot vector is computed per forward pass and reused by all 24 layers:
`write_slots` for the new tokens, `read_slots` for the prefix to attend over.
Reads gather and writes scatter, because without custom kernels that is how you
make a discontiguous sequence look contiguous to SDPA.

`blocks.py` is pure bookkeeping and never touches a tensor. It owns reference
counts, an LRU free list, and the prefix registry. **Freeing and eviction are
different events:** freeing drops a reference and deliberately leaves the
contents findable; eviction happens later, when the block is actually handed out
for new data, and `allocate_block` is the single place it happens.

### Batching

A batch's sequences disagree about everything: how many tokens each contributes,
how long its prefix is, which blocks hold it. Padding to a common shape makes the
cost the *maximum* rather than the sum — one 500-token prefill sharing a batch
with eight 1-token decodes would compute 4,500 query positions to produce 9
tokens.

So `batch.py` flattens. Queries become one row of `T = sum(query_len)`, keys one
run of `K = sum(kv_len)`, and sequences are held apart by the mask:

```python
allowed[t, k] = (seq_of_query[t] == seq_of_key[k]) & (key_pos[k] <= query_pos[t])
```

One term for "same sequence", one for causality. That covers prefill, decode,
mixed batches and chunk continuation with no special cases — which is why
batching needed no changes at all to `layers.py` or `cache.py`. The tensor batch
dimension stays 1, so batching happens along the token dimension, and the paged
cache already addressed memory by slot.

The cost is that the `T × K` mask is **materialised**. A varlen attention kernel
would read the sequence boundaries directly and never build it. That is the main
reason the token budget exists.

### Scheduling

Each iteration, in order:

1. **Preempt** until every running sequence can advance by at least one token.
2. **Admit urgent requests** — those whose TTFT deadline is in danger, and those
   resuming after preemption — ahead of everything else.
3. **Decodes**, one token each.
4. **Catch-up chunks** for sequences still building KV.
5. **Admit everything else**, in deadline order.

Decodes keep a reservation of both budget *and* blocks, so no amount of urgent or
chunked work can starve a conversation already in progress.

**Chunked prefill** means a long prompt overlaps with everyone else's decoding
instead of blocking an iteration. It also means **a prefill can shrink to fit
whatever memory is free, and a decode cannot** — a decode's minimum unit is one
token. That asymmetry is why chunking and preemption belong together: chunking
makes preemption rare, and preemption covers what chunking can't.

**Preemption is recompute-based.** Blocks are released and the KV is forgotten,
to be rebuilt from tokens the sequence still holds. Nothing is copied anywhere,
so preempting is as cheap as freeing; the cost is paid in recomputed FLOPs on
resume. The victim is always the **most recently admitted** sequence, which is
what makes progress guaranteed: the oldest running sequence is never evicted, so
it always finishes and frees memory for everyone behind it.

That also means the newest sequence bears the whole cost. Under sustained
pressure it can be evicted and recomputed repeatedly while an older one drains —
visible in `run_engine.py squeeze`, where a deliberately undersized pool produces
12 preemptions and 60 iterations for work that takes 26 when memory is adequate.
That is a real characteristic of newest-first eviction, and the honest fix is a
larger pool rather than a cleverer policy.

### Prefix caching

Every turn re-sends the same system prompt, tools and history, and changes only a
short tail. Recomputing that prefix is the single largest avoidable cost in the
workload, so blocks are shared between *separate requests*, not just within one.

Sharing is keyed on a **chained** hash:

```
identity(0) = hash(None,            tokens[0:B])
identity(i) = hash(identity(i-1),   tokens[i*B:(i+1)*B])
```

Chaining is the correctness condition, not an optimisation. A block's keys and
values are a function of everything before it, so the same 16 tokens at a
different offset hold entirely different KV; keying on content alone would share
them and produce fluent, wrong output.

On admission the scheduler walks the chain from position 0 and stops at the first
miss — a hit further along would be meaningless. Blocks that hit are adopted with
an `incref`, `num_computed_tokens` jumps forward, and the forward pass simply
starts later. Again, nothing in the attention path changed: write slots already
covered only the new tokens while read slots covered the whole prefix.

**Three ways this can be silently wrong**, each with a test named after it:

| Failure | Guard |
|---|---|
| Sharing across different prefixes | chained hashing |
| Accepting a hash collision | every hit is verified against the stored parent hash and token ids; the hash narrows the search, equality decides |
| Leaving an evicted block registered | `allocate_block` is the only place eviction happens, and it deregisters |

**Two deliberate limits.** Partial blocks are never shared, because a block that
is not full can still gain tokens — so reuse rounds down to a block boundary. And
one token is always left to compute, because a fully cached prompt would give the
forward pass nothing to produce logits from.

Prefix caching saves prefill *work*, not decode iterations. The win is in
time-to-first-token: in `demo_voice.py`, the first turn misses a 400 ms budget at
784 ms with a cold prefix, while later turns reusing 95–99% of it land at
103–143 ms.

---

## The specialisation

### Barge-in

When a user interrupts, the in-flight generation is worthless and its memory is
needed by the utterance that interrupted it. So `Engine.cancel(request_id)` does
not set a flag for the next scheduling pass to notice. It runs **synchronously,
outside the loop**, and the blocks are in the pool before it returns.

That is what `ticks_to_free == 0` means on every `CancellationRecord`, and it is
asserted on the tick counter rather than on wall clock because the claim is
structural rather than a timing measurement. Measured in `demo_voice.py`:

```
request   ticks  wall (us)  released  kept  tokens
turn2         0         19         7    14       5
turn4         0         22         9    14       4
```

The second half of that line matters as much as the first. A barge-in must
release exactly the blocks the abandoned generation alone owned, and leave the
shared system prompt standing — `7 released, 14 kept` above. That is the
reference count doing the job it was built for, and
`test_cancelling_leaves_a_shared_prefix_standing` asserts it against two
genuinely concurrent turns rather than relying on the demo's pinned prefix.

A cancel racing a completion returns `None` rather than raising, because for a
voice agent that race is ordinary traffic. An id that was never submitted is a
`KeyError`, because that is a bug.

### Turns

`Turn` declares which part of a request is reusable prefix and which part is the
new user segment. Prefix caching already discovers sharing by hashing, so the
declaration buys three things hashing cannot:

- **A miss becomes visible.** `declared_prefix_hit_ratio` reports reuse against
  the denominator the caller cares about. A turn that declared a 2,000-token
  prefix and reused none of it has a problem; the whole-prompt ratio hides that
  behind the new segment's length.
- **The prefix can be kept resident.** `retain_prefix` pins the blocks covering a
  declared prefix so eviction cannot take them while other conversations churn
  through the pool. A retention is just a sequence that holds references and
  never runs — the refcount already means "somebody is using this".
- **Reuse stops at the boundary** rather than speculating into the segment, which
  by construction is the part that differs.

A Turn is a constructor for a `Sequence`, not a parallel runtime type. Nothing in
the scheduler knows about turns.

### Deadlines

Each request may carry a TTFT budget. Waiting requests are ordered by deadline
rather than arrival, and a request whose deadline is close enough to be in danger
is admitted *ahead of the decode pass* — the one place in this scheduler where
latency is deliberately bought at the cost of throughput.

A sequence preempted after producing a token is admitted in the same priority
pass. Its TTFT is settled, but it is committed work whose memory was taken away,
and queueing it behind fresh arrivals would reopen the starvation that
newest-first eviction exists to prevent.

A miss is **recorded, not acted on**. `deadline_missed` and `ttft_s` are set when
the first token turns out to be late, and the request still runs to completion. A
deadline already in the past still counts as urgent.

---

## Correctness

The gate for the whole project is `tests/test_hf_parity.py`:
token-identical greedy output against HuggingFace transformers, on float32 CPU,
with both sides recomputing the full sequence so no cache implementation on
either side can muddy the comparison.

It is parametrised over **every** generation path, so each new one is re-checked
rather than self-certified once:

| Path | What it is |
|---|---|
| `generate_naive` | no cache at all; full recompute every step |
| `generate_cached` | contiguous cache, one buffer per sequence |
| `generate_paged` | paged pool, block table |
| `generate_engine` | one request through the real scheduler and engine |

Only the last is the real thing. The first three stay in the tree as oracles —
`generate_naive` in particular is the simplest thing that can possibly be
correct, which is exactly what makes it useful.

Three principles the test suite is written to:

**Assert properties, not thresholds.** An early prefix-caching test asserted
`hit_ratio > 0.5` and failed at 48%. The engine was right — only the first block
was fully shared. Retuning the number would have made the test meaningless, so it
now asserts the actual property: reuse equals the longest shared prefix, floored
to a block boundary, capped by the one-token rule.

**A test of a mechanism must prove the mechanism fired.** "Preemption doesn't
change the output" passes trivially if the pool was quietly large enough. So the
engine carries `num_preemptions` and `num_catch_up_chunks`, and those tests
assert them: `assert engine.num_preemptions > 0, "the pool was too big"`.

**Construct the hostile case rather than hoping for it.** On a fresh pool a block
table is `[0, 1, 2, ...]` and the slot indirection is indistinguishable from plain
offsets — a broken implementation passes. So `test_a_fragmented_pool_gives_the_same_output`
drains the pool and selectively releases blocks to force a scattered table like
`[9, 2, 14, 5, ...]`. Likewise `test_a_forced_collision_is_rejected_rather_than_served`
builds the hash collision explicitly.

### What bit-identity does and does not survive

A single sequence through the flattened batch path **is** bit-identical to the
stage-1 reference, so the explicit mask costs nothing on its own. Batched with
others it deviates by around `3e-5` and cannot be bit-identical: attention
reduces over every key in the batch, so a sequence batched with others is reduced
over a wider tensor. Same values, different accumulation. Both are pinned as
separate tests, which is what distinguishes "the mask is wrong" from "batching
moved the arithmetic".

---

## Instrumentation

Three record types, all written to one JSONL file with a `type` field.

**`request`** — one per finished or cancelled request:

```json
{"type": "request", "request_id": "turn0", "prompt_tokens": 247,
 "generated_tokens": 20, "queue_s": 2.26e-05, "ttft_s": 0.7835,
 "total_s": 2.0268, "ttft_budget_s": 0.4, "deadline_missed": true,
 "prefix_cached_tokens": 0, "prefix_hit_ratio": 0.0,
 "declared_prefix_len": 234, "declared_prefix_hit_ratio": 0.0,
 "num_preemptions": 0, "was_preempted": false, "was_cancelled": false,
 "finish_reason": "length"}
```

**`iteration`** — one per tick; the batch-size and pool-utilisation time series:

```json
{"type": "iteration", "tick": 1, "elapsed_s": 0.785, "forward_s": 0.782,
 "batch_size": 1, "query_tokens": 247, "num_prefill_chunks": 1,
 "num_decodes": 0, "blocks_used": 16, "blocks_free": 496,
 "block_utilization": 0.031, "cached_blocks": 15, "preempted": 0,
 "urgently_admitted": 0}
```

**`cancellation`** — the headline metric:

```json
{"type": "cancellation", "request_id": "turn2", "tick_at_cancel": 45,
 "tick_at_free": 45, "wall_seconds": 1.888e-05, "blocks_held": 21,
 "blocks_released": 7, "tokens_generated": 5, "had_first_token": true,
 "ticks_to_free": 0, "blocks_retained": 14}
```

Timings that never happened are `null`, not `0.0` — a request cancelled during
prefill has no TTFT, and reporting zero is a lie that averages badly.

Records are written as they arrive, so an interrupted run still leaves usable
data. `MetricsRecorder()` with no path collects in memory only, which is the
engine's default.

Phase 1 writes the numbers and stops. No plots, no dashboards, no analysis.

---

## What this is not

**It is not fast, and no comparison to any other engine is claimed or implied.**
No benchmark against vLLM, TensorRT-LLM, SGLang or anything else was run, and
none should be inferred from anything here. The timings in this README come from
one CPU in float32 and exist to show that a mechanism works, not how quickly.

Specifically absent, and deliberately so:

- **No custom kernels.** No Triton, no CUDA kernels, no CUDA graphs. Everything
  goes through `torch.nn.functional.scaled_dot_product_attention`.
- **The `T × K` attention mask is materialised.** A varlen kernel would read
  sequence boundaries directly. This is the single largest piece of avoidable
  work in the engine.
- **No tensor parallelism**, no multi-GPU, no pipeline parallelism.
- **No quantisation** and **no speculative decoding**.
- **No HTTP server** and no OpenAI-compatible API. It is a library.
- **Sampling is greedy and temperature only.** No top-p, top-k, repetition
  penalty, logit bias, beam search or structured output.
- **No streaming interface.** `step()` returns a snapshot per scheduled request,
  but there is no token callback or async iterator.
- **The tokenizer is HuggingFace's.** Implementing BPE was never the point.

Known limitations that are real rather than merely unimplemented:

- **Single-threaded.** `cancel()` is synchronous and assumes no `step()` is in
  progress. A threaded or async embedding would need a lock around both.
- **`engine.requests` grows without bound.** Every request ever seen is retained
  so `cancel()` can find it and metrics can be emitted after it leaves. A
  long-running process needs a retention policy.
- **Prefix hashes are process-local.** `hash()` on a tuple is not stable across
  processes, so a cache cannot be shared or persisted. Hits are verified against
  stored tokens, so this is a portability limit, not a correctness one.
- **`at_risk_margin_s` is a flat margin**, not an estimate of remaining prefill
  time. Estimating that properly needs a calibrated token rate, and measuring one
  is out of scope for phase 1.
- **Preemption can thrash** under a sustained undersized pool, as described in
  [Scheduling](#scheduling). There is no anti-thrash heuristic and no
  swap-to-host alternative to recompute.
- **A request that outgrows the pool mid-generation raises.** A prompt too large
  for an empty pool is refused at admission; one that grows past it while
  decoding fails loudly rather than livelocking on a recompute it can never
  finish. Loud, but still a failure.
- **Deadline misses are recorded, not avoided.** Nothing sheds doomed work, even
  though "late is useless" arguably argues for it. That is a product decision
  that was left out rather than guessed at.
- **One model architecture.** `config.py` refuses anything that is not
  `Qwen2ForCausalLM` with `silu` and unscaled RoPE, because the alternative is
  decoding subtly wrong rather than failing.
