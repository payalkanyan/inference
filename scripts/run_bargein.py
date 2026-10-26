"""Stage 7 demo: barge-in, declared prefixes, and TTFT deadlines.

    python scripts/run_bargein.py

A short conversation in which two generations are cancelled mid-stream and
replaced immediately. Reports, per cancellation, how many engine ticks and how
much wall clock passed between cancel() being called and the blocks being back
in the pool -- and how many blocks were *kept* because the next turn still
references them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import vox without installing

import torch
from transformers import AutoTokenizer

from vox.cache import PagedKVCache
from vox.engine import Engine
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.turn import Turn
from vox.weights import load_model

SYSTEM = (
    "You are a concise voice assistant for a scheduling service. Answer in one "
    "short sentence and never more than twenty words. Speak plainly and do not "
    "use lists. You have access to a calendar tool that can read and create "
    "events, a weather tool that reports current conditions for a city, and a "
    "contacts tool that looks up phone numbers by name. Never invent facts, and "
    "if a tool is unavailable say so rather than guessing. User question: "
)

#: (id, utterance, ticks to run before the user interrupts). None = run to the end.
SCRIPT = [
    ("t0", "What is the capital of France?", None),
    ("t1", "Tell me everything about the history of Rome.", 4),   # barge-in
    ("t2", "Actually, just name a colour.", None),
    ("t3", "Describe every planet in the solar system in detail.", 3),  # barge-in
    ("t4", "Never mind. What is two plus two?", None),
]
TTFT_BUDGET_S = 0.30
MAX_NEW_TOKENS = 16

if __name__ == "__main__":
    model_dir = resolve_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_model(model_dir, dtype=torch.float32, device="cpu")

    prefix = tuple(tokenizer(SYSTEM, add_special_tokens=False)["input_ids"])
    cache = PagedKVCache(model.cfg, num_blocks=256, block_size=16, dtype=model.dtype)
    engine = Engine(model, cache, SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256))

    print(f"declared prefix: {len(prefix)} tokens   TTFT budget: {TTFT_BUDGET_S * 1000:.0f} ms")
    print(f"pool: {cache}\n")

    finished = {}
    retained = False
    for request_id, utterance, interrupt_after in SCRIPT:
        # Once the prefix has been computed once, pin it. This is what declaring
        # a prefix buys: the blocks are referenced, so eviction cannot take them
        # between turns -- and a barge-in cannot free them either.
        if not retained and finished:
            pinned = engine.retain_prefix(list(prefix))
            print(f"--- retained {pinned} prefix tokens "
                  f"({engine.num_retained_blocks} blocks) ---\n")
            retained = True

        turn = Turn(
            request_id=request_id,
            prefix_token_ids=prefix,
            segment_token_ids=tuple(tokenizer(utterance, add_special_tokens=False)["input_ids"]),
            params=SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS),
            stop_token_ids=frozenset({tokenizer.eos_token_id}),
            ttft_budget_s=TTFT_BUDGET_S,
        )
        engine.add_turn(turn)
        print(f"[{request_id}] user: {utterance!r}")

        ticks = 0
        while engine.has_work:
            if interrupt_after is not None and ticks >= interrupt_after:
                record = engine.cancel(request_id)
                partial = tokenizer.decode(
                    engine.requests[request_id].output_token_ids, skip_special_tokens=True
                )
                print(f"         ...BARGE-IN after {record.tokens_generated} tokens: {partial!r}")
                print(f"         freed in {record.ticks_to_free} ticks / "
                      f"{record.wall_seconds * 1e6:.0f} us   "
                      f"released {record.blocks_released} blocks, "
                      f"kept {record.blocks_retained} still shared")
                break
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output
            ticks += 1

        if request_id in finished:
            output = finished[request_id]
            print(f"         assistant: "
                  f"{tokenizer.decode(output.token_ids, skip_special_tokens=True)!r}")
            print(f"         ttft {output.ttft_s * 1000:.0f} ms"
                  f"{' (MISSED)' if output.deadline_missed else ''}   "
                  f"declared prefix reused {output.declared_prefix_hit_ratio:.0%}")
        print()

    print("-" * 72)
    print(f"cancellations: {len(engine.cancellations)}")
    for record in engine.cancellations:
        print(f"  {record.request_id}: {record.ticks_to_free} ticks, "
              f"{record.wall_seconds * 1e6:.0f} us, "
              f"{record.blocks_released} released / {record.blocks_retained} kept")
    print(f"deadline misses: {engine.num_deadline_misses}   "
          f"urgent admissions: {engine.num_urgent_admissions}")
    print(f"prompt tokens reused from the prefix cache: {engine.num_cached_tokens_reused}")
    print(f"blocks still referenced: {cache.pool.num_used_blocks} "
          f"(the retained prefix)   cached and reusable: {cache.pool.num_cached_blocks}")
    engine.release_prefix(list(prefix))
    print(f"after releasing the retained prefix: "
          f"{cache.pool.num_used_blocks} blocks referenced")
