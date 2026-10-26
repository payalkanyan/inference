"""Stage 6 demo: a multi-turn conversation re-sending the same system prompt.

    python scripts/run_prefix.py

Runs the same turns twice, with prefix caching off and on, and reports how much
of each prompt did not have to be prefilled. Then checks that both runs produced
identical tokens.
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
from vox.weights import load_model

SYSTEM = (
    "You are a concise voice assistant for a scheduling service. Answer in one "
    "short sentence and never more than twenty words. Speak plainly and do not "
    "use lists. You have access to a calendar tool that can read and create "
    "events, a weather tool that reports current conditions for a city, and a "
    "contacts tool that looks up phone numbers by name. Never invent facts, and "
    "if a tool is unavailable say so rather than guessing. Previous turns of this "
    "conversation are included above for context. User question: "
)
TURNS = [
    "What is the capital of France?",
    "And of Italy?",
    "Name a colour.",
    "What about Spain?",
]
MAX_NEW_TOKENS = 12


def run(model, turns, *, enabled: bool):
    cache = PagedKVCache(
        model.cfg, num_blocks=512, block_size=16,
        dtype=model.dtype, enable_prefix_caching=enabled,
    )
    engine = Engine(model, cache, SchedulerConfig())
    results = []
    # One turn at a time, as a conversation actually arrives.
    for index, ids in enumerate(turns):
        engine.add_request(f"turn{index}", ids, SamplingParams(0.0, MAX_NEW_TOKENS))
        results.append(engine.run_until_done(max_steps=500)[f"turn{index}"])
    return engine, cache, results


if __name__ == "__main__":
    model_dir = resolve_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_model(model_dir, dtype=torch.float32, device="cpu")

    turns = [tokenizer(SYSTEM + t, add_special_tokens=False)["input_ids"] for t in TURNS]
    system_len = len(tokenizer(SYSTEM, add_special_tokens=False)["input_ids"])
    print(f"shared system prompt: {system_len} tokens; "
          f"each turn adds {min(len(t) for t in turns) - system_len}-"
          f"{max(len(t) for t in turns) - system_len} more\n")

    _, _, without = run(model, turns, enabled=False)
    engine, cache, with_cache = run(model, turns, enabled=True)

    print(f"{'turn':>5}  {'prompt':>6}  {'prefilled':>9}  {'reused':>6}  {'hit':>5}")
    print("-" * 44)
    for index, output in enumerate(with_cache):
        prompt = len(output.prompt_token_ids)
        print(f"{index:>5}  {prompt:>6}  {prompt - output.num_cached_tokens:>9}  "
              f"{output.num_cached_tokens:>6}  {output.prefix_hit_ratio:>4.0%}")
    print("-" * 44)

    total = sum(len(t) for t in turns)
    reused = sum(o.num_cached_tokens for o in with_cache)
    print(f"\n{reused} of {total} prompt tokens reused ({reused / total:.0%}) "
          f"across {len(turns)} turns")
    print(f"blocks published: {engine.num_blocks_published}   "
          f"still cached: {cache.pool.num_cached_blocks}   "
          f"referenced: {cache.pool.num_used_blocks}")
    print("\nNote: this saves prefill work, not decode iterations. The win is in "
          "\ntime-to-first-token, which stage 7 starts scheduling against.\n")

    for index, output in enumerate(with_cache):
        print(f"[turn{index}] {TURNS[index]!r}")
        print(f"    -> {tokenizer.decode(output.token_ids, skip_special_tokens=True)!r}")

    mismatched = [
        i for i, (a, b) in enumerate(zip(with_cache, without)) if a.token_ids != b.token_ids
    ]
    if mismatched:
        print(f"\nMISMATCH on turns {mismatched}")
        raise SystemExit(1)
    print("\nOK: prefix caching changed nothing")
