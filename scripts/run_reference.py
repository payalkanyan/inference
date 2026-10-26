"""Run both single-sequence reference paths and check they agree.

    python scripts/run_reference.py "What is the capital of France?" 48
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import vox without installing

import torch
from transformers import AutoTokenizer

from vox.generate import generate_cached, generate_naive, generate_paged
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.weights import load_model

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    model_dir = resolve_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    t0 = time.perf_counter()
    model = load_model(model_dir, dtype=torch.float32, device="cpu")
    print(f"loaded in {time.perf_counter() - t0:.2f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    # transformers 5 returns a BatchEncoding; older versions returned a bare list.
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    stop = [tokenizer.eos_token_id]

    results = {}
    paths = [("naive", generate_naive), ("cached", generate_cached), ("paged", generate_paged)]
    for name, fn in paths:
        t0 = time.perf_counter()
        results[name] = (fn(model, token_ids, params, stop_token_ids=stop),
                         time.perf_counter() - t0)

    print(f"\nprompt ({len(token_ids)} tokens): {prompt}")
    for name, (result, elapsed) in results.items():
        print(f"\n[{name}] {len(result.token_ids)} tokens, {result.finish_reason}, "
              f"{elapsed:.2f}s ({len(result.token_ids) / elapsed:.1f} tok/s)")
        print(f"  {tokenizer.decode(result.token_ids, skip_special_tokens=True)}")

    expected = results["naive"][0].token_ids
    disagreed = {n: r.token_ids for n, (r, _) in results.items() if r.token_ids != expected}
    if not disagreed:
        print(f"\nOK: all {len(results)} paths produced identical tokens "
              f"({len(expected)} of them)")
    else:
        print("\nMISMATCH against the naive reference:")
        print(f"  naive: {expected}")
        for name, ids in disagreed.items():
            print(f"  {name}: {ids}")
        raise SystemExit(1)
