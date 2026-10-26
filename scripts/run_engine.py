"""Stage 4/5 demo: requests sharing the engine, joining mid-flight.

    python scripts/run_engine.py                 # roomy pool, whole prefills
    python scripts/run_engine.py squeeze         # tight budget and pool

Prints the batch composition each iteration so the scheduling is visible:
prefills appearing, decodes accumulating, finished requests dropping out, and
under `squeeze`, prompts arriving in chunks and sequences being preempted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import vox without installing

import torch
from transformers import AutoTokenizer

from vox.cache import PagedKVCache
from vox.engine import Engine
from vox.generate import generate_naive
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.weights import load_model

PROMPTS = {
    "france": "The capital of France is",
    "sky": "The sky is",
    "count": "Count from one to ten:",
    "code": "def add(a, b):",
}
#: A tight pool and budget, to force chunked prefill and preemption.
SQUEEZE = dict(num_blocks=8, block_size=4, max_num_batched_tokens=6, max_num_seqs=8)
ROOMY = dict(num_blocks=256, block_size=16, max_num_batched_tokens=256, max_num_seqs=8)
#: Which iteration each request shows up at, to exercise mid-flight admission.
ARRIVES_AT = {"france": 0, "sky": 0, "count": 3, "code": 6}
MAX_NEW_TOKENS = 20

if __name__ == "__main__":
    model_dir = resolve_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_model(model_dir, dtype=torch.float32, device="cpu")
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)

    encoded = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in PROMPTS.items()}

    squeeze = len(sys.argv) > 1 and sys.argv[1] == "squeeze"
    settings = SQUEEZE if squeeze else ROOMY
    cache = PagedKVCache(
        model.cfg,
        num_blocks=settings["num_blocks"],
        block_size=settings["block_size"],
        dtype=model.dtype,
    )
    engine = Engine(model, cache, SchedulerConfig(
        max_num_seqs=settings["max_num_seqs"],
        max_num_batched_tokens=settings["max_num_batched_tokens"],
    ))

    print(f"mode: {'squeeze' if squeeze else 'roomy'}   budget: "
          f"{settings['max_num_batched_tokens']} tokens/iter   pool: {cache}\n")
    print(f"{'iter':>4}  {'seqs':>4}  {'tok':>4}  {'blk':>4}  composition")
    print("-" * 76)

    collected: dict[str, list[int]] = {}
    for iteration in range(300):
        for name, when in ARRIVES_AT.items():
            if when == iteration:
                engine.add_request(name, encoded[name], params)
        if not engine.has_work:
            break

        scheduled = engine.scheduler.schedule()
        if scheduled.is_empty:
            continue
        # Re-run the body of step() so the composition can be printed. The
        # scheduler has already committed this decision.
        composition = ", ".join(
            f"{s.request_id}:{kind}" + (f"x{q}" if q > 1 else "")
            for s, kind, q in zip(scheduled.sequences, scheduled.kinds, scheduled.query_lens)
        )
        if scheduled.preempted:
            composition += "   PREEMPTED: " + ", ".join(
                s.request_id for s in scheduled.preempted
            )
        print(f"{iteration:>4}  {len(scheduled.sequences):>4}  "
              f"{scheduled.num_query_tokens:>4}  "
              f"{cache.pool.num_used_blocks:>4}  {composition}")

        for output in engine.run_scheduled(scheduled):
            collected[output.request_id] = output.token_ids

    print("-" * 76)
    print(f"\niterations: {engine.num_iterations}   "
          f"preemptions: {engine.num_preemptions}   "
          f"catch-up chunks: {engine.num_catch_up_chunks}")
    print(f"all blocks returned: {cache.pool.num_used_blocks == 0}\n")

    for name in PROMPTS:
        text = tokenizer.decode(collected[name], skip_special_tokens=True)
        print(f"[{name}] {PROMPTS[name]!r}\n    -> {text!r}")

    print("\nchecking each against an unbatched run ...")
    ok = True
    for name, ids in encoded.items():
        expected = generate_naive(model, ids, params).token_ids
        same = collected[name] == expected
        ok &= same
        print(f"  {name:<8} {'OK' if same else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)
    print("\nOK: batching changed nothing")
