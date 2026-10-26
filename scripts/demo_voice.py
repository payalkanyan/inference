"""Stage 8 demo: a voice conversation with barge-ins.

    python scripts/demo_voice.py [out.jsonl]

A long system prompt, several short turns, and two barge-ins where a generation
is cancelled mid-stream and the replacement turn starts immediately. Conversation
history accumulates into the prefix, as it does in a real agent, so each turn
re-sends everything said so far.

Prints the per-request metrics at the end and writes every record to JSONL.
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
from vox.metrics import MetricsRecorder
from vox.sampler import SamplingParams
from vox.scheduler import SchedulerConfig
from vox.turn import Turn
from vox.weights import load_model

SYSTEM = (
    "You are Ada, the voice assistant for a dental practice. You speak aloud, so "
    "answer in one short sentence and never more than twenty words. Never use "
    "lists, markdown, or symbols that cannot be spoken. If you do not know "
    "something, say so plainly instead of guessing.\n"
    "\n"
    "Tools available to you:\n"
    "calendar.find_slots(date, duration_minutes) returns open appointment times "
    "for a given day. Use it whenever the caller asks about availability.\n"
    "calendar.book(slot_id, patient_name, phone) reserves an appointment and "
    "returns a confirmation code. Always read the code back to the caller.\n"
    "calendar.cancel(confirmation_code) releases an appointment. Confirm with the "
    "caller before calling it.\n"
    "patients.lookup(phone) returns a patient record including name, last visit, "
    "and outstanding balance. Never read a balance aloud unless asked.\n"
    "hours.today() returns opening hours for the current day, including any "
    "holiday closures.\n"
    "\n"
    "Rules: never invent an appointment time that a tool did not return. Never "
    "state a price. If the caller sounds distressed or mentions pain, offer the "
    "earliest available slot before anything else. If a tool fails, say that the "
    "system is unavailable and offer to take a message.\n"
    "\n"
    "Conversation so far:\n"
)

#: (utterance, interrupt after N ticks). None means let it finish.
SCRIPT = [
    ("Hi, what time do you close today?", None),
    ("Great. Do you have anything free tomorrow morning?", None),
    ("Tell me everything about how root canals work.", 5),      # barge-in
    ("Sorry, forget that. Book me the earliest slot.", None),
    ("Also read me the full practice history and policies.", 4),  # barge-in
    ("Never mind. Just confirm my appointment.", None),
]

TTFT_BUDGET_S = 0.40
MAX_NEW_TOKENS = 20


def encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("voice_run.jsonl")

    model_dir = resolve_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_model(model_dir, dtype=torch.float32, device="cpu")

    cache = PagedKVCache(model.cfg, num_blocks=512, block_size=16, dtype=model.dtype)
    recorder = MetricsRecorder(out_path)
    engine = Engine(
        model,
        cache,
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256),
        metrics=recorder,
    )

    system_tokens = encode(tokenizer, SYSTEM)
    print(f"system prompt: {len(system_tokens)} tokens   "
          f"TTFT budget: {TTFT_BUDGET_S * 1000:.0f} ms   pool: {cache}\n")

    history = ""          # grows as the conversation proceeds
    retained = False
    transcript: list[tuple[str, str, str]] = []

    for index, (utterance, interrupt_after) in enumerate(SCRIPT):
        request_id = f"turn{index}"
        # The reusable prefix is the system prompt plus everything said so far.
        prefix = encode(tokenizer, SYSTEM + history)
        segment = encode(tokenizer, f"Caller: {utterance}\nAda:")

        engine.add_turn(Turn(
            request_id=request_id,
            prefix_token_ids=prefix,
            segment_token_ids=segment,
            params=SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS),
            stop_token_ids=frozenset({tokenizer.eos_token_id}),
            ttft_budget_s=TTFT_BUDGET_S,
        ))
        print(f"caller: {utterance}")

        outcome, finished = "finished", None
        ticks = 0
        while engine.has_work:
            if interrupt_after is not None and ticks >= interrupt_after:
                record = engine.cancel(request_id)
                partial = tokenizer.decode(
                    engine.requests[request_id].output_token_ids, skip_special_tokens=True
                )
                print(f"  ada:  {partial.strip()!r}")
                print(f"  *** BARGE-IN *** cancelled after {record.tokens_generated} tokens; "
                      f"freed in {record.ticks_to_free} ticks / "
                      f"{record.wall_seconds * 1e6:.0f} us "
                      f"({record.blocks_released} blocks released, "
                      f"{record.blocks_retained} kept)")
                outcome = "cancelled"
                break
            for output in engine.step():
                if output.finished:
                    finished = output
            ticks += 1

        if finished is not None:
            reply = tokenizer.decode(finished.token_ids, skip_special_tokens=True).strip()
            print(f"  ada:  {reply!r}")
            print(f"        ttft {finished.ttft_s * 1000:.0f} ms"
                  f"{' MISSED' if finished.deadline_missed else ''}   "
                  f"prefix reused {finished.declared_prefix_hit_ratio:.0%}")
            # Only answers the caller actually heard become history.
            history += f"Caller: {utterance}\nAda: {reply}\n"
            transcript.append((request_id, utterance, reply))
        else:
            transcript.append((request_id, utterance, "<interrupted>"))
        print()

        # Pin the prefix once it exists, so churn cannot evict the system prompt.
        if not retained:
            pinned = engine.retain_prefix(list(system_tokens))
            if pinned:
                print(f"--- pinned {pinned} system-prompt tokens "
                      f"({engine.num_retained_blocks} blocks) ---\n")
                retained = True

    recorder.flush()

    # -- per-request metrics ------------------------------------------------
    print("=" * 96)
    print("per-request metrics\n")
    header = (f"{'request':<8} {'prompt':>6} {'gen':>4} {'queue':>7} {'ttft':>8} "
              f"{'total':>7} {'reuse':>6} {'pre':>4} {'canc':>5}  outcome")
    print(header)
    print("-" * len(header))
    for m in engine.metrics.requests:
        def ms(value):
            return "     -" if value is None else f"{value * 1000:5.0f}"
        print(f"{m.request_id:<8} {m.prompt_tokens:>6} {m.generated_tokens:>4} "
              f"{ms(m.queue_s):>7} "
              f"{ms(m.ttft_s)}{'!' if m.deadline_missed else ' '} "
              f"{ms(m.total_s):>7} "
              f"{m.declared_prefix_hit_ratio:>5.0%} "
              f"{m.num_preemptions:>4} {str(m.was_cancelled):>5}  {m.finish_reason}")
    print(f"\n(times in ms; ! marks a missed {TTFT_BUDGET_S * 1000:.0f} ms TTFT budget; "
          f"reuse is of the declared prefix)")

    # -- cancellations ------------------------------------------------------
    print("\n" + "=" * 96)
    print("cancellations -- cancel() call to blocks freed\n")
    print(f"{'request':<8} {'ticks':>6} {'wall (us)':>10} {'released':>9} {'kept':>5} {'tokens':>7}")
    print("-" * 50)
    for r in engine.metrics.cancellations:
        print(f"{r.request_id:<8} {r.ticks_to_free:>6} {r.wall_seconds * 1e6:>10.0f} "
              f"{r.blocks_released:>9} {r.blocks_retained:>5} {r.tokens_generated:>7}")

    # -- engine totals ------------------------------------------------------
    series = engine.metrics.iterations
    print("\n" + "=" * 96)
    print("engine\n")
    print(f"  iterations            {engine.num_iterations}")
    print(f"  forward time          {sum(m.forward_s for m in series):.2f} s")
    print(f"  batch size            max {max(m.batch_size for m in series)}, "
          f"mean {sum(m.batch_size for m in series) / len(series):.2f}")
    print(f"  block utilisation     max {max(m.block_utilization for m in series):.1%}, "
          f"mean {sum(m.block_utilization for m in series) / len(series):.1%}")
    print(f"  prompt tokens reused  {engine.num_cached_tokens_reused}")
    print(f"  preemptions           {engine.num_preemptions}")
    print(f"  deadline misses       {engine.num_deadline_misses}")
    print(f"  urgent admissions     {engine.num_urgent_admissions}")
    print(f"  blocks referenced     {cache.pool.num_used_blocks} (the pinned system prompt)")

    engine.release_prefix(list(system_tokens))
    recorder.close()
    print(f"\n  after release         {cache.pool.num_used_blocks} blocks referenced")
    print(f"\nwrote {len(series) + len(engine.metrics.requests) + len(engine.metrics.cancellations)}"
          f" records to {out_path}")
