"""Single-sequence reference generation paths.

Neither of these is the finished engine -- from stage 4 onwards real requests go
through the scheduler. They exist as references that the batched, paged engine is
checked against, so they stay in the tree and stay tested:

``generate_naive``
    No cache at all. Recomputes the entire sequence every step, O(n^2). The
    simplest thing that can possibly be correct, and bit-identical to
    HuggingFace on float32 CPU.

``generate_cached``
    Stage 2. Prefills the prompt once, then decodes one token at a time against a
    contiguous KV cache. Must produce the same tokens as ``generate_naive``;
    that equality is the stage-2 gate.

``generate_paged``
    Stage 3. Same loop, but KV lives in a shared block pool addressed through the
    sequence's block table, with blocks allocated as the sequence grows and
    released when it ends. Must produce the same tokens as both of the above.

``generate_engine``
    Stage 4. A single request driven through the real scheduler and engine loop,
    so the batching machinery is held to the same gate as everything else. This
    one is a thin adapter rather than a separate implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from vox.cache import ContiguousKVCache, PagedKVCache
from vox.model import Qwen2ForCausalLM
from vox.sampler import SamplingParams, sample
from vox.sequence import Sequence, SequenceStatus


@dataclass
class GenerationResult:
    prompt_token_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "length"  # "length" | "stop"

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.token_ids


@torch.inference_mode()
def generate_naive(
    model: Qwen2ForCausalLM,
    prompt_token_ids: list[int],
    params: SamplingParams = SamplingParams(),
    stop_token_ids: Iterable[int] = (),
    generator: torch.Generator | None = None,
) -> GenerationResult:
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids is empty")

    stop = set(stop_token_ids)
    result = GenerationResult(prompt_token_ids=list(prompt_token_ids))
    tokens = list(prompt_token_ids)

    for _ in range(params.max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        logits = model(input_ids)[:, -1, :]
        next_id = int(sample(logits, [params], generator)[0])

        tokens.append(next_id)
        result.token_ids.append(next_id)
        if next_id in stop:
            result.finish_reason = "stop"
            break

    return result


def generate_cached(
    model: Qwen2ForCausalLM,
    prompt_token_ids: list[int],
    params: SamplingParams = SamplingParams(),
    stop_token_ids: Iterable[int] = (),
    generator: torch.Generator | None = None,
    cache: ContiguousKVCache | None = None,
) -> GenerationResult:
    """Prefill the prompt, then decode one token at a time against the cache.

    Pass ``cache`` to reuse a buffer across calls; otherwise one is sized for
    exactly this request. It is allocated outside ``inference_mode`` so it stays
    a normal tensor and can be reused afterwards.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids is empty")

    if cache is None:
        # Worst case is the prompt plus every generated token but the last, which
        # is never written back. The +1 keeps the arithmetic obvious.
        cache = ContiguousKVCache(
            model.cfg,
            len(prompt_token_ids) + params.max_new_tokens + 1,
            device=model.device,
            dtype=model.dtype,
        )
    else:
        cache.reset()

    stop = set(stop_token_ids)
    result = GenerationResult(prompt_token_ids=list(prompt_token_ids))

    # The first iteration feeds the whole prompt; every later one feeds the single
    # token just produced. Positions come from the cache length, so the loop body
    # does not need to distinguish prefill from decode.
    step_tokens: list[int] = list(prompt_token_ids)

    with torch.inference_mode():
        for _ in range(params.max_new_tokens):
            input_ids = torch.tensor([step_tokens], dtype=torch.long, device=model.device)
            logits = model(input_ids, cache=cache)[:, -1, :]
            next_id = int(sample(logits, [params], generator)[0])

            result.token_ids.append(next_id)
            if next_id in stop:
                result.finish_reason = "stop"
                break
            step_tokens = [next_id]

    return result


def generate_paged(
    model: Qwen2ForCausalLM,
    prompt_token_ids: list[int],
    params: SamplingParams = SamplingParams(),
    stop_token_ids: Iterable[int] = (),
    generator: torch.Generator | None = None,
    cache: PagedKVCache | None = None,
    block_size: int = 16,
) -> GenerationResult:
    """Decode one sequence against a paged block pool.

    Blocks are allocated just before each forward, covering only the tokens about
    to be computed, and released in a ``finally`` so an exception mid-decode does
    not leak them. That release path is the one cancellation reuses in stage 7.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids is empty")

    if cache is None:
        # Enough blocks for this request, plus one for the partial tail.
        needed = len(prompt_token_ids) + params.max_new_tokens
        cache = PagedKVCache(
            model.cfg,
            num_blocks=(needed + block_size - 1) // block_size + 1,
            block_size=block_size,
            device=model.device,
            dtype=model.dtype,
        )

    stop = set(stop_token_ids)
    sequence = Sequence(token_ids=list(prompt_token_ids))
    sequence.status = SequenceStatus.RUNNING
    result = GenerationResult(prompt_token_ids=list(prompt_token_ids))

    # The first iteration feeds the whole prompt; every later one feeds the single
    # token just produced. Positions come from the state, so the loop body does
    # not distinguish prefill from decode.
    step_tokens: list[int] = list(prompt_token_ids)

    try:
        with torch.inference_mode():
            for _ in range(params.max_new_tokens):
                cache.allocate_for(sequence, len(step_tokens))
                state = cache.build_state(
                    sequence, len(step_tokens), device=model.device, dtype=model.dtype
                )

                input_ids = torch.tensor([step_tokens], dtype=torch.long, device=model.device)
                logits = model(input_ids, state=state)[:, -1, :]
                sequence.num_computed_tokens += len(step_tokens)

                next_id = int(sample(logits, [params], generator)[0])
                sequence.append_token(next_id)
                result.token_ids.append(next_id)

                if next_id in stop:
                    result.finish_reason = "stop"
                    sequence.status = SequenceStatus.FINISHED
                    break
                step_tokens = [next_id]
            else:
                sequence.status = SequenceStatus.FINISHED
    finally:
        cache.free(sequence)

    return result


def generate_engine(
    model: Qwen2ForCausalLM,
    prompt_token_ids: list[int],
    params: SamplingParams = SamplingParams(),
    stop_token_ids: Iterable[int] = (),
    generator: torch.Generator | None = None,
    cache: PagedKVCache | None = None,
    block_size: int = 16,
) -> GenerationResult:
    """Run one request through the real engine, for comparison against the rest.

    Deliberately not a fourth implementation: it drives ``Engine``, so a
    divergence here is a batching bug, not a second thing to keep in sync.
    """
    from vox.engine import Engine  # local: engine imports batch imports cache

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids is empty")

    if cache is None:
        needed = len(prompt_token_ids) + params.max_new_tokens
        cache = PagedKVCache(
            model.cfg,
            num_blocks=(needed + block_size - 1) // block_size + 1,
            block_size=block_size,
            device=model.device,
            dtype=model.dtype,
        )

    engine = Engine(model, cache, generator=generator)
    engine.add_request("ref", list(prompt_token_ids), params, frozenset(stop_token_ids))
    output = engine.run_until_done()["ref"]

    return GenerationResult(
        prompt_token_ids=list(prompt_token_ids),
        token_ids=list(output.token_ids),
        finish_reason=output.finish_reason or "length",
    )
