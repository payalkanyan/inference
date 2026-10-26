"""Token sampling: greedy and temperature.

Sampling parameters are per-sequence from the start, because the scheduler in
later stages batches independent requests together and they will not agree on
temperature. The all-greedy case gets a vectorised fast path, since that is the
common one for voice agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    #: 0.0 means greedy (argmax). Anything above 0 samples from the softmax.
    temperature: float = 0.0
    max_new_tokens: int = 64

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def sample(
    logits: Tensor,
    params: Sequence[SamplingParams],
    generator: torch.Generator | None = None,
) -> Tensor:
    """logits: [batch, vocab_size] -> token ids: [batch].

    One row of ``logits`` and one ``SamplingParams`` per sequence.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [batch, vocab] logits, got {tuple(logits.shape)}")
    if len(params) != logits.shape[0]:
        raise ValueError(f"got {len(params)} sampling params for {logits.shape[0]} rows")

    if all(p.greedy for p in params):
        return logits.argmax(dim=-1)

    out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    for i, p in enumerate(params):
        if p.greedy:
            out[i] = logits[i].argmax()
        else:
            probs = torch.softmax(logits[i].to(torch.float32) / p.temperature, dim=-1)
            out[i] = torch.multinomial(probs, num_samples=1, generator=generator)
    return out
