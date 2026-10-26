"""The Qwen2 decoder stack.

Module and parameter names deliberately mirror HuggingFace's
(``model.layers.0.self_attn.q_proj.weight`` and friends). The forward pass is
entirely ours; matching the names only means weight loading is a plain
``load_state_dict`` instead of a hand-maintained mapping table, which removes a
whole class of silent transposed/swapped-weight bugs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vox.cache import ContiguousKVCache
from vox.config import Qwen2Config
from vox.layers import RMSNorm, Attention, RotaryCache, SwiGLU
from vox.state import AttentionState


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = SwiGLU(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, state: AttentionState) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, state)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryCache(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        cache: ContiguousKVCache | None = None,
        state: AttentionState | None = None,
    ) -> Tensor:
        """Run the stack. Pass at most one of ``cache`` or ``state``.

        ``cache`` is the single-sequence convenience path: the model builds the
        state and commits the cache length itself. ``state`` is the explicit path
        used by the paged cache and, from stage 4, by the model runner -- there
        the caller owns all bookkeeping and the model mutates nothing.
        """
        if cache is not None and state is not None:
            raise ValueError("pass either cache or state, not both")
        if cache is not None and not isinstance(cache, ContiguousKVCache):
            raise TypeError(
                f"{type(cache).__name__} needs a prebuilt state; "
                "call its build_state() and pass state="
            )

        batch, seq = input_ids.shape
        x = self.embed_tokens(input_ids)

        owns_bookkeeping = state is None
        if state is None:
            state = (
                cache.build_state(seq, device=x.device, dtype=x.dtype)
                if cache is not None
                else AttentionState.uncached(seq)
            )

        if position_ids is None:
            # Taken from the state, so a decode step is at position `start`
            # rather than 0. Getting this wrong is survivable-looking: the model
            # keeps emitting fluent text while attending at the wrong distances.
            position_ids = (
                torch.arange(state.start, state.start + seq, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch, seq)
            )

        cos, sin = self.rotary(position_ids)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, state)

        # Committed once, after every layer has written its slice.
        if owns_bookkeeping and cache is not None:
            cache.advance(seq)
        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        cache: ContiguousKVCache | None = None,
        state: AttentionState | None = None,
    ) -> Tensor:
        """Returns logits of shape [batch, seq, vocab_size]."""
        return self.lm_head(self.model(input_ids, position_ids, cache, state))

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype
