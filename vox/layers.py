"""The four primitives Qwen2 is built from: RMSNorm, RoPE, GQA attention, SwiGLU.

Everything here is plain PyTorch. Attention goes through ``F.scaled_dot_product_attention``
so the same code path works on CPU and CUDA.

Numerical-parity notes, since token-identity with HuggingFace is the gate for the
whole project:
  * RMSNorm accumulates the variance in float32 and casts back to the input dtype,
    the same order HF uses. In float32 this is a no-op, but it keeps the bf16/fp16
    fast path honest too.
  * RoPE uses the "rotate_half" convention (split the head dim in half, rotate the
    halves against each other). This is what HF's Qwen2 does. It is *not* the
    pairwise-interleaved GPT-NeoX layout, and the two are not interchangeable:
    swapping them loads fine and produces fluent-looking garbage.
  * GQA expands KV heads so that query head ``i`` reads KV head ``i // kv_repeat``,
    matching HF's expand+reshape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vox.config import Qwen2Config
from vox.state import AttentionState


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class RotaryCache(nn.Module):
    """Precomputed cos/sin tables, indexed by absolute position.

    Indexing by explicit ``position_ids`` rather than assuming ``arange(seq_len)``
    is what later stages need: a decode step is at position ``n``, and a prefill
    chunk starts partway through a prompt.

    The tables are non-persistent buffers, so they follow ``.to(device)`` but stay
    out of ``state_dict()`` and cannot collide with checkpoint keys.
    """

    def __init__(self, head_dim: int, max_position: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)              # [max_position, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)               # [max_position, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k are [batch, heads, seq, head_dim]; cos, sin are [batch, seq, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: Tensor, n_repeat: int) -> Tensor:
    """[batch, kv_heads, seq, head_dim] -> [batch, kv_heads * n_repeat, seq, head_dim]."""
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        # Required, not defaulted: a default of 0 would make every layer write
        # over layer 0's cache and still produce plausible-looking output.
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_out, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_out, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(q_out, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, state: AttentionState) -> Tensor:
        batch, seq, _ = x.shape
        if state.cache is not None and batch != 1:
            raise ValueError("a cached forward holds one sequence; batching arrives in stage 4")

        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied before the cache write, so stored keys are already rotated.
        q, k = apply_rope(q, k, cos, sin)

        # Attention does not know which cache this is. Contiguous and paged
        # differ in how they address memory, not in what attention asks for.
        if state.cache is not None:
            state.cache.write(self.layer_idx, k, v, state)
            k, v = state.cache.read(self.layer_idx, state)

        k = repeat_kv(k, self.cfg.kv_repeat)
        v = repeat_kv(v, self.cfg.kv_repeat)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=state.mask, is_causal=state.is_causal
        )
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
