"""Weight-free unit tests for the primitives.

These need no checkpoint and run in milliseconds. They exist because the two
easiest ways to get Qwen2 subtly wrong -- the RoPE convention and the GQA head
mapping -- both still produce fluent-looking output, so they need to be pinned
down directly rather than only inferred from an end-to-end comparison.
"""

from __future__ import annotations

import math

import pytest
import torch

from vox.config import Qwen2Config
from vox.layers import RMSNorm, RotaryCache, apply_rope, repeat_kv, rotate_half
from vox.state import AttentionState


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))

    expected = norm.weight * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_accumulates_in_float32():
    """The float32 upcast must happen even when the input is low precision."""
    x = torch.full((1, 1, 4), 1e-3, dtype=torch.bfloat16)
    norm = RMSNorm(4, eps=1e-6).to(torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16
    # Without the upcast, squaring 1e-3 in bfloat16 loses the magnitude entirely.
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0.5


def test_rotate_half_is_the_half_split_convention():
    """rotate_half([a,b,c,d]) == [-c,-d,a,b]; NOT the pairwise-interleaved layout."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rope_is_a_rotation():
    """RoPE must preserve vector norms -- it is a rotation, nothing else."""
    torch.manual_seed(0)
    rotary = RotaryCache(head_dim=64, max_position=128, theta=1_000_000.0)
    q = torch.randn(1, 4, 10, 64)
    k = torch.randn(1, 2, 10, 64)
    cos, sin = rotary(torch.arange(10).unsqueeze(0))

    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


def test_apply_rope_at_position_zero_is_identity():
    rotary = RotaryCache(head_dim=16, max_position=8, theta=10_000.0)
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 1, 16)
    cos, sin = rotary(torch.zeros(1, 1, dtype=torch.long))
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_rope_depends_only_on_relative_position():
    """The property the whole KV cache design rests on.

    q.k after RoPE depends on (pos_q - pos_k), not on the absolute positions. If
    this did not hold, a cached key written at position n could not be reused.
    """
    torch.manual_seed(0)
    rotary = RotaryCache(head_dim=32, max_position=256, theta=1_000_000.0)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def dot_at(pos_q: int, pos_k: int) -> float:
        cos_q, sin_q = rotary(torch.tensor([[pos_q]]))
        cos_k, sin_k = rotary(torch.tensor([[pos_k]]))
        q_rot, _ = apply_rope(q, k, cos_q, sin_q)
        _, k_rot = apply_rope(q, k, cos_k, sin_k)
        return float((q_rot * k_rot).sum())

    assert dot_at(5, 3) == pytest.approx(dot_at(105, 103), abs=1e-4)
    assert dot_at(5, 3) != pytest.approx(dot_at(5, 4), abs=1e-4)


def test_rotary_cache_indexes_by_position():
    rotary = RotaryCache(head_dim=8, max_position=16, theta=10_000.0)
    cos, sin = rotary(torch.tensor([[3, 1, 3]]))
    assert cos.shape == (1, 3, 8)
    # Same position must give the same row, regardless of where it appears.
    assert torch.equal(cos[0, 0], cos[0, 2])
    assert not torch.equal(cos[0, 0], cos[0, 1])


def test_rotary_cache_buffers_stay_out_of_state_dict():
    """Non-persistent, so they can never collide with checkpoint keys."""
    assert RotaryCache(head_dim=8, max_position=16, theta=10_000.0).state_dict() == {}


def test_repeat_kv_maps_query_head_to_the_right_kv_head():
    """Query head i must read KV head i // kv_repeat, matching HF's expand+reshape."""
    # 2 KV heads, repeat 3 -> 6 query heads. Tag each KV head with its index.
    x = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
    out = repeat_kv(x, 3)
    assert out.shape == (1, 6, 1, 1)
    assert out.flatten().tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_repeat_kv_identity_when_not_grouped():
    x = torch.randn(1, 4, 3, 8)
    assert repeat_kv(x, 1) is x


def test_config_rejects_non_divisible_head_counts():
    with pytest.raises(ValueError, match="multiple of"):
        Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=5, num_key_value_heads=2, max_position_embeddings=8,
            rms_norm_eps=1e-6, rope_theta=1e4, tie_word_embeddings=True, head_dim=4,
        )


def test_attention_matches_dense_reference():
    """Our GQA attention against a hand-rolled dense causal reference.

    Exercises the full head reshape/transpose chain, which is where an off-by-one
    in a ``view`` silently mixes heads together.
    """
    torch.manual_seed(0)
    from vox.layers import Attention

    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10_000.0, tie_word_embeddings=True, head_dim=4,
    )
    attn = Attention(cfg, layer_idx=0).eval()
    rotary = RotaryCache(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    seq = 6
    x = torch.randn(1, seq, cfg.hidden_size)
    cos, sin = rotary(torch.arange(seq).unsqueeze(0))

    with torch.no_grad():
        got = attn(x, cos, sin, AttentionState.uncached(seq))

        # Reference: explicit per-head loop with an explicit causal mask.
        q = attn.q_proj(x).view(1, seq, 8, 4).transpose(1, 2)
        k = attn.k_proj(x).view(1, seq, 2, 4).transpose(1, 2)
        v = attn.v_proj(x).view(1, seq, 2, 4).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        mask = torch.full((seq, seq), float("-inf")).triu(1)
        heads = []
        for h in range(8):
            kh, vh = k[0, h // 4], v[0, h // 4]
            scores = (q[0, h] @ kh.T) / math.sqrt(4) + mask
            heads.append(torch.softmax(scores, dim=-1) @ vh)
        expected = attn.o_proj(torch.cat(heads, dim=-1).unsqueeze(0))

    assert torch.allclose(got, expected, atol=1e-5)


def test_swiglu_matches_reference():
    torch.manual_seed(0)
    from vox.layers import SwiGLU

    cfg = Qwen2Config(
        vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8,
        rms_norm_eps=1e-6, rope_theta=1e4, tie_word_embeddings=True, head_dim=4,
    )
    mlp = SwiGLU(cfg).eval()
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        gate = mlp.gate_proj(x)
        expected = mlp.down_proj((gate * torch.sigmoid(gate)) * mlp.up_proj(x))
        assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_qkv_bias_present_and_o_proj_bias_absent():
    """A Qwen2-specific detail: biased q/k/v, unbiased o_proj."""
    from vox.layers import Attention

    cfg = Qwen2Config(
        vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8,
        rms_norm_eps=1e-6, rope_theta=1e4, tie_word_embeddings=True, head_dim=4,
    )
    attn = Attention(cfg, layer_idx=0)
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None
