"""Model configuration, read straight out of a HuggingFace ``config.json``.

We parse the fields we actually use and nothing else, so that an unexpected
architecture shows up as a KeyError at load time rather than as silently wrong
output later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    head_dim: int
    # Qwen2 carries a bias on the q/k/v projections, but not on o_proj or the MLP.
    qkv_bias: bool = True

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> Qwen2Config:
        raw = json.loads(Path(path).read_text())
        arch = raw.get("architectures", ["<missing>"])[0]
        if arch != "Qwen2ForCausalLM":
            raise ValueError(f"vox implements Qwen2ForCausalLM, got {arch!r}")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError(f"vox implements SwiGLU/silu, got {raw['hidden_act']!r}")
        return cls(
            rope_theta=cls._rope_theta(raw),
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            head_dim=raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"],
        )

    @staticmethod
    def _rope_theta(raw: dict) -> float:
        """Read rope_theta from either config layout, and refuse scaled RoPE.

        Older checkpoints store ``rope_theta`` at the top level; transformers 5
        writes a ``rope_parameters`` dict instead. Either way, a ``rope_type``
        other than "default" means the model expects scaled RoPE (linear, YaRN,
        ...), which we do not implement -- and which would decode subtly wrong
        rather than fail loudly, so it has to be rejected here.
        """
        params = raw.get("rope_parameters")
        if params:
            rope_type = params.get("rope_type", "default")
            if rope_type != "default":
                raise ValueError(f"vox implements plain RoPE only, got rope_type={rope_type!r}")
            return float(params["rope_theta"])
        if raw.get("rope_scaling"):
            raise ValueError(f"vox implements plain RoPE only, got rope_scaling={raw['rope_scaling']!r}")
        return float(raw["rope_theta"])

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen2Config:
        return cls.from_json(Path(model_dir) / "config.json")

    @property
    def kv_repeat(self) -> int:
        """How many query heads share each KV head (the GQA group size)."""
        return self.num_attention_heads // self.num_key_value_heads
