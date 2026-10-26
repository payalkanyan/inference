"""Loading HuggingFace safetensors into our own module tree."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from vox.config import Qwen2Config
from vox.model import Qwen2ForCausalLM


def _read_shards(model_dir: Path) -> dict[str, torch.Tensor]:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        tensors.update(load_file(str(shard)))
    return tensors


def load_model(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Qwen2ForCausalLM:
    """Build the model and load weights.

    The published checkpoint is bfloat16. Casting it to float32 here is exactly
    what ``from_pretrained(dtype=torch.float32)`` does on the HF side, so the two
    models start from bit-identical parameters and any output divergence is our
    arithmetic, not our loading.
    """
    model_dir = Path(model_dir)
    cfg = Qwen2Config.from_pretrained(model_dir)

    model = Qwen2ForCausalLM(cfg)
    model.to(device=device, dtype=dtype)

    state = _read_shards(model_dir)
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}

    # Tied embeddings are stored once, under embed_tokens; lm_head shares the
    # same Parameter object, so there is nothing to load for it.
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
        expected_missing = {"lm_head.weight"}
    else:
        expected_missing = set()

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = set(missing) - expected_missing
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model.\n  missing: {sorted(missing)}\n"
            f"  unexpected: {sorted(unexpected)}"
        )

    model.eval()
    return model
