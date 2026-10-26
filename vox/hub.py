"""Locating model weights on disk.

``resolve_model_dir`` returns a local directory of HuggingFace-format weights,
downloading them into the usual HF cache if they are not there yet. Set
``VOX_MODEL_DIR`` to point at a directory you manage yourself and nothing is
downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# The forward pass needs config + weights; the tokenizer files come along because
# we use HF's tokenizer (implementing BPE was never part of the point).
_ALLOW = ["config.json", "*.safetensors", "*.json", "*.txt"]


def resolve_model_dir(model_id: str = DEFAULT_MODEL_ID, *, allow_download: bool = True) -> Path:
    override = os.environ.get("VOX_MODEL_DIR")
    if override:
        path = Path(override).expanduser()
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"VOX_MODEL_DIR={path} has no config.json")
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            allow_patterns=_ALLOW,
            local_files_only=not allow_download,
        )
    )
