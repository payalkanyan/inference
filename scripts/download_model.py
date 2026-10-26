"""Fetch the model weights into the HuggingFace cache.

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import vox without installing

from vox.hub import DEFAULT_MODEL_ID, resolve_model_dir

if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    print(f"resolving {model_id} ...")
    path = resolve_model_dir(model_id)
    print(f"ready: {path}")
    for entry in sorted(path.iterdir()):
        print(f"  {entry.name}")
