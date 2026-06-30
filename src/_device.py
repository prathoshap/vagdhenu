"""Centralized torch device selection — Apple Silicon (MPS) / CUDA / CPU portable.

Upstream IndicF5 already auto-detects MPS (see f5_tts/infer/utils_infer.py), but Vāgdhenu's
inference scripts originally hardcoded "cuda" / .cuda(). Importing DEVICE from here lets every
entry point agree on one backend, and turns on PyTorch's per-op CPU fallback so any residual
op MPS doesn't (yet) implement degrades gracefully instead of crashing.

Override at runtime by setting VAGDHENU_DEVICE=cuda|mps|cpu in the environment.
"""
import os
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(prefer: str = None) -> str:
    if prefer:
        if prefer not in ("cuda", "mps", "cpu"):
            raise ValueError(f"unknown device '{prefer}', expected cuda|mps|cpu")
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VAGDHENU_DEVICE=cuda but torch.cuda.is_available() is False")
        if prefer == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("VAGDHENU_DEVICE=mps but torch.backends.mps.is_available() is False")
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device(os.environ.get("VAGDHENU_DEVICE"))
