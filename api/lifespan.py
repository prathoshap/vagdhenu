"""Model lifecycle: load the Renderer once at startup, reuse across requests.

Keeps the heavy DiT + BigVGAN load off the request path. The Renderer is
imported lazily so the API module can be imported on a CPU-only machine
(e.g. for linting) without pulling torch/CUDA.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

# Make src/ importable so `render_core` and `prep_text` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import limits  # noqa: E402  (shared abuse guards)

# Re-export the limits helpers so api.app can call them via this module.
limits_validate = limits.validate_one_shloka
limits_count = limits.check_and_count
limits_ip = limits.client_ip

BANK_PATH = os.path.join(_SRC, "reference_bank", "bank.json")
VOCAB_PATH = os.path.join(_SRC, "reference_bank", "vocab.txt")
_AUTO = "__auto__"

# Weight paths: prefer local models/ dir (scripts/download_weights.py), then env overrides.
_MODELS = os.path.join(_REPO, "models")
DEFAULT_VOICE = os.environ.get(
    "VAGDHENU_VOICE", os.path.join(_MODELS, "voice_steer_ema_2026-06-17.pt")
)
DEFAULT_VOC = os.environ.get(
    "VAGDHENU_VOC", os.path.join(_MODELS, "voc_bigvgan_EMA_2026-06-11.pth")
)

# Meter list + alias LUT, read once at import (no GPU needed).
_bank: dict = json.load(open(BANK_PATH, encoding="utf-8"))
METERS = [
    k
    for k, v in _bank.items()
    if not k.startswith("_") and isinstance(v, dict) and "wav" in v
]
_PREFERRED = [
    m
    for m in ("anuṣṭubh", "upajāti", "śārdūlavikrīḍita", "vasantatilakā", "mālinī")
    if m in METERS
]
METERS = _PREFERRED + [m for m in METERS if m not in _PREFERRED]
_FALLBACK = "vasantatilakā" if "vasantatilakā" in METERS else (METERS[0] if METERS else "anushtubh")

_ALIAS: dict[str, str] = {}
for _k, _v in _bank.items():
    if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v:
        continue
    _ALIAS[_k.lower()] = _k
    _ALIAS[_v["wav"].replace(".wav", "").lower()] = _k


def resolve_meter(name: Optional[str]) -> tuple[str, bool]:
    """Resolve a detected/chosen meter name to (pretty bank key, recognized?)."""
    k = _ALIAS.get((name or "").lower())
    return (k, True) if k else (_FALLBACK, False)


_renderer = None


def get_renderer():
    """Lazily build the Renderer (loads DiT + BigVGAN onto CUDA). Thread-safe-ish:
    FastAPI lifespan calls this once at startup; subsequent calls return the cached instance."""
    global _renderer
    if _renderer is None:
        from render_core import Renderer  # noqa: E402  (lazy: pulls torch)

        voice = DEFAULT_VOICE
        voc = DEFAULT_VOC
        if not os.path.exists(voice):
            # Fall back to HF download (Colab convenience).
            from huggingface_hub import hf_hub_download

            repo = os.environ.get("VAGDHENU_HF", "prathoshap/vagdhenu")
            voice = hf_hub_download(repo, "voice_steer_ema_2026-06-17.pt")
            voc = hf_hub_download(repo, "voc_bigvgan_EMA_2026-06-11.pth")
        _renderer = Renderer(voice, voc, BANK_PATH, device="cuda", vocab_file=VOCAB_PATH)
    return _renderer


def detect_meter(text: str) -> str:
    """Best-effort meter detection (pure text, no GPU). Returns '' if unknown."""
    from render_core import detect_meter_key  # noqa: E402

    return detect_meter_key(text)
