"""Pydantic request/response schemas for the Vāgdhenu REST API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChantRequest(BaseModel):
    """One shloka → chant synthesis request.

    `text` is a single Sanskrit verse in any Indic script (Devanagari, Kannada,
    Telugu, …). Split into pādas on newline / daṇḍa automatically. `meter` is
    optional — leave empty for auto-detect. `seed` changes the take.
    """

    text: str = Field(..., min_length=1, description="A single Sanskrit verse.")
    meter: Optional[str] = Field(
        default=None, description="Meter name (chandas). Empty = auto-detect."
    )
    seed: int = Field(default=60, ge=0, le=1000, description="Random seed for the take.")


class ChantMeta(BaseModel):
    """Metadata returned alongside every chant (in the JSON envelope)."""

    meter: str
    recognized: bool
    seed: int
    duration_s: float
    sample_rate: int


class ChantResponse(BaseModel):
    """JSON envelope for the `/api/chant` JSON endpoint (audio is base64 mp3)."""

    meta: ChantMeta
    audio_base64: str
    audio_mime: str = "audio/mpeg"
