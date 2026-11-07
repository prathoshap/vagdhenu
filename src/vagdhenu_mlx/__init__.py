"""Vāgdhenu on Apple silicon: IndicF5 DiT + BigVGAN-v2 inference in MLX (no torch at runtime).

    from vagdhenu_mlx import Renderer
    r = Renderer("models/mlx")            # converted with `python -m vagdhenu_mlx.convert`
    sr, audio = r.render_one("तस्मै नमः ...", meter="anuṣṭubh")
"""
from .renderer import Renderer, detect_meter_key  # noqa: F401

__all__ = ["Renderer", "detect_meter_key"]
