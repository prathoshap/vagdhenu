"""Vāgdhenu on Apple silicon: IndicF5 DiT + BigVGAN-v2 inference in MLX (no torch at runtime).

    from vagdhenu_mlx import Renderer
    r = Renderer("models/mlx")            # converted with `python -m vagdhenu_mlx.convert`
    sr, audio = r.render_one("तस्मै नमः ...", meter="anuṣṭubh")

`Renderer` / `detect_meter_key` are imported lazily so that `python -m vagdhenu_mlx.convert` (the
one torch-only step) works in an environment without mlx.
"""

__all__ = ["Renderer", "detect_meter_key"]


def __getattr__(name):
    if name in __all__:
        from . import renderer
        return getattr(renderer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
