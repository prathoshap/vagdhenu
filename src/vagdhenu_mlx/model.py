"""Loading the converted weights into the MLX modules."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from . import bigvgan as BV
from . import dit as D
from .text import load_vocab


def _load_tree(module: nn.Module, params: dict, dtype):
    params = {k: v.astype(dtype) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
              for k, v in params.items()}
    module.load_weights(list(params.items()), strict=True)
    mx.eval(module.parameters())
    return module


def load_dit(model_dir: str, dtype=mx.bfloat16) -> tuple[D.DiT, dict[str, int]]:
    vocab = load_vocab(os.path.join(model_dir, "vocab.txt"))
    sd = mx.load(os.path.join(model_dir, "dit.safetensors"))
    dit = D.DiT(text_num_embeds=len(vocab), **D.DIT_CFG)
    return _load_tree(dit, D.remap_torch_state(sd), dtype), vocab


def load_bigvgan(model_dir: str, dtype=mx.float32) -> BV.BigVGAN:
    sd = mx.load(os.path.join(model_dir, "bigvgan.safetensors"))
    voc = BV.BigVGAN()
    return _load_tree(voc, BV.remap_torch_state(sd), dtype)
