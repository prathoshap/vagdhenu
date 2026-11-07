"""IndicF5 / F5-TTS DiT (v0 architecture) in MLX.

Numerically mirrors f5_tts.model.backbones.dit.DiT at IndicF5 commit 13f7c4d, with one change in
*how* (not *what*) it computes. The per-generation constants are hoisted out of the ODE loop:
  * the text embedding (embedding + 4 ConvNeXt-V2 blocks) is computed once, not 2x per step,
  * the conditioning part of the input projection, W_c·cond + W_t·text + b, is computed once,
  * the AdaLN-Zero modulations depend only on t, so they are computed for every step up front.
Classifier-free guidance runs the conditional and the null pass as one batch. Several
utterances can share a batch: padding is masked exactly (convs, GRN and attention), so a
batched item gives the same result as rendering it alone.

Quirk kept on purpose: as in the original F5-TTS v0, RoPE (x-transformers RotaryEmbedding with
dim_head=64) is applied to the *first 64 channels of the full 1024-d projection* (i.e. head 0)
before the heads are split, with interleaved (GPT-J style) pairs.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

DIT_CFG = dict(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, text_dim=512, conv_layers=4, mel_dim=100)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * (x * x * x))))


def mish(x):
    return x * mx.tanh(nn.softplus(x))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: dim // 2].astype(mx.float32) / dim))
    t = mx.arange(end).astype(mx.float32)
    f = t[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


def sinus_time_embed(t: mx.array, dim: int = 256, scale: float = 1000.0):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = mx.exp(mx.arange(half).astype(mx.float32) * -emb)
    emb = scale * t[:, None] * emb[None, :]
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


def _layer_norm(x, eps=1e-6):
    return mx.fast.layer_norm(x, None, None, eps)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x):
        # L2 norm over the *sequence* axis (as in the reference); accumulate in fp32.
        gx = mx.sqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=1, keepdims=True))
        nx = (gx / (mx.mean(gx, axis=-1, keepdims=True) + 1e-6)).astype(x.dtype)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x, valid=None):
        residual = x
        if valid is not None:
            x = x * valid
        x = self.dwconv(x)
        x = self.norm(x)
        x = nn.gelu(self.pwconv1(x))
        if valid is not None:
            x = x * valid
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class TextEmbedding(nn.Module):
    MAX_POS = 4096

    def __init__(self, text_num_embeds, text_dim, conv_layers):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.text_blocks = [ConvNeXtV2Block(text_dim, text_dim * 2) for _ in range(conv_layers)]
        self._freqs = precompute_freqs_cis(text_dim, self.MAX_POS)

    def __call__(self, tokens, valid=None):
        """tokens: (B, N) int, already shifted by +1 with 0 as the filler token."""
        n = tokens.shape[1]
        x = self.text_embed(tokens)
        idx = mx.minimum(mx.arange(n), self.MAX_POS - 1)
        x = x + self._freqs[idx].astype(x.dtype)
        for blk in self.text_blocks:
            x = blk(x, valid)
        return x


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups)

    def __call__(self, x, valid=None):
        if valid is not None:
            x = x * valid
        x = mish(self.conv1(x))
        if valid is not None:
            x = x * valid
        return mish(self.conv2(x))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.attn_norm = nn.Linear(dim, dim * 6)      # AdaLayerNormZero.linear (applied up front)
        self.qkv = nn.Linear(dim, 3 * heads * dim_head)  # fused to_q | to_k | to_v
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.ff1 = nn.Linear(dim, int(dim * ff_mult))
        self.ff2 = nn.Linear(int(dim * ff_mult), dim)

    def __call__(self, x, mod, attn_mask=None):
        # mod: (6, dim) for this step = shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        H, Dh = self.heads, self.dim_head
        h = _layer_norm(x) * (1 + mod[1]) + mod[0]
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        # RoPE on the first dim_head channels of the un-split projection (head 0), GPT-J pairs.
        q = mx.fast.rope(q, Dh, traditional=True, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, Dh, traditional=True, base=10000.0, scale=1.0, offset=0)
        # unflatten/flatten (not reshape with B, N) keep this traceable with mx.compile(shapeless=True)
        q = mx.unflatten(q, -1, (H, Dh)).transpose(0, 2, 1, 3)
        k = mx.unflatten(k, -1, (H, Dh)).transpose(0, 2, 1, 3)
        v = mx.unflatten(v, -1, (H, Dh)).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh ** -0.5, mask=attn_mask)
        o = self.to_out(mx.flatten(o.transpose(0, 2, 1, 3), -2, -1))
        x = x + mod[2] * o
        h = _layer_norm(x) * (1 + mod[4]) + mod[3]
        return x + mod[5] * self.ff2(gelu_tanh(self.ff1(h)))


class DiT(nn.Module):
    def __init__(self, text_num_embeds, dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                 text_dim=512, conv_layers=4, mel_dim=100):
        super().__init__()
        self.dim, self.mel_dim = dim, mel_dim
        self.time_mlp1 = nn.Linear(256, dim)
        self.time_mlp2 = nn.Linear(dim, dim)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers)
        self.proj = nn.Linear(mel_dim * 2 + text_dim, dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim)
        self.transformer_blocks = [DiTBlock(dim, heads, dim_head, ff_mult) for _ in range(depth)]
        self.norm_out = nn.Linear(dim, dim * 2)
        self.proj_out = nn.Linear(dim, mel_dim)

    # ---- per-generation precompute ------------------------------------------------------
    def time_modulations(self, ts: mx.array):
        """ts: (S,) step times -> per-block (S, 6, dim) and final (S, 2, dim) modulations."""
        t = self.time_mlp2(nn.silu(self.time_mlp1(sinus_time_embed(ts.astype(mx.float32)).astype(self.time_mlp1.weight.dtype))))
        st = nn.silu(t)
        S = ts.shape[0]
        mods = mx.stack([blk.attn_norm(st).reshape(S, 6, self.dim) for blk in self.transformer_blocks], axis=1)
        fin = self.norm_out(st).reshape(S, 2, self.dim)  # (scale, shift)
        return mods, fin  # (S, depth, 6, dim), (S, 2, dim)

    def _proj_split(self):
        """proj.weight split into its x / cond / text column blocks (cached; done outside the
        compiled step, where slicing is not shapeless-traceable)."""
        w = self.proj.weight
        if getattr(self, "_split_of", None) is not w:
            m = self.mel_dim
            self._wx, self._wc, self._wt = w[:, :m].T, w[:, m:2 * m].T, w[:, 2 * m:].T
            mx.eval(self._wx, self._wc, self._wt)
            self._split_of = w
        return self._wx, self._wc, self._wt

    def context(self, cond, tokens, valid=None):
        """Constant part of the input projection: W_c·cond + W_t·text_embed(tokens) + b."""
        te = self.text_embed(tokens, valid)
        _, wc, wt = self._proj_split()
        return cond @ wc + te @ wt + self.proj.bias

    # ---- one ODE step ---------------------------------------------------------------------
    def step(self, x, ctx, mods, fin, valid=None, attn_mask=None):
        """x: (B, N, mel) noisy mel; ctx: (B, N, dim); mods: (depth, 6, dim); fin: (2, dim)."""
        wx, _, _ = self._proj_split()
        h = x.astype(ctx.dtype) @ wx + ctx
        h = self.conv_pos_embed(h, valid) + h
        for i, blk in enumerate(self.transformer_blocks):
            h = blk(h, mods[i], attn_mask)
        h = _layer_norm(h) * (1 + fin[0]) + fin[1]
        return self.proj_out(h)


def remap_torch_state(sd: dict) -> dict:
    """Map the converted torch state dict (dit.safetensors) onto this module's parameter tree."""
    out = {}
    blocks: dict[int, dict] = {}
    for k, v in sd.items():
        k = k.removeprefix("transformer.")
        if k.startswith("time_embed.time_mlp.0."):
            out["time_mlp1." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("time_embed.time_mlp.2."):
            out["time_mlp2." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("text_embed."):
            if ".dwconv.weight" in k:
                v = v.transpose(0, 2, 1)
            out[k] = v
        elif k.startswith("input_embed.proj."):
            out[k.replace("input_embed.", "")] = v
        elif k.startswith("input_embed.conv_pos_embed.conv1d."):
            idx, p = k.split(".")[-2:]
            name = {"0": "conv1", "2": "conv2"}[idx]
            out[f"conv_pos_embed.{name}.{p}"] = v.transpose(0, 2, 1) if p == "weight" else v
        elif k.startswith("transformer_blocks."):
            _, i, rest = k.split(".", 2)
            i = int(i)
            b = blocks.setdefault(i, {})
            if rest.startswith("attn_norm.linear."):
                out[f"transformer_blocks.{i}.attn_norm.{rest.rsplit('.', 1)[1]}"] = v
            elif rest.startswith("attn.to_q.") or rest.startswith("attn.to_k.") or rest.startswith("attn.to_v."):
                b[rest] = v
            elif rest.startswith("attn.to_out.0."):
                out[f"transformer_blocks.{i}.to_out.{rest.rsplit('.', 1)[1]}"] = v
            elif rest.startswith("ff.ff.0.0."):
                out[f"transformer_blocks.{i}.ff1.{rest.rsplit('.', 1)[1]}"] = v
            elif rest.startswith("ff.ff.2."):
                out[f"transformer_blocks.{i}.ff2.{rest.rsplit('.', 1)[1]}"] = v
            else:
                raise KeyError(k)
        elif k.startswith("norm_out.linear."):
            out["norm_out." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("proj_out."):
            out[k] = v
        elif k.startswith("rotary_embed."):
            continue
        else:
            raise KeyError(k)
    for i, b in blocks.items():
        for p in ("weight", "bias"):
            out[f"transformer_blocks.{i}.qkv.{p}"] = mx.concatenate(
                [b[f"attn.to_q.{p}"], b[f"attn.to_k.{p}"], b[f"attn.to_v.{p}"]], axis=0)
    return out
