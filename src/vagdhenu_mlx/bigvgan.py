"""BigVGAN-v2 generator (nvidia/bigvgan_v2_24khz_100band_256x topology) in MLX.

Mirrors NVIDIA's bigvgan.py torch path (use_cuda_kernel=False) after remove_weight_norm():
conv_pre -> 6 x [ConvTranspose1d upsample, 3 AMPBlock1 (SnakeBeta between anti-aliased 2x
up/down-sampling)] -> SnakeBeta -> conv_post -> clamp(-1, 1).

Layout is channels-last (B, T, C). The anti-aliased activation's 2x upsample is implemented in
polyphase form (two 6-tap depthwise convolutions, interleaved) instead of zero-stuffing + a
12-tap transposed convolution; it is the same linear operator, at half the memory traffic.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CFG = dict(
    upsample_rates=[4, 4, 2, 2, 2, 2],
    upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
    upsample_initial_channel=1536,
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    num_mels=100,
    use_bias_at_final=False,
    use_tanh_at_final=False,
)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size) -> np.ndarray:
    """alias_free_activation.torch.filter.kaiser_sinc_filter1d, in numpy (float64)."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)  # == torch.kaiser_window(periodic=False)
    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size
    filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    return filt / filt.sum()


_AA_SNAKE_SRC = r"""
    uint idx = thread_position_in_grid.x;
    const uint T = x_shape[1];
    const uint C = x_shape[2];
    if (idx >= x_shape[0] * T * C) return;
    const uint c = idx % C;
    const uint n = (idx / C) % T;
    const uint b = idx / (C * T);
    const device float* xb = x + (size_t)b * T * C;
    const float a = alpha[c];
    const float ib = ibeta[c];
    const int Ti = (int)T;
    float acc = 0.0f;
    // y[n] = sum_j f[j] * s[2n - 5 + j]   (12-tap low-pass, stride 2, replicate padding 5/6)
    // s[m] = snake(u[m]),  u = 2x polyphase upsample of x (replicate padding 3), see _ops()
    for (int j = 0; j < 12; ++j) {
        int m = 2 * (int)n - 5 + j;
        m = metal::clamp(m, 0, 2 * Ti - 1);
        const int q = m >> 1;
        const int p = m & 1;
        const int base = q - (p ? 2 : 3);
        float u = 0.0f;
        for (int k = 0; k < 6; ++k) {
            const int i = metal::clamp(base + k, 0, Ti - 1);
            u += wup[p * 6 + k] * xb[(size_t)i * C + c];
        }
        const float s = metal::precise::sin(u * a);
        acc += wdown[j] * (u + ib * s * s);
    }
    out[idx] = acc;
"""
_KERNEL = None


def _aa_snake_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="vagdhenu_aa_snakebeta",
            input_names=["x", "alpha", "ibeta", "wup", "wdown"],
            output_names=["out"],
            source=_AA_SNAKE_SRC,
        )
    return _KERNEL


_FUSED = None


def _use_fused() -> bool:
    """Fused Metal kernel only on the GPU stream (the CPU stream runs the plain-ops version)."""
    global _FUSED
    if _FUSED is None:
        _FUSED = mx.default_device() == mx.Device(mx.gpu) and mx.metal.is_available()
    return _FUSED and mx.default_device() == mx.Device(mx.gpu)


def set_fused(enabled: bool | None):
    """Force the fused kernel on/off (None = auto)."""
    global _FUSED
    _FUSED = enabled


class AntiAliasedSnakeBeta(nn.Module):
    """Activation1d(SnakeBeta): 2x upsample -> x + (1/beta) sin^2(alpha x) -> 2x downsample."""

    K = 12  # up/down kernel size

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.alpha = mx.zeros((channels,))  # log-scale (snake_logscale=True)
        self.beta = mx.zeros((channels,))
        f = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=self.K)
        # Polyphase decomposition of  2 * conv_transpose1d(replicate_pad(x, 5), f, stride=2)[15:-15]:
        #   y[2n]   = 2 * sum_{k=0..5} x[n-3+k] f[11-2k]
        #   y[2n+1] = 2 * sum_{k=0..5} x[n-2+k] f[10-2k]
        w0 = np.array([f[11 - 2 * k] for k in range(6)]) * 2
        w1 = np.array([f[10 - 2 * k] for k in range(6)]) * 2
        # depthwise conv weight (C_out=2C, k=6, 1): per channel c -> rows [2c]=w0, [2c+1]=w1
        up = np.stack([w0, w1], axis=0)[None].repeat(channels, axis=0).reshape(2 * channels, 6, 1)
        self._up = mx.array(up.astype(np.float32))
        self._down = mx.array(np.tile(f.astype(np.float32)[None, :, None], (channels, 1, 1)))
        self._cache_dtype = None

    def _weights(self, dtype):
        if self._cache_dtype is None or self._cache_dtype != dtype:
            self._w_up = self._up.astype(dtype)
            self._w_down = self._down.astype(dtype)
            self._a = mx.exp(self.alpha).astype(dtype)
            self._ib = (1.0 / (mx.exp(self.beta) + 1e-9)).astype(dtype)
            self._cache_dtype = dtype
        return self._w_up, self._w_down, self._a, self._ib

    def __call__(self, x):
        if x.dtype == mx.float32 and _use_fused():
            return self._fused(x)
        return self._ops(x)

    def _fused(self, x):
        """One Metal kernel for upsample -> snake -> downsample (the ops version does ~10 passes
        over memory; this does one, and is where BigVGAN spends most of its time)."""
        B, T, C = x.shape
        w_up, w_down, a, ib = self._weights(mx.float32)
        return _aa_snake_kernel()(
            inputs=[x, a, ib, self._up_flat(), w_down[0, :, 0]],
            grid=(B * T * C, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

    def _up_flat(self):
        if not hasattr(self, "_upf"):
            self._upf = self._up[:2, :, 0].reshape(12)  # [w0 (6), w1 (6)]
        return self._upf

    def _ops(self, x):
        B, T, C = x.shape
        w_up, w_down, a, ib = self._weights(x.dtype)
        # --- 2x upsample (polyphase) ---
        xp = mx.pad(x, [(0, 0), (3, 3), (0, 0)], mode="edge")
        y = mx.conv1d(xp, w_up, stride=1, padding=0, groups=C)  # (B, T+1, 2C)
        y = y.reshape(B, T + 1, C, 2)
        y = mx.stack([y[:, :T, :, 0], y[:, 1:, :, 1]], axis=2).reshape(B, 2 * T, C)
        # --- snake-beta ---
        y = y + ib * mx.square(mx.sin(y * a))
        # --- 2x downsample (replicate pad 5/6, 12-tap, stride 2) ---
        yp = mx.pad(y, [(0, 0), (5, 6), (0, 0)], mode="edge")
        return mx.conv1d(yp, w_down, stride=2, padding=0, groups=C)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * d - d) // 2, dilation=d)
                       for d in dilations]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        self.activations = [AntiAliasedSnakeBeta(channels) for _ in range(2 * len(dilations))]

    def __call__(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = c1(self.activations[2 * i](x))
            xt = c2(self.activations[2 * i + 1](xt))
            x = xt + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        ch0 = cfg["upsample_initial_channel"]
        self.conv_pre = nn.Conv1d(cfg["num_mels"], ch0, 7, padding=3)
        self.ups = []
        self.resblocks = []
        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            cin, cout = ch0 // (2 ** i), ch0 // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(cin, cout, k, stride=u, padding=(k - u) // 2))
            for ks, ds in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(AMPBlock1(cout, ks, ds))
        ch = cout
        self.activation_post = AntiAliasedSnakeBeta(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=cfg["use_bias_at_final"])
        self.use_tanh = cfg["use_tanh_at_final"]

    def __call__(self, mel):
        """mel: (B, frames, 100) log-mel -> (B, frames*256) waveform in [-1, 1]."""
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        x = mx.tanh(x) if self.use_tanh else mx.clip(x, -1.0, 1.0)
        return x[..., 0]


def remap_torch_state(sd: dict) -> dict:
    """Converted (weight-norm folded) torch state dict -> this module's parameter tree.

    torch Conv1d weight (C_out, C_in, k) -> MLX (C_out, k, C_in);
    torch ConvTranspose1d weight (C_in, C_out, k) -> MLX (C_out, k, C_in).
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("ups."):
            _, i, _z, p = k.split(".")  # ups.i.0.{weight,bias}
            out[f"ups.{i}.{p}"] = v.transpose(1, 2, 0) if p == "weight" else v
        elif ".act." in k:  # resblocks.i.activations.j.act.{alpha,beta}
            out[k.replace(".act.", ".")] = v
        elif k.startswith("activation_post.act."):
            out[k.replace(".act.", ".")] = v
        elif k.endswith(".filter"):
            continue  # kaiser-sinc filters are recomputed
        elif k.endswith("weight") and v.ndim == 3:
            out[k] = v.transpose(0, 2, 1)
        else:
            out[k] = v
    return out
