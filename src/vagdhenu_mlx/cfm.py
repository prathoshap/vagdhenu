"""Flow-matching sampler + reference conditioning (port of f5_tts CFM.sample / infer_process).

`Synth.generate` does, for a list of (reference, text, duration) requests, exactly what
`infer_process(...)` → `CFM.sample(...)` does for each of them in turn (Euler ODE on the
sway-sampled time grid, classifier-free guidance, cond infill), then returns the generated mel
frames (the part after the reference). Items are rendered as one batch; padding is masked
so each item matches its single-item result.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import audio as A
from .dit import DiT
from .text import tokenize

HOP = A.HOP
SR = A.SR
MAX_DURATION = 4096


@dataclass
class Reference:
    """A preprocessed reference clip: what F5 conditions on."""
    wave: np.ndarray          # float32, 24 kHz, RMS-normalised to >= 0.1
    text: str                 # ref_text with the ". " / "  " suffix rules applied
    rms: float                # RMS before normalisation (used to scale the output back down)
    mel: np.ndarray           # (frames, 100) float32 log-mel of `wave`
    duration_s: float         # len(wave)/SR

    @property
    def frames(self) -> int:
        """ref_audio_len as infer_batch_process defines it (wave // hop): where the generated
        part starts. Note the conditioning mel has one frame more (center=True STFT)."""
        return len(self.wave) // HOP

    @property
    def cond_frames(self) -> int:
        return self.mel.shape[0]

    @classmethod
    def load(cls, wav_path: str, ref_text: str, clip_short: bool = True, log=lambda *_: None):
        wave, text = A.preprocess_ref(wav_path, ref_text, clip_short=clip_short, log=log)
        dur = len(wave) / SR
        wave, rms = A.normalize_rms(wave)
        if len(text[-1].encode("utf-8")) == 1:
            text = text + " "
        return cls(wave=wave, text=text, rms=rms, mel=A.log_mel(wave), duration_s=dur)


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    """f5_tts.infer.utils_infer.chunk_text, verbatim."""
    chunks, cur = [], ""
    for s in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text):
        if len(cur.encode("utf-8")) + len(s.encode("utf-8")) <= max_chars:
            cur += s + " " if s and len(s[-1].encode("utf-8")) == 1 else s
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s + " " if s and len(s[-1].encode("utf-8")) == 1 else s
    if cur:
        chunks.append(cur.strip())
    return chunks


def sway_time_grid(steps: int, coef: float | None = -1.0) -> np.ndarray:
    t = np.linspace(0, 1, steps + 1, dtype=np.float64)
    if coef is not None:
        t = t + coef * (np.cos(np.pi / 2 * t) - 1 + t)
    return t


@dataclass
class Request:
    ref: Reference
    gen_text: str
    duration: int | None = None   # total frames (ref + gen); None -> text-ratio estimate
    noise: np.ndarray | None = None  # (duration, 100) optional explicit y0 (parity tests)
    seed: int | None = None


class Synth:
    def __init__(self, dit: DiT, vocab: dict[str, int], dtype=mx.bfloat16, compile: bool = True):
        self.dit = dit
        self.vocab = vocab
        self.dtype = dtype
        self._compile = compile
        self._steps: dict[bool, object] = {}

    def _step_fn(self, masked: bool):
        """The DiT step under mx.compile (fuses the elementwise work between the matmuls, ~15%).
        MLX re-traces per input shape; a new (batch, length) costs one trace, then it is cached."""
        if not self._compile:
            return self.dit.step
        if masked not in self._steps:
            if masked:
                fn = lambda x, ctx, m, f, valid, am: self.dit.step(x, ctx, m, f, valid, am)  # noqa: E731
            else:
                fn = lambda x, ctx, m, f: self.dit.step(x, ctx, m, f)  # noqa: E731
            self._steps[masked] = mx.compile(fn)
        return self._steps[masked]

    # ------------------------------------------------------------------ duration (upstream rules)
    @staticmethod
    def estimate_duration(ref: Reference, gen_text: str, speed: float = 1.0) -> int:
        ref_len = len(ref.wave) // HOP
        return ref_len + int(ref_len / len(ref.text.encode("utf-8")) * len(gen_text.encode("utf-8")) / speed)

    @staticmethod
    def fixed_duration(seconds: float) -> int:
        return int(seconds * SR / HOP)

    # ------------------------------------------------------------------ batched sampling
    def generate(self, reqs: list[Request], steps: int = 32, cfg_strength: float = 2.0,
                 sway_coef: float | None = -1.0) -> list[np.ndarray]:
        """Returns, per request, the generated mel (gen_frames, 100) float32 (reference part removed)."""
        B = len(reqs)
        # --- per-item lengths, exactly as CFM.sample computes them ---
        toks, durs, lens, cond_len = [], [], [], []
        for r in reqs:
            ids = tokenize(r.ref.text + r.gen_text, self.vocab)
            cl = r.ref.cond_frames
            ln = max(len(ids), cl)
            d = r.duration if r.duration is not None else self.estimate_duration(r.ref, r.gen_text)
            d = min(max(ln + 1, d), MAX_DURATION)
            toks.append(ids); durs.append(d); lens.append(ln); cond_len.append(cl)
        N = max(durs)

        # --- padded batch tensors ---
        tok = np.zeros((B, N), dtype=np.int32)          # +1 shift, 0 = filler
        cond = np.zeros((B, N, A.N_MELS), dtype=np.float32)
        cond_mask = np.zeros((B, N, 1), dtype=bool)
        valid = np.zeros((B, N, 1), dtype=np.float32)
        y0 = np.zeros((B, N, A.N_MELS), dtype=np.float32)
        for i, r in enumerate(reqs):
            ids = toks[i][: durs[i]]
            tok[i, : len(ids)] = np.asarray(ids, dtype=np.int32) + 1
            cond[i, : cond_len[i]] = r.ref.mel[: cond_len[i]]
            cond_mask[i, : lens[i]] = True
            valid[i, : durs[i]] = 1.0
            if r.noise is not None:
                y0[i, : durs[i]] = r.noise[: durs[i]]
            else:
                key = mx.random.key(r.seed) if r.seed is not None else None
                y0[i, : durs[i]] = np.asarray(mx.random.normal((durs[i], A.N_MELS), key=key))
        step_cond = np.where(cond_mask, cond, 0.0).astype(np.float32)

        dt = self.dtype
        tok_mx = mx.array(tok)
        valid_mx = mx.array(valid).astype(dt) if B > 1 else None
        attn_mask = None
        if B > 1:
            # (B, 1, 1, N) additive mask over keys, broadcast over heads/queries; duplicated for CFG
            am = mx.where(mx.array(valid[:, None, :, 0]) > 0, 0.0, -1e9).astype(dt)[:, None]
            attn_mask = mx.concatenate([am, am], axis=0)
            valid_mx = mx.concatenate([valid_mx, valid_mx], axis=0)

        # --- constants of the ODE: context for the conditional and the null (CFG) branch ---
        sc = mx.array(step_cond).astype(dt)
        ctx_c = self.dit.context(sc, tok_mx, valid_mx[:B] if valid_mx is not None else None)
        ctx_n = self.dit.context(mx.zeros_like(sc), mx.zeros_like(tok_mx),
                                 valid_mx[:B] if valid_mx is not None else None)
        ctx = mx.concatenate([ctx_c, ctx_n], axis=0)
        ts = sway_time_grid(steps, sway_coef)
        mods, fin = self.dit.time_modulations(mx.array(ts[:-1].astype(np.float32)))
        mx.eval(ctx, mods, fin)

        # --- Euler integration ---
        x = mx.array(y0)
        step = self._step_fn(valid_mx is not None)
        for k in range(steps):
            xin = mx.concatenate([x, x], axis=0).astype(dt)
            if valid_mx is None:
                pred = step(xin, ctx, mods[k], fin[k]).astype(mx.float32)
            else:
                pred = step(xin, ctx, mods[k], fin[k], valid_mx, attn_mask).astype(mx.float32)
            p, n = pred[:B], pred[B:]
            v = p + (p - n) * cfg_strength if cfg_strength >= 1e-5 else p
            x = x + float(ts[k + 1] - ts[k]) * v
            mx.eval(x)

        out = mx.where(mx.array(cond_mask), mx.array(cond), x)
        out = np.asarray(out.astype(mx.float32))
        return [out[i, r.ref.frames: durs[i]] for i, r in enumerate(reqs)]
