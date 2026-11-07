"""`Renderer` for Apple silicon: the render_core.Renderer pipeline on the MLX backend.

Same inputs, same text front end, same reference bank, same gate/stitch post-processing and
the same defaults (speed 0.90, nfe 64, cfg 3.0, gaps 0.55/0.20) as render_core.Renderer — only
the model execution differs:
  * all pādas of a verse are sampled as ONE batch (padding masked, so each equals its solo render),
  * the DiT runs in bfloat16 by default (fp32 available with dtype=mx.float32),
  * BigVGAN vocodes the DiT's mel directly (upstream decodes with vocos first only to capture
    the mel and to test its loudness; here the loudness test uses the BigVGAN output).
"""
from __future__ import annotations

import json
import os
import re
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import prep_text as PT  # noqa: E402
from chant_text import (  # noqa: E402
    FALLBACK_METER, SR, _aksharas, _anusvara_m, _danda_fix, _ends_halant, _hna_metathesis,
    _rep_depths, _satva, _visarga_ksha, _vocalic_l, detect_meter_key, gate, n_aksharas, split_padas,
)
from .cfm import Reference, Request, Synth, chunk_text  # noqa: E402
from .model import load_bigvgan, load_dit  # noqa: E402

__all__ = ["Renderer", "detect_meter_key", "SR"]

DEFAULT_BANK = os.path.join(SRC, "reference_bank", "bank.json")
CROSS_FADE_S = 0.15
TARGET_RMS = 0.1


class Renderer:
    """Loads the DiT + BigVGAN + the reference bank ONCE; render_one() synthesizes a single input."""

    def __init__(self, model_dir, bank_path=DEFAULT_BANK, speed=0.90, nfe=64, cfg=3.0,
                 gap=0.55, gap_halant=0.20, dtype=mx.bfloat16, log=print):
        self.speed, self.nfe, self.cfg = speed, nfe, cfg
        self.gap, self.gap_halant = gap, gap_halant
        self.log = log
        # Bound MLX's Metal buffer cache: back-to-back renders of different lengths otherwise let it
        # grow until a 16 GB machine swaps. Override with VAGDHENU_MLX_CACHE_MB (0 = unbounded).
        cache_mb = int(os.environ.get("VAGDHENU_MLX_CACHE_MB", "512"))
        if cache_mb > 0:
            (getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(cache_mb << 20)
        dit, vocab = load_dit(model_dir, dtype=dtype)
        self.synth = Synth(dit, vocab, dtype=dtype)
        self.vocoder = load_bigvgan(model_dir, dtype=mx.float32)

        self._bank = json.load(open(bank_path, encoding="utf-8"))
        self._bdir = os.path.dirname(bank_path)
        self._lut = {}
        for k, v in self._bank.items():
            if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                continue
            self._lut[k.lower()] = v
            self._lut[v["wav"].replace(".wav", "").lower()] = v
        self._primes = self._bank.get("repeat_primes", {})
        self._refcache: dict[str, tuple[Reference, float]] = {}

    def meters(self):
        return [k for k, v in self._bank.items()
                if not k.startswith("_") and isinstance(v, dict) and "wav" in v]

    def warmup(self):
        """Compile/allocate once so the first real request is not slow."""
        m = self.meters()[0]
        self.render_one(self._lut[m.lower()]["full_verse"].split("।")[0], m, seed=0)

    # ------------------------------------------------------------------ references
    def _load_ref(self, wav, ref_text) -> Reference:
        return Reference.load(os.path.join(self._bdir, wav), ref_text, clip_short=True, log=self.log)

    def _get_ref(self, meter):
        key = meter.lower().replace(".wav", "")
        if key in self._refcache:
            return self._refcache[key]
        if key not in self._lut:
            if FALLBACK_METER not in self._lut:
                raise ValueError(f"meter '{meter}' not in bank (and fallback '{FALLBACK_METER}' missing)")
            self.log(f"[meter] unknown vṛtta '{meter}' -> fallback '{FALLBACK_METER}'", flush=True)
            orig, key = key, FALLBACK_METER
            if key in self._refcache:
                self._refcache[orig] = self._refcache[key]
                return self._refcache[key]
        e = self._lut[key]
        val = (self._load_ref(e["wav"], e["ref_text"]), float(e.get("sec_per_syll", 0.26)))
        self._refcache[key] = val
        return val

    def _get_prime(self, name) -> Reference:
        key = "prime:" + name
        if key not in self._refcache:
            p = self._primes[name]
            self._refcache[key] = (self._load_ref(p["wav"], p["ref_text"]), 0.0)
        return self._refcache[key][0]

    # ------------------------------------------------------------------ model
    def _vocode(self, mel: np.ndarray) -> np.ndarray:
        y = self.vocoder(mx.array(mel[None]))
        y = np.asarray(y, dtype=np.float32)[0]
        return y

    def _infer(self, ref: Reference, pieces: list[str], fix_durations: list[float | None],
               seeds: list[int], speed: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """Port of infer_process for several pieces at once. Returns [(wave, mel)] per piece.
        Each piece is chunked like upstream (max_chars rule), chunks share the batch."""
        max_chars = int(len(ref.text.encode("utf-8")) / ref.duration_s * (25 - ref.duration_s))
        reqs, owner = [], []
        for i, (p, fd, sd) in enumerate(zip(pieces, fix_durations, seeds)):
            for j, chunk in enumerate(chunk_text(p, max_chars=max_chars)):
                dur = Synth.fixed_duration(fd) if fd is not None else Synth.estimate_duration(ref, chunk, speed)
                reqs.append(Request(ref, chunk, duration=dur, seed=sd * 1000003 + j))
                owner.append(i)
        mels = self.synth.generate(reqs, steps=self.nfe, cfg_strength=self.cfg, sway_coef=-1.0)
        out = []
        for i in range(len(pieces)):
            waves, chunk_mels = [], []
            for r_idx in [k for k, o in enumerate(owner) if o == i]:
                m = mels[r_idx]
                w = self._vocode(m)
                if ref.rms < TARGET_RMS:
                    w = w * (ref.rms / TARGET_RMS)
                waves.append(w); chunk_mels.append(m)
            out.append((_cross_fade(waves), np.concatenate(chunk_mels, axis=0)))
        return out

    # ------------------------------------------------------------------ public API
    def _stitch(self, segs, GAPS, fric=False, halant=False):
        if len(segs) == 1:
            return gate(segs[0], fric=fric, halant=halant)
        b = []; last = len(segs) - 1
        for i, s in enumerate(segs):
            b += [gate(s, fric=(fric and i == 0), halant=(halant and i == last)),
                  GAPS[i] if i < len(GAPS) else GAPS[-1]]
        return np.concatenate(b[:-1])

    def render_one(self, text, meter, seed=60, no_sandhi=True, speed=None, sps=None):
        """Synthesize one shloka. text = free Devanagari (split into padas on newline/danda).
        Returns (sr, audio float32). Pipeline is identical to render_core.Renderer.render_one()."""
        padas = text if isinstance(text, list) else split_padas(text)
        if not padas:
            raise ValueError("empty text")
        ref, ref_sps = self._get_ref(meter)
        if sps is not None:
            ref_sps = float(sps)
        spd = float(speed) if speed is not None else self.speed

        def _basetext(p):
            return PT.model_text_sandhi(p, echo_final=False) if not no_sandhi else PT.model_text(p)
        PIECES = [_basetext(p) for p in padas]
        if not no_sandhi:
            PIECES = [_satva(x) for x in PIECES]
        PIECES = [_danda_fix(_visarga_ksha(_anusvara_m(x))) for x in PIECES]
        PIECES = [_hna_metathesis(x) for x in PIECES]
        PIECES = [_vocalic_l(x) for x in PIECES]

        _mono = max((_rep_depths(_aksharas(x))[0] for x in PIECES), default=1)
        _di = max((_rep_depths(_aksharas(x))[1] for x in PIECES), default=1)
        _pick = None
        if _di >= 3:
            _pick = next((k for k in ["prime_jaya", "prime_chata"]
                          if k in self._primes and self._primes[k].get("di_max", 0) >= _di), None) \
                or next((k for k, v in self._primes.items()
                         if isinstance(v, dict) and v.get("di_max", 0) >= _di), None)
        if _pick is None and _mono >= 2 and "prime_mono" in self._primes \
                and self._primes["prime_mono"].get("mono_max", 0) >= _mono:
            _pick = "prime_mono"
        if _pick:
            ref = self._get_prime(_pick)

        NSYLL = [n_aksharas(x) for x in PIECES]
        GAPS = [np.zeros(int(self.gap * SR) + (int(self.gap_halant * SR) if _ends_halant(p) else 0),
                         dtype=np.float32) for p in PIECES]
        fixd = [(ref.duration_s + n * ref_sps) if (ref_sps > 0 and NSYLL) else None for n in NSYLL]

        # attempt 0 for every pāda in one batch; retry quiet ones (RMS <= 0.04) with seed+att
        results = self._infer(ref, PIECES, fixd, [seed] * len(PIECES), spd)
        final = [None] * len(PIECES)
        for i, (w, m) in enumerate(results):
            if float(np.sqrt((w ** 2).mean())) > 0.04:
                final[i] = w
        for i in range(len(PIECES)):
            if final[i] is not None:
                continue
            w = results[i][0]
            for att in range(1, 4):
                w, _ = self._infer(ref, [PIECES[i]], [fixd[i]], [seed + att], spd)[0]
                if float(np.sqrt((w ** 2).mean())) > 0.04:
                    break
            final[i] = w
        bseg = []
        for y in final:
            mxv = np.abs(y).max()
            bseg.append(y / mxv * 0.97 if mxv > 1 else y)

        _slp = PT.align_slp1(padas[0])
        fric = bool(_slp) and _slp[0] in ("S", "z", "s", "h")
        halant = _ends_halant(PIECES[-1])
        return SR, self._stitch(bseg, GAPS, fric=fric, halant=halant)


def _cross_fade(waves: list[np.ndarray], dur_s: float = CROSS_FADE_S) -> np.ndarray:
    final = waves[0]
    for nxt in waves[1:]:
        n = min(int(dur_s * SR), len(final), len(nxt))
        if n <= 0:
            final = np.concatenate([final, nxt]); continue
        mix = final[-n:] * np.linspace(1, 0, n) + nxt[:n] * np.linspace(0, 1, n)
        final = np.concatenate([final[:-n], mix, nxt[n:]]).astype(np.float32)
    return final
