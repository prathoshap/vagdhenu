"""Audio front end without torch: reference-clip preprocessing and the vocos log-mel.

* ``preprocess_ref`` = f5_tts.infer.utils_infer.preprocess_ref_audio_text(clip_short=True) +
  the RMS normalisation in infer_batch_process. It uses the same pydub silence logic (pydub
  reads/writes WAV natively, so no ffmpeg is needed).
* ``log_mel`` = torchaudio MelSpectrogram(n_fft=1024, hop=256, n_mels=100, power=1,
  center=True, reflect pad, periodic Hann, HTK mel scale, no norm) -> clamp(1e-5).log().
  This is computed once per reference clip and then cached, so plain float64 numpy is fine.
"""
from __future__ import annotations

import functools

import numpy as np

SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 100
TARGET_RMS = 0.1


@functools.lru_cache(maxsize=None)
def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, f_min=0.0, f_max=None):
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale='htk'), shape (n_freqs, n_mels)."""
    f_max = f_max or sr / 2
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sr // 2, n_freqs)
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_pts = np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2)
    f_pts = mel2hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


def log_mel(wave: np.ndarray) -> np.ndarray:
    """(T,) float waveform at 24 kHz -> (frames, 100) float32 log-mel, frames = T//256 + 1."""
    x = np.asarray(wave, dtype=np.float64)
    pad = N_FFT // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=-1))
    mel = spec @ mel_filterbank()
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)


def _pydub():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Couldn't find ffmpeg" — not needed for WAV
        from pydub import AudioSegment, silence
    return AudioSegment, silence


def _remove_silence_edges(audio, silence_threshold=-42):
    _, silence = _pydub()
    start = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[start:]
    dur = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        dur -= 0.001
    return audio[: int(dur * 1000)]


def preprocess_ref(path: str, ref_text: str, clip_short: bool = True, log=print):
    """Returns (wave float32 at 24 kHz, ref_text) exactly as upstream would feed the model."""
    AudioSegment, silence = _pydub()
    aseg = AudioSegment.from_file(path)
    if clip_short:
        segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50,
                                        keep_silence=1000, seek_step=10)
        wave = AudioSegment.silent(duration=0)
        for s in segs:
            if len(wave) > 6000 and len(wave + s) > 15000:
                log("Audio is over 15s, clipping short. (1)")
                break
            wave += s
        if len(wave) > 15000:
            segs = silence.split_on_silence(aseg, min_silence_len=100, silence_thresh=-40,
                                            keep_silence=1000, seek_step=10)
            wave = AudioSegment.silent(duration=0)
            for s in segs:
                if len(wave) > 6000 and len(wave + s) > 15000:
                    log("Audio is over 15s, clipping short. (2)")
                    break
                wave += s
        aseg = wave
        if len(aseg) > 15000:
            aseg = aseg[:15000]
            log("Audio is over 15s, clipping short. (3)")
    aseg = _remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    # torchaudio.load(...) of the exported 16-bit wav == int16 / 32768, channels averaged later
    aseg = aseg.set_sample_width(2)
    sr = aseg.frame_rate
    samples = np.array(aseg.get_array_of_samples(), dtype=np.float32) / 32768.0
    if aseg.channels > 1:
        samples = samples.reshape(-1, aseg.channels).mean(axis=1)
    if sr != SR:
        samples = resample(samples, sr, SR)

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + (" " if ref_text.endswith(".") else ". ")
    return samples.astype(np.float32), ref_text


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling (the bank is 24 kHz already; this is for user-supplied clips)."""
    from math import gcd
    try:
        from scipy.signal import resample_poly
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(f"reference audio is {sr_in} Hz; install scipy or supply 24 kHz audio") from e
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def normalize_rms(wave: np.ndarray, target_rms: float = TARGET_RMS):
    """infer_batch_process: quiet references are scaled up to target_rms; returns (wave, rms)."""
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    if rms < target_rms:
        wave = wave * (target_rms / rms)
    return wave.astype(np.float32), rms
