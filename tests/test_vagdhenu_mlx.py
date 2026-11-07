"""Unit tests for the MLX backend that need no weights and no torch:  pytest tests/

They run on MLX's CPU stream (exact fp32) so the checks are tight. The torch-vs-MLX parity suite
(needs the torch reference env and the weights) is tools/parity_mlx.py.
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vagdhenu_mlx import audio as A, bigvgan as BV, cfm as C, dit as D, text as T  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


# ---------------------------------------------------------------- text
@pytest.mark.parametrize("s, expected", [
    ("abc def", list("abc") + [" "] + list("def")),
    ("ab,cd", ["a", "b", ",", " ", "c", "d"]),              # jieba: space before an ASCII word run
    ("a,b", ["a", ",", "b"]),
    ("x:y 'q' “z”; ok", list("x:y 'q' \"z\", ") + ["o", "k"]),
    ("सन्ततं चिन्तयेत्.  ಸಂತತಂ", list("सन्ततं चिन्तयेत्.  ಸಂತತಂ")),  # Indic: one token per code point
    ("12.5% off", ["1", "2", ".", "5", "%", " ", "o", "f", "f"]),
])
def test_tokenizer_matches_f5_char_pipeline(s, expected):
    # expectations frozen from f5_tts.model.utils.convert_char_to_pinyin (see tools/parity_mlx.py)
    assert T.to_char_list(s) == expected


def test_tokenizer_rejects_cjk():
    with pytest.raises(ValueError):
        T.to_char_list("你好")


def test_load_vocab(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text(" \na\nब\n", encoding="utf-8")
    v = T.load_vocab(str(p))
    assert v == {" ": 0, "a": 1, "ब": 2}
    assert T.tokenize("aब?", v) == [1, 2, 0]


def test_chunk_text():
    assert C.chunk_text("abc") == ["abc"]
    assert C.chunk_text("one, two. three", max_chars=8) == ["one,", "two.", "three"]


# ---------------------------------------------------------------- audio
def test_mel_filterbank_shape_and_support():
    fb = A.mel_filterbank()
    assert fb.shape == (513, 100)
    assert (fb >= 0).all() and fb[:, 0].max() > 0 and fb[0, :].sum() == 0


def test_log_mel_frames_center_true():
    wave = np.random.RandomState(0).randn(24000 * 2).astype(np.float32) * 0.1
    m = A.log_mel(wave)
    assert m.shape == (24000 * 2 // 256 + 1, 100)
    assert m.min() >= np.log(1e-5) - 1e-6


def test_normalize_rms():
    w = np.full(1000, 0.01, np.float32)
    out, rms = A.normalize_rms(w)
    assert abs(rms - 0.01) < 1e-6 and abs(float(np.sqrt((out ** 2).mean())) - 0.1) < 1e-6


# ---------------------------------------------------------------- BigVGAN activation
def _ref_activation(x, alpha, beta):
    """Direct numpy transcription of NVIDIA's Activation1d(SnakeBeta) torch path, (T, C)."""
    T_, C_ = x.shape
    f = BV.kaiser_sinc_filter1d(0.25, 0.3, 12)
    out = np.zeros((2 * T_, C_))
    xp = np.pad(x, ((5, 5), (0, 0)), mode="edge")  # replicate pad 5
    # conv_transpose1d stride 2 then crop [15:-15]
    yt = np.zeros((2 * (T_ + 10 - 1) + 12, C_))
    for i in range(T_ + 10):
        yt[2 * i: 2 * i + 12] += xp[i][None, :] * f[:, None]
    out = 2 * yt[15:-15]
    a, b = np.exp(alpha), np.exp(beta)
    s = out + (1.0 / (b + 1e-9)) * np.sin(out * a) ** 2
    sp = np.pad(s, ((5, 6), (0, 0)), mode="edge")
    y = np.zeros((T_, C_))
    for n in range(T_):
        y[n] = (sp[2 * n: 2 * n + 12] * f[:, None]).sum(0)
    return y


def test_anti_aliased_snakebeta_matches_reference():
    rng = np.random.RandomState(1)
    C_, T_ = 5, 37
    act = BV.AntiAliasedSnakeBeta(C_)
    act.alpha = mx.array(rng.randn(C_).astype(np.float32) * 0.3)
    act.beta = mx.array(rng.randn(C_).astype(np.float32) * 0.3)
    x = rng.randn(1, T_, C_).astype(np.float32)
    got = np.asarray(act._ops(mx.array(x)))[0]
    ref = _ref_activation(x[0], np.asarray(act.alpha), np.asarray(act.beta))
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="fused kernel needs Metal")
def test_fused_kernel_matches_ops():
    rng = np.random.RandomState(2)
    C_, T_ = 8, 200
    act = BV.AntiAliasedSnakeBeta(C_)
    act.alpha = mx.array(rng.randn(C_).astype(np.float32) * 0.3)
    act.beta = mx.array(rng.randn(C_).astype(np.float32) * 0.3)
    x = mx.array(rng.randn(2, T_, C_).astype(np.float32) * 3)
    ref = np.asarray(act._ops(x))
    with mx.stream(mx.gpu):
        got = np.asarray(act._fused(x))
    assert np.abs(got - ref).max() < 1e-5 * np.abs(ref).max()


def test_bigvgan_output_length():
    small = dict(BV.CFG, upsample_initial_channel=128)
    voc = BV.BigVGAN(small)
    mel = mx.zeros((1, 7, 100))
    assert voc(mel).shape == (1, 7 * 256)


# ---------------------------------------------------------------- DiT + sampler (tiny random model)
def _tiny_dit():
    mx.random.seed(0)
    return D.DiT(text_num_embeds=20, dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, text_dim=32, conv_layers=1, mel_dim=100)


def test_dit_batched_equals_single():
    dit = _tiny_dit()
    N1, N2 = 50, 33
    x1, x2 = mx.random.normal((1, N1, 100)), mx.random.normal((1, N2, 100))
    c1, c2 = mx.random.normal((1, N1, 100)), mx.random.normal((1, N2, 100))
    t1 = mx.array(np.random.RandomState(0).randint(1, 21, (1, N1)).astype(np.int32))
    t2 = mx.array(np.random.RandomState(1).randint(1, 21, (1, N2)).astype(np.int32))
    mods, fin = dit.time_modulations(mx.array([0.4]))
    s1 = dit.step(x1, dit.context(c1, t1), mods[0], fin[0])
    s2 = dit.step(x2, dit.context(c2, t2), mods[0], fin[0])
    pad = lambda a, n: mx.pad(a, [(0, 0), (0, n - a.shape[1]), (0, 0)])  # noqa: E731
    xb = mx.concatenate([x1, pad(x2, N1)]); cb = mx.concatenate([c1, pad(c2, N1)])
    tb = mx.concatenate([t1, mx.pad(t2, [(0, 0), (0, N1 - N2)])])
    valid = np.zeros((2, N1, 1), np.float32); valid[0] = 1; valid[1, :N2] = 1
    am = mx.where(mx.array(valid[:, None, :, 0]) > 0, 0.0, -1e9)[:, None]
    out = dit.step(xb, dit.context(cb, tb, mx.array(valid)), mods[0], fin[0], mx.array(valid), am)
    assert np.abs(np.asarray(out[0]) - np.asarray(s1[0])).max() < 1e-4
    assert np.abs(np.asarray(out[1, :N2]) - np.asarray(s2[0])).max() < 1e-4


def test_sampler_lengths_and_cond_infill():
    dit = _tiny_dit()
    vocab = {chr(97 + i): i for i in range(20)}
    wave = (np.random.RandomState(3).randn(24000) * 0.05).astype(np.float32)
    ref = C.Reference(wave=wave, text="abc. ", rms=0.05, mel=A.log_mel(wave), duration_s=1.0)
    synth = C.Synth(dit, vocab, dtype=mx.float32, compile=False)
    dur = C.Synth.fixed_duration(1.7)
    out = synth.generate([C.Request(ref, "defg", duration=dur, seed=1), C.Request(ref, "hi", duration=dur - 5, seed=2)],
                         steps=3, cfg_strength=2.0)
    assert out[0].shape == (dur - ref.frames, 100) and out[1].shape == (dur - 5 - ref.frames, 100)
    # frame ref.frames (the last conditioning frame, center=True STFT) is copied from the reference mel
    assert np.allclose(out[0][0], ref.mel[ref.frames], atol=1e-6)
    assert np.isfinite(out[0]).all()
