"""Staged self-test for a Vāgdhenu install — run it first on the GPU box, then on the Mac.

Each stage prints PASS / FAIL / SKIP and the script exits non-zero if any stage failed.
Stages 1-3 need no model weights, so they are the fast loop while setting an environment up.

    python scripts/selftest.py                      # everything
    python scripts/selftest.py --stages 1 2 3       # no weights needed
    python scripts/selftest.py --nfe 32             # the serving config, ~2x faster than the default
    python scripts/selftest.py --compare out/gpu_reference.wav

Stage 2 is the one that matters on Apple Silicon: it re-runs the FFT kernels in a child process
with PYTORCH_ENABLE_MPS_FALLBACK=0, so a kernel Metal does not implement raises instead of quietly
running on CPU. That tells you whether the fallback is load-bearing and where the time is going.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

CHAMP = os.environ.get("CHAMP_ROOT", os.path.join(REPO, "models"))
VOICE = os.path.join(CHAMP, "voice_steer_ema_2026-06-17.pt")
VOC = os.path.join(CHAMP, "voc_bigvgan_EMA_2026-06-11.pth")
BANK = os.path.join(SRC, "reference_bank", "bank.json")
VOCAB = os.path.join(SRC, "reference_bank", "vocab.txt")

# The verse from examples/sample_shard.json — a known-good anuṣṭubh.
VERSE = "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"
SEED = 60

_results = []


def stage(num, title):
    def deco(fn):
        fn._stage = (num, title)
        return fn
    return deco


def _run(fn, args):
    num, title = fn._stage
    print(f"\n── stage {num}: {title} " + "─" * max(0, 58 - len(title)))
    t0 = time.time()
    try:
        note = fn(args)
        verdict = "SKIP" if note and note.startswith("skipped") else "PASS"
    except Exception as exc:  # a failed stage should not hide the stages after it
        verdict = "FAIL"
        note = f"{type(exc).__name__}: {exc}"
    dt = time.time() - t0
    print(f"   {verdict}  {note or ''}   [{dt:.1f}s]")
    _results.append((num, title, verdict, note or ""))
    return verdict


# ── stage 1 ────────────────────────────────────────────────────────────────────────────
@stage(1, "environment and backend selection")
def s1(args):
    from device import pick_device, describe

    import torch

    print(f"   python          {sys.version.split()[0]}")
    print(f"   torch           {torch.__version__}")
    print(f"   platform        {sys.platform} {os.uname().machine}")
    dev = pick_device()
    print(f"   backend         {describe(dev)}")
    print(f"   mps fallback    PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"   ffmpeg          {shutil.which('ffmpeg') or 'NOT FOUND (brew install ffmpeg)'}")

    try:
        import bigvgan  # noqa: F401
        print("   bigvgan         importable")
    except ImportError:
        raise RuntimeError("`import bigvgan` failed — export PYTHONPATH=$PWD/BigVGAN:$PYTHONPATH")

    for label, path in (("voice", VOICE), ("vocoder", VOC), ("bank", BANK), ("vocab", VOCAB)):
        print(f"   {label:<15} {'ok ' if os.path.exists(path) else 'MISSING '}{path}")

    if sys.version_info[:2] != (3, 10):
        print(f"   note: project is validated on python 3.10, this is {sys.version_info[0]}.{sys.version_info[1]}")
    return f"backend={dev}"


# ── stage 2 ────────────────────────────────────────────────────────────────────────────
_PROBE = r'''
import json, sys, torch
dev = sys.argv[1]
out = {}
def probe(name, fn):
    try:
        fn(); out[name] = "native"
    except NotImplementedError as e:
        out[name] = "missing"
    except Exception as e:
        out[name] = f"error: {type(e).__name__}: {e}"[:120]

x = torch.randn(1, 24000, device=dev)
win = torch.hann_window(1024, device=dev)
spec = [None]

def _stft():
    spec[0] = torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                         window=win, return_complex=True)
def _istft():
    s = spec[0] if spec[0] is not None else torch.stft(
        x.cpu(), n_fft=1024, hop_length=256, win_length=1024,
        window=win.cpu(), return_complex=True).to(dev)
    torch.istft(s, n_fft=1024, hop_length=256, win_length=1024, window=win)
def _mel():
    import torchaudio
    torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=100).to(dev)(x)
def _sdpa():
    q = torch.randn(1, 16, 128, 64, device=dev)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
def _convt():
    torch.nn.ConvTranspose1d(100, 50, 8, stride=4).to(dev)(torch.randn(1, 100, 64, device=dev))

probe("torch.stft (mel front-end)", _stft)
probe("torch.istft (vocos head)", _istft)
probe("torchaudio MelSpectrogram", _mel)
probe("scaled_dot_product_attention (DiT)", _sdpa)
probe("ConvTranspose1d (BigVGAN)", _convt)
print("@@" + json.dumps(out))
'''


@stage(2, "kernel coverage on the selected backend")
def s2(args):
    from device import pick_device

    dev = pick_device()
    if dev.startswith("cpu"):
        return "skipped — CPU backend implements everything"

    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0")
    proc = subprocess.run([sys.executable, "-c", _PROBE, dev],
                          capture_output=True, text=True, env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(f"probe process produced no result:\n{proc.stdout}\n{proc.stderr}")

    coverage = json.loads(line[2:])
    missing = []
    for name, verdict in coverage.items():
        mark = {"native": "native ", "missing": "CPU-FALLBACK"}.get(verdict, "?")
        print(f"   {mark:<13} {name}" + ("" if verdict in ("native", "missing") else f"  ({verdict})"))
        if verdict != "native":
            missing.append(name)

    if missing:
        print(f"   -> {len(missing)} kernel(s) run on CPU via PYTORCH_ENABLE_MPS_FALLBACK.")
        print("      Correct, but each one costs a device round-trip. Expect a slower RTF in stage 6.")
        return f"{len(missing)} kernel(s) fall back to CPU"
    return "all probed kernels native"


# ── stage 3 ────────────────────────────────────────────────────────────────────────────
@stage(3, "text frontend and meter detection (device independent)")
def s3(args):
    import prep_text as PT
    from render_core import detect_meter_key, split_padas, n_aksharas

    model = PT.model_text(VERSE)
    print(f"   deva in         {VERSE}")
    print(f"   model text      {model}")
    print(f"   script detected {PT.detect_script(VERSE)}")
    print(f"   aksharas        {n_aksharas(VERSE)}")
    print(f"   padas           {split_padas(VERSE)}")

    if not model.strip():
        raise RuntimeError("model_text() returned empty — the frontend is broken, not the backend")
    if PT.detect_script(VERSE) != "devanagari":
        raise RuntimeError("script detection failed on a plain Devanagari verse")

    meter = detect_meter_key(VERSE)
    print(f"   meter           {meter or '(none — falls back to vasantatilaka)'}")

    # This stage is pure Python; it must give identical output on every platform. Print a digest
    # so a Mac run can be diffed against a GPU-box run without shipping wavs around.
    import hashlib
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    print(f"   frontend digest {digest}   (must match across platforms)")
    return f"digest={digest}"


# ── stage 4 ────────────────────────────────────────────────────────────────────────────
_RENDERER = None


@stage(4, "model load onto the backend")
def s4(args):
    global _RENDERER
    for path in (VOICE, VOC, BANK, VOCAB):
        if not os.path.exists(path):
            return f"skipped — missing {path} (run scripts/download_weights.py)"

    from render_core import Renderer

    t0 = time.time()
    _RENDERER = Renderer(VOICE, VOC, BANK, vocab_file=VOCAB, nfe=args.nfe)
    print(f"   loaded onto     {_RENDERER.device} in {time.time() - t0:.1f}s")
    print(f"   meters in bank  {len(_RENDERER.meters())}")

    import torch

    n_params = sum(p.numel() for p in _RENDERER.cfm.parameters())
    print(f"   DiT parameters  {n_params / 1e6:.0f}M")
    dev_of_weights = str(next(_RENDERER.cfm.parameters()).device)
    print(f"   weights live on {dev_of_weights}")
    if not dev_of_weights.startswith(_RENDERER.device.split(":")[0]):
        raise RuntimeError(f"weights landed on {dev_of_weights}, expected {_RENDERER.device}")
    return f"loaded on {_RENDERER.device}"


# ── stage 5 ────────────────────────────────────────────────────────────────────────────
@stage(5, "render one verse and check the audio is sane")
def s5(args):
    if _RENDERER is None:
        return "skipped — no model loaded (stage 4 did not run)"

    import numpy as np

    t0 = time.time()
    sr, audio = _RENDERER.render_one(VERSE, meter="anushtubh", seed=SEED)
    elapsed = time.time() - t0
    audio = np.asarray(audio, dtype=np.float32)

    duration = len(audio) / sr
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    print(f"   sample rate     {sr}")
    print(f"   duration        {duration:.2f}s")
    print(f"   peak / rms      {peak:.3f} / {rms:.4f}")
    print(f"   wall clock      {elapsed:.1f}s")

    if not np.isfinite(audio).all():
        raise RuntimeError("output contains NaN or inf — a kernel misbehaved on this backend")
    if duration < 1.0:
        raise RuntimeError(f"output is {duration:.2f}s, far too short for this verse")
    if rms < 0.01:
        raise RuntimeError(f"output is effectively silent (rms {rms:.5f})")
    if peak > 1.5:
        raise RuntimeError(f"output is not normalized (peak {peak:.2f})")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"selftest_{_RENDERER.device.replace(':', '')}_nfe{args.nfe}.wav")
    import soundfile as sf

    sf.write(out, audio, sr)
    print(f"   wrote           {out}")
    print("   -> LISTEN TO IT. Numbers cannot tell you the chant is musically right.")

    s5.audio = (sr, audio, elapsed, out)
    return f"{duration:.2f}s of audio in {elapsed:.1f}s"


# ── stage 6 ────────────────────────────────────────────────────────────────────────────
@stage(6, "real-time factor and cross-device comparison")
def s6(args):
    got = getattr(s5, "audio", None)
    if got is None:
        return "skipped — nothing was rendered"
    sr, audio, elapsed, out = got

    import numpy as np

    duration = len(audio) / sr
    rtf = elapsed / duration
    print(f"   RTF             {rtf:.2f}  (compute seconds per audio second, lower is better)")
    print(f"   reference       A6000 nfe64 = 1.24, A6000 nfe32 = 0.63, CPU = ~130+")
    print(f"   a 40-verse adhyaya (~80 hemistichs) ≈ {rtf * duration * 80 / 60:.0f} min at this rate")

    if not args.compare:
        print("   no --compare reference given; to compare against the GPU box:")
        print("     scp gpubox:.../selftest_cuda_nfe%d.wav ." % args.nfe)
        print("     python scripts/selftest.py --stages 6 --compare selftest_cuda_nfe%d.wav" % args.nfe)
        return f"RTF {rtf:.2f}"

    import soundfile as sf

    ref, ref_sr = sf.read(args.compare, dtype="float32")
    if ref_sr != sr:
        raise RuntimeError(f"sample rate mismatch: reference {ref_sr}, this run {sr}")
    ref_dur = len(ref) / ref_sr
    print(f"\n   reference       {args.compare}")
    print(f"   duration        {ref_dur:.2f}s vs {duration:.2f}s   (delta {abs(ref_dur - duration):.2f}s)")
    print(f"   rms             {np.sqrt((ref ** 2).mean()):.4f} vs {np.sqrt((audio ** 2).mean()):.4f}")

    # Backends will not be sample-identical, so compare the mel envelope, not the waveform.
    n = min(len(ref), len(audio))
    import librosa

    m_ref = librosa.feature.melspectrogram(y=ref[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=100)
    m_new = librosa.feature.melspectrogram(y=audio[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=100)
    d_ref = librosa.power_to_db(m_ref)
    d_new = librosa.power_to_db(m_new)
    mae = float(np.abs(d_ref - d_new).mean())
    corr = float(np.corrcoef(d_ref.ravel(), d_new.ravel())[0, 1])
    print(f"   mel L1 (dB)     {mae:.2f}")
    print(f"   mel correlation {corr:.4f}")
    print("   -> identical backends give ~0 dB and ~1.0. Different backends drift; what matters is")
    print("      that duration tracks and the chant sounds right, not that the numbers match.")
    return f"RTF {rtf:.2f}, mel L1 {mae:.2f} dB vs reference"


ALL = [s1, s2, s3, s4, s5, s6]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", type=int, default=None, help="subset of stage numbers to run")
    ap.add_argument("--nfe", type=int, default=64, help="flow-matching steps (64 = locked config, 32 = serving)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "out"))
    ap.add_argument("--compare", default="", help="reference wav from another backend, for stage 6")
    args = ap.parse_args()

    wanted = set(args.stages) if args.stages else None
    for fn in ALL:
        if wanted is None or fn._stage[0] in wanted:
            _run(fn, args)

    print("\n" + "═" * 70)
    for num, title, verdict, note in _results:
        print(f"  {verdict:<5} stage {num}  {title}" + (f"  — {note}" if note else ""))
    failed = [r for r in _results if r[2] == "FAIL"]
    print("═" * 70)
    print(f"  {len(_results) - len(failed)}/{len(_results)} stages ok" + (f", {len(failed)} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
