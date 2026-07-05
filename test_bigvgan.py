"""Quick BigVGAN smoke-test — no arguments needed.

Loads the base BigVGAN model, vocoders a demo wav, and plays the result.
Run from the vagdhenu repo root:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=BigVGAN \
      /opt/anaconda3/envs/py310/bin/python test_bigvgan.py
"""
import os, sys, torch, librosa, soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BigVGAN"))
import bigvgan
from meldataset import get_mel_spectrogram

# ── device ──────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")

# ── load model ───────────────────────────────────────────────────────────────
print("loading BigVGAN (downloads ~1 GB on first run) …")
g = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
)
g.eval()
g.remove_weight_norm()
g = g.to(device)
print("model loaded ✓")

# ── input audio ──────────────────────────────────────────────────────────────
INPUT = os.path.join(os.path.dirname(__file__),
                     "BigVGAN", "demo", "examples", "jensen_24k.wav")
OUTPUT = "/tmp/bigvgan_out.wav"

print(f"input  : {INPUT}")
wav, _ = librosa.load(INPUT, sr=g.h.sampling_rate, mono=True)
wav_t  = torch.FloatTensor(wav).unsqueeze(0).to(device)
mel    = get_mel_spectrogram(wav_t, g.h)

# ── run vocoder ───────────────────────────────────────────────────────────────
with torch.no_grad():
    out = g(mel).squeeze().cpu().numpy()

sf.write(OUTPUT, out, g.h.sampling_rate)
print(f"output : {OUTPUT}  ({len(out)/g.h.sampling_rate:.2f}s)  ✓")

# ── play on Mac ───────────────────────────────────────────────────────────────
print("playing …")
os.system(f"afplay {OUTPUT}")
