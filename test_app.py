"""Local test for Vāgdhenu — based on demo/app.py, stripped of ZeroGPU/rate-limits.

Synthesizes a Sanskrit verse and plays the result on your Mac.

Usage (from the vagdhenu repo root):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=BigVGAN \
        /opt/anaconda3/envs/py310/bin/python test_app.py

Edit VERSE and METER below to try different inputs.
"""
import os, sys, soundfile as sf

# Some MPS ops (e.g. istft/unfold_backward) are not yet implemented; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── paths ────────────────────────────────────────────────────────────────────
HERE  = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(HERE, "BigVGAN"))

from huggingface_hub import hf_hub_download
from render_core import Renderer, detect_meter_key

# ── choose your verse ─────────────────────────────────────────────────────────
# Paste any Sanskrit shloka in any Indian script. A few examples to try:

VERSE = """\
दैवं तात न पश्यामि नास्ति दैवस्य साधनम् ⁠।
स्वभावतो हि संसिद्धा देवगन्धर्वदानवाः ॥"""

# VERSE = """\
# शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।
# प्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥"""

# VERSE = """\
# वक्रतुण्ड महाकाय सूर्यकोटिसमप्रभ ।
# निर्विघ्नं कुरु मे देव सर्वकार्येषु सर्वदा ॥"""

METER = "auto"   # "auto" to detect, or a name like "anushtubh", "vasantatilaka", etc.
SEED  = 60
OUTPUT = "/tmp/vagdhenu_test.wav"

# ── device auto-select (MPS on Apple Silicon, else CPU) ───────────────────────
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")

# ── load model weights ────────────────────────────────────────────────────────
REPO = "prathoshap/vagdhenu"
MODELS = os.path.join(HERE, "models")
BANK   = os.path.join(SRC, "reference_bank", "bank.json")

voice = os.path.join(MODELS, "voice_steer_ema_2026-06-17.pt")
voc   = os.path.join(MODELS, "voc_bigvgan_EMA_2026-06-11.pth")
vocab = os.path.join(MODELS, "vocab.txt")

# fall back to HuggingFace if local weights are missing
if not os.path.exists(voice):
    print("downloading voice weights …")
    voice = hf_hub_download(REPO, "voice_steer_ema_2026-06-17.pt")
if not os.path.exists(voc):
    print("downloading vocoder weights …")
    voc = hf_hub_download(REPO, "voc_bigvgan_EMA_2026-06-11.pth")
if not os.path.exists(vocab):
    vocab = hf_hub_download(REPO, "vocab.txt")

print("loading model (takes ~30s on first run) …")
renderer = Renderer(voice, voc, BANK, device=device, vocab_file=vocab)
print("model loaded ✓")

# ── meter detection ───────────────────────────────────────────────────────────
if METER == "auto":
    detected = detect_meter_key(VERSE)
    meter = detected if detected else "vasantatilaka"
    print(f"meter  : {meter} (auto-detected)" if detected else f"meter  : {meter} (fallback)")
else:
    meter = METER
    print(f"meter  : {meter}")

# ── synthesize ────────────────────────────────────────────────────────────────
print(f"verse  :\n{VERSE}\n")
print("synthesizing …")
sr, audio = renderer.render_one(VERSE, meter, seed=SEED)

sf.write(OUTPUT, audio, sr)
print(f"output : {OUTPUT}  ({len(audio)/sr:.2f}s)  ✓")

# ── play on Mac ───────────────────────────────────────────────────────────────
print("playing …")
os.system(f"afplay {OUTPUT}")
