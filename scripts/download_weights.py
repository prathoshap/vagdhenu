"""Fetch Vāgdhenu weights into models/, plus the IndicF5 base (its checkpoints/vocab.txt is read by render.py from the HF cache)."""
import os
from huggingface_hub import hf_hub_download, snapshot_download
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models"); os.makedirs(MODELS, exist_ok=True)
HF_MODEL = os.environ.get("VAGDHENU_HF", "prathoshap/vagdhenu")   # set once the HF model repo exists
for f in ["voice_steer_ema_2026-06-17.pt", "voice_armA_ema_2026-06-11.pt", "voc_bigvgan_EMA_2026-06-11.pth"]:
    print("↓", f); hf_hub_download(HF_MODEL, f, local_dir=MODELS)
print("↓ ai4bharat/IndicF5 (base DiT + vocab)"); snapshot_download("ai4bharat/IndicF5")
print("✓ weights in", MODELS)
