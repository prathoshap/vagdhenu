"""Fetch Vāgdhenu weights + the tokenizer vocab.txt into models/.

Everything needed for inference lives in our public repo (prathoshap/vagdhenu): the DiT voice
checkpoints, the BigVGAN vocoder, and vocab.txt (IndicF5's MIT tokenizer vocab, redistributed here).
No dependency on the gated ai4bharat/IndicF5 repo. BigVGAN's code is cloned at setup (see setup.sh);
its weights are pulled by from_pretrained at runtime.
"""
import os
from huggingface_hub import hf_hub_download
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models"); os.makedirs(MODELS, exist_ok=True)
HF_MODEL = os.environ.get("VAGDHENU_HF", "prathoshap/vagdhenu")
for f in ["voice_steer_ema_2026-06-17.pt", "voice_armA_ema_2026-06-11.pt",
          "voc_bigvgan_EMA_2026-06-11.pth", "vocab.txt"]:
    print("↓", f); hf_hub_download(HF_MODEL, f, local_dir=MODELS)
print("✓ weights + vocab in", MODELS)
