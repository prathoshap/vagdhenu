#!/usr/bin/env bash
# Apple silicon setup: MLX runtime + weights converted to safetensors. No CUDA, no torch at runtime.
#   bash scripts/setup_mlx.sh            # -> models/ (torch .pt/.pth) and models/mlx/ (dit + bigvgan safetensors, vocab)
#   python -c "import sys; sys.path.insert(0,'src'); from vagdhenu_mlx import Renderer; ..."
set -e
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements-mlx.txt huggingface_hub
python scripts/download_weights.py                       # weights -> models/
if [ ! -f models/mlx/dit.safetensors ]; then
  echo "converting checkpoints -> models/mlx (one-off; uses a temporary torch install)"
  python -m venv .convert-venv
  .convert-venv/bin/pip install -q torch safetensors numpy
  .convert-venv/bin/python -m pip install -q -e . 2>/dev/null || true
  PYTHONPATH=src .convert-venv/bin/python -m vagdhenu_mlx.convert \
      --voice models/voice_steer_ema_2026-06-17.pt --vocoder models/voc_bigvgan_EMA_2026-06-11.pth \
      --vocab models/vocab.txt --out models/mlx
  rm -rf .convert-venv
fi
echo "✓ MLX setup complete: models/mlx — try: python src/render_mlx.py --text 'वसुदेवसुतं देवं कंसचाणूरमर्दनम् । देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥' --out out.wav"
