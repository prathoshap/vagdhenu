#!/usr/bin/env bash
set -e
# Python 3.10 + a CUDA 12.1 GPU required for inference.
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/NVIDIA/BigVGAN.git    # provides the 'bigvgan' module (not on PyPI)
python scripts/download_weights.py                        # our weights -> models/ + IndicF5 base (vocab)
echo "✓ setup complete — see Quickstart in README.md"
