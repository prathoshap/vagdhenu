#!/usr/bin/env bash
set -e
# Python 3.10 required. Device is auto-selected by src/_device.py:
#   MPS on Apple Silicon, CUDA if available, else CPU.
#   Override with: export VAGDHENU_DEVICE=cpu   (or mps|cuda)
#
# Pick the torch build that matches your platform. The default PyPI wheel ships
# MPS (Apple Silicon) + CUDA + CPU in one binary, so most users need NO --index-url.

if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    # Apple Silicon (M1-M4): default PyPI wheel has MPS. 2.6.0+ has mature M4 support.
    pip install torch==2.6.0 torchaudio==2.6.0
else
    # Linux/x86 with a CUDA 12.1 GPU (original upstream target).
    pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements.txt

# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
# src/ is needed on the path too (prep_text, _device, tts_* helpers). Recommended in shell rc:
export PYTHONPATH="$PWD/BigVGAN:$PWD/src:$PYTHONPATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # any op MPS lacks falls back to CPU instead of crashing
python scripts/download_weights.py                        # our weights -> models/ + IndicF5 base (vocab)
echo "✓ setup complete — see Quickstart in README.md"
