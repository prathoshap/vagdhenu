#!/usr/bin/env bash
set -e
# Python 3.10. Linux needs a CUDA 12.1 GPU; macOS runs on Apple Metal (MPS) or CPU.

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
  if [ "$ARCH" != "arm64" ]; then
    echo "✗ Intel Macs are not supported: PyTorch stopped publishing macOS x86_64 wheels after 2.2.2," >&2
    echo "  and there is no Metal backend for them. Apple Silicon or a CUDA box only." >&2
    exit 1
  fi
  # PyPI ships the macOS arm64 wheels; the cu121 index has no Darwin builds at all.
  command -v ffmpeg >/dev/null || echo "⚠ ffmpeg not found — 'brew install ffmpeg' (pydub decodes the reference clips)"
fi

# Use uv as the installer — pip's resolver spends 45-75 min on nvidia CUDA package backtracking
# on a fresh box because x-transformers transitively pulls torch-einops-utils, which declares
# torch>=2.5, and pip explores many nvidia-*-cu13 wheels before settling. uv resolves in seconds.
pip install --quiet uv
PYBIN="$(command -v python)"
UV_INSTALL="uv pip install --python $PYBIN"
UV_UNINSTALL="uv pip uninstall --python $PYBIN"
if [ "$OS" = "Darwin" ]; then TORCH_IDX=""; else TORCH_IDX="--index-url https://download.pytorch.org/whl/cu121"; fi

$UV_INSTALL torch==2.4.1 torchaudio==2.4.1 $TORCH_IDX
$UV_INSTALL -r requirements.txt
# IndicF5's setup.py asks for torch>=2.0.0, and x-transformers pulls torch-einops-utils (torch>=2.5),
# which lets a resolver upgrade the 2.4.1 just pinned. Re-assert the validated version with --no-deps
# so nothing walks it back, then fail loudly if it slipped.
$UV_INSTALL --reinstall --no-deps torch==2.4.1 torchaudio==2.4.1 $TORCH_IDX
python -c "import torch,sys; v=torch.__version__; sys.exit(0) if v.startswith('2.4.1') else sys.exit('torch is '+v+', expected 2.4.1 - a dependency overrode the pin')"

# On Linux, if a transitive dep briefly upgraded torch to >=2.5 during resolution, cu13 nvidia
# packages will be installed alongside the cu12 ones that torch 2.4.1 needs. Files can land in
# `.venv/lib/.../nvidia/cudnn/lib/` at cu13's cuDNN 9.24 (vs torch 2.4.1's expected 9.1), and the
# first convolution then fails with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. Purge the interlopers
# and force-reinstall the cu12 cuDNN to restore the correct so files.
if [ "$OS" != "Darwin" ]; then
  $UV_UNINSTALL nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
    nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx 2>/dev/null || true
  $UV_INSTALL --reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70
fi

# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"   # add this to your shell rc for persistent use
python scripts/download_weights.py             # our weights -> models/ + IndicF5 base (vocab)

python - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from device import pick_device, describe
print(f"✓ inference backend: {describe(pick_device())}")
PY
echo "✓ setup complete — see Quickstart in README.md"
