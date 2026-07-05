#!/usr/bin/env bash
# Mac Apple Silicon (MPS) setup for Vāgdhenu.
# Uses the existing 'py310' conda env from /opt/anaconda3 (Python 3.10, MPS-enabled torch).
# Run once; subsequent renders only need the 'Run' block at the bottom.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Prefer the pre-existing py310 env that already has torch+MPS; fall back to miniconda vagdhenu env.
if [ -x "/opt/anaconda3/envs/py310/bin/python" ]; then
    PIP="/opt/anaconda3/envs/py310/bin/pip"
    PYTHON="/opt/anaconda3/envs/py310/bin/python"
    echo "=== Using existing conda env: /opt/anaconda3/envs/py310 ==="
else
    CONDA_BASE="$(conda info --base)"
    ENV_NAME="vagdhenu"
    if ! conda info --envs | grep -q "${ENV_NAME}"; then
        echo "→ creating conda env '${ENV_NAME}' with Python 3.10 ..."
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
    PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    echo "=== Using conda env: ${ENV_NAME} ==="
    echo "→ installing PyTorch (Mac/MPS) ..."
    "$PIP" install --upgrade torch torchaudio
fi

echo "Python: $PYTHON ($($PYTHON --version 2>&1))"

# Verify MPS
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" -c "
import torch
if torch.backends.mps.is_available():
    print('✓ MPS available — Apple Silicon GPU will be used')
else:
    print('⚠ MPS not available — will run on CPU (slow)')
"

# ── Python dependencies ────────────────────────────────────────────────────
echo "→ installing requirements (IndicF5, transformers, vocos, …) ..."
# Skip torch/torchaudio — already installed above
grep -v "^#" "$REPO/requirements.txt" | "$PIP" install -r /dev/stdin 2>&1 | tail -5

# ── BigVGAN (not a pip package — clone the repo) ──────────────────────────
if [ -d "$REPO/BigVGAN/.git" ]; then
    echo "✓ BigVGAN already cloned"
else
    echo "→ cloning NVIDIA BigVGAN ..."
    git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git "$REPO/BigVGAN"
fi

# ── Download model weights ─────────────────────────────────────────────────
echo "→ downloading Vāgdhenu weights + vocab.txt to models/ ..."
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$REPO/BigVGAN:$PYTHONPATH" \
    "$PYTHON" "$REPO/scripts/download_weights.py"

echo ""
echo "✓ Setup complete!"
echo ""
echo "To render a verse:"
echo "  cd $REPO"
echo "  mkdir -p out"
echo "  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=\"\$PWD/BigVGAN\" \\"
echo "    $PYTHON src/render.py \\"
echo "    --shard examples/sample_shard.json \\"
echo "    --results /tmp/res.json --outdir out"
echo ""
echo "Note: First render downloads BigVGAN base weights from HuggingFace (~1 GB)."
