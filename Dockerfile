# Vāgdhenu — Sanskrit Chant TTS REST API
# CUDA 12.1 + Python 3.10 (matches the validated torch 2.4.1 + cu121 stack).
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

# ── system deps: python 3.10, ffmpeg (mp3 encode), git (BigVGAN clone), curl ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# ── uv (fast installer) ────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

# ── 1. API deps via uv (fastapi, uvicorn, pydub, …) ───────────────────────────
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# ── 2. ML deps via uv pip (torch cu121 + the requirements.txt stack) ──────────
#    Installed INTO the uv venv so `uv run` picks them up. We use uv pip (not
#    uv add) because x-transformers==2.19.7 pins torch>=2.5 transitively —
#    pip's looser resolver matches the original setup.sh exactly.
#    Order matters: requirements.txt first (pulls f5-tts + deps, may upgrade
#    torch), then torch cu121 reinstalled on top to overwrite with the CUDA
#    build that matches the host driver (12.2).
COPY requirements.txt ./
RUN uv pip install -r requirements.txt
RUN uv pip install torch==2.4.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 --reinstall-package torch --reinstall-package torchaudio

# ── app source ────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# ── BigVGAN (NVIDIA repo, not a pip package) ──────────────────────────────────
RUN git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git /app/BigVGAN
ENV PYTHONPATH="/app/BigVGAN:${PYTHONPATH}"

# ── weights: download at build time (HF) so the image is self-contained ───────
RUN uv run python scripts/download_weights.py || echo "[warn] weight download failed — mount models/ at runtime"

EXPOSE 8000

# Warm the model at boot (not lazy). Override with VAGDHENU_LAZY=1 to defer.
ENV VAGDHENU_LAZY=0

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
