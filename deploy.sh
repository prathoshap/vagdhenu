#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Vāgdhenu — one-shot deploy script
#  Clones the repo, injects the FastAPI REST API + Docker setup, builds & launches.
#
#  USAGE:
#    bash deploy.sh                          # deploys to this machine (localhost, Docker)
#    bash deploy.sh --colab                  # deploys on Google Colab (native, no Docker)
#    bash deploy.sh user@host                # deploys over SSH (key-based auth)
#    bash deploy.sh user@host -i key.pem     # deploys over SSH with a specific key
#
#  PREREQS on the target:
#    - NVIDIA GPU + driver ≥ 525 (CUDA 12.1 compatible)
#    - Docker + Docker Compose v2 + nvidia-container-toolkit  (not needed for --colab)
#    - (if remote) SSH key-based auth for the target user
#    - (if --colab) Google Colab with GPU runtime (T4/L4/A100)
#
#  After deploy:
#    Web UI:  http://<host>:8000
#    API:     POST http://<host>:8000/api/chant   (JSON → MP3 bytes)
#             POST http://<host>:8000/api/chant/json  (JSON → JSON+base64)
#             GET  http://<host>:8000/api/meters
#             GET  http://<host>:8000/api/health
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
SSH_TARGET=""
SSH_KEY=""
REMOTE=""
COLAB=false
while [ $# -gt 0 ]; do
  case "$1" in
    --colab) COLAB=true ;;
    -i) shift; [ $# -gt 0 ] && SSH_KEY="-i $1" || { echo "ERROR: -i requires a key path" >&2; exit 2; } ;;
    -i*) SSH_KEY="-i ${1#-i}" ;;
    *)  SSH_TARGET="$1" ;;
  esac
  shift
done
# --colab implies local execution (no SSH to Colab).
if [ "$COLAB" = true ] && [ -n "$SSH_TARGET" ]; then
  echo "WARNING: --colab ignores SSH target (Colab is always local)" >&2
  SSH_TARGET=""
fi
if [ -n "$SSH_TARGET" ]; then
  REMOTE="ssh $SSH_KEY -o StrictHostKeyChecking=no $SSH_TARGET"
fi

# ── paths (this script lives in the repo root alongside api/, Dockerfile, …) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/prathoshap/vagdhenu.git"
# Default deploy destination (used for remote SSH or fresh local clone).
DEST="~/vagdhenu"
# If running locally and the script is already inside a git repo, deploy
# in-place instead of re-cloning to a hardcoded path. This makes
# "git clone … && cd vagdhenu && bash deploy.sh" work from any directory.
if [ -z "$REMOTE" ] && [ -d "$SCRIPT_DIR/.git" ]; then
  DEST="$SCRIPT_DIR"
fi

run() { if [ -n "$REMOTE" ]; then $REMOTE "$1"; else eval "$1"; fi; }
copy() {
  local src="$1" dst="$2"
  if [ -n "$REMOTE" ]; then
    scp $SSH_KEY -o StrictHostKeyChecking=no -r "$src" "$SSH_TARGET:$dst"
  else
    cp -r "$src" "$dst"
  fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Vāgdhenu — Sanskrit Chant TTS · one-shot deploy             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
if [ "$COLAB" = true ]; then
  echo "Target: Google Colab (native, no Docker)"
else
  echo "Target: ${SSH_TARGET:-localhost}"
fi
echo ""

# ── 1. clone / update the repo ────────────────────────────────────────────────
echo "[1/6] Cloning/updating repo…"
run "if [ -d $DEST/.git ]; then cd $DEST && git pull --ff-only || true; else rm -rf $DEST && git clone --depth 1 $REPO_URL $DEST; fi"

# ── 2. copy our API + Docker files into the repo ──────────────────────────────
echo "[2/6] Injecting API + Docker files…"
# The api/, Dockerfile, docker-compose.yml, etc. live alongside this script.
# Copy them into the cloned repo (overlays our additions on the upstream code).
# When deploying locally (no SSH), if the script IS inside the repo dir, skip.
_in_repo=false
if [ -z "$REMOTE" ]; then
  real_script="$(cd "$SCRIPT_DIR" && pwd)"
  real_dest="$(eval cd "$DEST" 2>/dev/null && pwd)" || real_dest=""
  if [ "$real_script" = "$real_dest" ]; then _in_repo=true; fi
fi
if [ "$_in_repo" = true ]; then
  echo "  (script is inside the repo — files already in place)"
else
  for f in api Dockerfile docker-compose.yml .dockerignore pyproject.toml uv.lock deploy.sh .gitattributes; do
    if [ -e "$SCRIPT_DIR/$f" ]; then copy "$SCRIPT_DIR/$f" "$DEST/"; fi
  done
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  COLAB PATH — native install via uv venv (no Docker), cloudflared tunnel
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$COLAB" = true ]; then

# ── 3. check GPU ──────────────────────────────────────────────────────────────
echo "[3/6] Checking GPU…"
run "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || { echo 'ERROR: no GPU detected — enable GPU runtime in Colab'; exit 1; }"

# ── 4. install Python deps via uv venv ────────────────────────────────────────
echo "[4/6] Installing Python deps via uv (fast, isolated venv)…"
run "pip install -q uv"
run "cd $DEST && uv venv --python 3.10 --no-managed-python"
run "cd $DEST && uv pip install --python .venv/bin/python -r requirements.txt"
# requirements.txt can upgrade torch transitively (x-transformers pins torch>=2.5).
# Reinstall the validated CUDA 12.1 stack on top. Include torchvision so
# transformers' image utils don't crash when imported by f5_tts.
run "cd $DEST && uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio"
run "cd $DEST && uv pip install --python .venv/bin/python fastapi uvicorn[standard] pydub python-multipart jinja2"
# Sanity check: the venv must see torch 2.4.1 + CUDA 12.1.
run "cd $DEST && .venv/bin/python -c 'import torch; assert torch.__version__.startswith(\"2.4.1\"), f\"torch {torch.__version__} != 2.4.1\"; print(\"torch\", torch.__version__, \"CUDA\", torch.version.cuda)'"

# ── 5. clone BigVGAN + download weights + launch uvicorn ──────────────────────
echo "[5/6] Setting up BigVGAN + weights + launching server…"
run "[ -d $DEST/BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git $DEST/BigVGAN"
run "cd $DEST && .venv/bin/python scripts/download_weights.py"
# Kill any stale uvicorn on port 8000, then start fresh in background.
run "pkill -f 'uvicorn api.app:app' 2>/dev/null || true"
run "cd $DEST && MPLBACKEND=agg PYTHONPATH=\"$DEST/BigVGAN\" nohup .venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 > /tmp/vagdhenu-uvicorn.log 2>&1 &"

# ── 6. start cloudflared tunnel (non-blocking) ────────────────────────────────
echo "[6/6] Starting tunnel…"
# Install cloudflared if missing (no auth needed for quick tunnels).
run "command -v cloudflared >/dev/null 2>&1 || { wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared; }"
# Start tunnel in background.
run "pkill -f 'cloudflared tunnel' 2>/dev/null || true"
run "nohup cloudflared tunnel --url http://localhost:8000 > /tmp/vagdhenu-tunnel.log 2>&1 &"
# Give cloudflared a few seconds to register the URL, then capture it.
run "sleep 8"
TUNNEL_URL="$(grep -o 'https://[^ ]*\.trycloudflare\.com' /tmp/vagdhenu-tunnel.log 2>/dev/null | head -1 || true)"
if [ -z "$TUNNEL_URL" ]; then
  TUNNEL_URL="http://localhost:8000"
fi

# ── done (Colab) ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Deploy started (Colab)                                    ║"
echo "║                                                              ║"
echo "║  The server + tunnel are running in the background.          ║"
echo "║  Model warm-up takes ~60-120s on first request.              ║"
echo "║                                                              ║"
echo "║  Web UI:   $TUNNEL_URL"
echo "║  Health:   GET  $TUNNEL_URL/api/health"
echo "║  Meters:   GET  $TUNNEL_URL/api/meters"
echo "║  Chant:    POST $TUNNEL_URL/api/chant  (→ MP3)"
echo "║  Chant:    POST $TUNNEL_URL/api/chant/json (→ JSON)"
echo "║                                                              ║"
echo "║  Check:    curl -s http://localhost:8000/api/health          ║"
echo "║  Logs:     cat /tmp/vagdhenu-uvicorn.log                     ║"
echo "║  Tunnel:   cat /tmp/vagdhenu-tunnel.log                      ║"
echo "║  Stop:     pkill -f 'uvicorn api.app' ; pkill -f cloudflared ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCKER PATH — build image, launch container, wait for health
# ═══════════════════════════════════════════════════════════════════════════════

# ── 3. check GPU + Docker on target ───────────────────────────────────────────
echo "[3/6] Checking GPU + Docker…"
run "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || { echo 'ERROR: no GPU detected'; exit 1; }"
run "docker --version 2>&1 || { echo 'ERROR: docker not installed'; exit 1; }"
run "docker compose version 2>&1 || { echo 'ERROR: docker compose v2 not installed'; exit 1; }"
run "docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | tail -1 || { echo 'ERROR: nvidia-container-toolkit not configured'; exit 1; }"

# ── 4. build the Docker image ─────────────────────────────────────────────────
echo "[4/6] Building Docker image (this takes ~5-10 min on first run)…"
run "cd $DEST && docker compose build"

# ── 5. launch ─────────────────────────────────────────────────────────────────
echo "[5/6] Launching container…"
run "cd $DEST && docker compose down 2>/dev/null || true && docker compose up -d"

# ── 6. wait for health ────────────────────────────────────────────────────────
echo "[6/6] Waiting for the model to warm up (60-120s)…"
run "for i in \$(seq 1 60); do
  status=\$(docker inspect --format='{{.State.Health.Status}}' vagdhenu-api 2>/dev/null || echo 'starting')
  if [ \"\$status\" = 'healthy' ]; then echo '✓ healthy'; break; fi
  printf '.'
  sleep 5
done
echo ''
docker compose -f $DEST/docker-compose.yml ps
"

# ── done (Docker) ─────────────────────────────────────────────────────────────
HOST="${SSH_TARGET#*@}"
HOST="${HOST%% *}"
if [ -z "$SSH_TARGET" ]; then HOST="localhost"; fi
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Deploy complete                                           ║"
echo "║                                                              ║"
echo "║  Web UI:   http://$HOST:8000                                 ║"
echo "║  Health:   GET  http://$HOST:8000/api/health                 ║"
echo "║  Meters:   GET  http://$HOST:8000/api/meters                 ║"
echo "║  Chant:    POST http://$HOST:8000/api/chant  (→ MP3)         ║"
echo "║  Chant:    POST http://$HOST:8000/api/chant/json (→ JSON)    ║"
echo "║                                                              ║"
echo "║  Logs:     docker compose -f $DEST/docker-compose.yml        ║"
echo "║            logs -f                                           ║"
echo "║  Stop:     docker compose -f $DEST/docker-compose.yml        ║"
echo "║            down                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
