#!/usr/bin/env bash
# Launch the local Vāgdhenu Gradio server for personal use.
#   ./start_server.sh mac      # Apple Silicon (MPS), localhost
#   ./start_server.sh linux    # x86 Linux CPU dev box, localhost
#   ./start_server.sh gpu      # NVIDIA CUDA (on-prem A6000 / AWS g4dn), LAN-visible
#
# Assumes the venv is already active. Rate-limits are disabled (personal use). To expose a
# public UI, unset VAGDHENU_DAILY_LIMIT / VAGDHENU_MAX_AKSHARAS before invoking — see
# demo/README.md for the multi-mode guide.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: ./start_server.sh {mac|linux|gpu}
  mac     Apple Silicon (MPS), binds 127.0.0.1:7860
  linux   x86 Linux CPU, binds 127.0.0.1:7860
  gpu     NVIDIA CUDA, binds 0.0.0.0:7860 (reachable from your laptop)
env overrides: VAGDHENU_HOST, VAGDHENU_PORT, VAGDHENU_DAILY_LIMIT, VAGDHENU_MAX_AKSHARAS
EOF
  exit 2
}

[[ $# -eq 1 ]] || usage
MODE="$1"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: no venv active. run 'source .venv/bin/activate' first." >&2
  exit 1
fi

if [[ ! -d "$REPO/BigVGAN" ]]; then
  echo "error: BigVGAN/ missing — run 'bash scripts/setup.sh' first." >&2
  exit 1
fi

export PYTHONPATH="$REPO/BigVGAN:${PYTHONPATH:-}"
# personal use: disable both abuse guards (caller can override before invoking)
export VAGDHENU_DAILY_LIMIT="${VAGDHENU_DAILY_LIMIT:-0}"
export VAGDHENU_MAX_AKSHARAS="${VAGDHENU_MAX_AKSHARAS:-0}"
export VAGDHENU_PORT="${VAGDHENU_PORT:-7860}"

case "$MODE" in
  mac)
    export VAGDHENU_DEVICE="${VAGDHENU_DEVICE:-mps}"
    export VAGDHENU_HOST="${VAGDHENU_HOST:-127.0.0.1}"
    ;;
  linux)
    export VAGDHENU_DEVICE="${VAGDHENU_DEVICE:-cpu}"
    export VAGDHENU_HOST="${VAGDHENU_HOST:-127.0.0.1}"
    ;;
  gpu)
    export VAGDHENU_DEVICE="${VAGDHENU_DEVICE:-cuda}"
    export VAGDHENU_HOST="${VAGDHENU_HOST:-0.0.0.0}"
    ;;
  *)
    usage
    ;;
esac

_url_host="$VAGDHENU_HOST"
[[ "$_url_host" == "0.0.0.0" ]] && _url_host="<box-ip>"
echo "-> mode=$MODE device=$VAGDHENU_DEVICE"
echo "-> open http://${_url_host}:${VAGDHENU_PORT} once boot completes"
exec python demo/server.py
