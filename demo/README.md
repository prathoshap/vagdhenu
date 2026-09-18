---
title: Vāgdhenu — Sanskrit Chant TTS
emoji: 🎶
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: demo/app.py
pinned: false
license: apache-2.0
short_description: Metered Sanskrit/Vedic chant text-to-speech (Devanagari in, audio out)
---

# Vāgdhenu — Sanskrit chant TTS (interactive UI)

The interactive web UI: paste a shloka in any Indic script, get chanted audio back. Same underlying
render pipeline as `src/render.py`, wrapped in Gradio.

Two entry points live in this directory, and they support **three deployment modes** between them:

| Mode | Entry point | Where the model runs | GPU cost | Suits |
|---|---|---|---|---|
| Local | `demo/server.py` | Wherever you run it (Mac / dev box / rented GPU VM) | Your own hardware | Personal use, offline pārāyaṇa, development |
| Dedicated GPU server | `demo/server.py` | An always-on GPU box you own or rent | Full-time GPU | Public-facing UI without per-request cold start |
| Hugging Face Space | `demo/app.py` | HF ZeroGPU pool (H100, allocated on demand) | Free tier + per-visitor quota | Public demo with zero recurring cost |

Both scripts serve the same Gradio UI (one shloka in, chanted audio out; meter auto-detect;
seed hidden under Advanced). The choice is about hosting, not features.

---

## Mode 1 — Local

Run everything on your own machine. The whole pipeline stays offline. Backend auto-detected:
CUDA if you have it, MPS on Apple Silicon, CPU otherwise. Model loads once at startup and stays
warm for the life of the process.

### On Apple Silicon (Mac, MPS backend)

```bash
brew install ffmpeg                                        # required — pydub decodes reference clips
git clone git@github.com:shyamsfo/vagdhenu.git && cd vagdhenu
git checkout mac-port
python3.10 -m venv .venv && source .venv/bin/activate      # or `uv venv --seed .venv`
bash scripts/setup.sh                                      # torch (arm64 PyPI wheel), deps, BigVGAN, weights
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"
export VAGDHENU_DAILY_LIMIT=0 VAGDHENU_MAX_AKSHARAS=0      # personal use: disable abuse guards
python demo/server.py
# -> Open http://127.0.0.1:7860 in a browser
```

Model load takes ~60–90 s the first time on MPS. Each render is a couple of seconds.

### On Linux (CPU-only or CUDA)

```bash
sudo apt-get update && sudo apt-get install -y python3.10-venv ffmpeg
git clone git@github.com:shyamsfo/vagdhenu.git && cd vagdhenu
git checkout mac-port
python3.10 -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh                                      # torch (cu121 wheel; harmless on CPU box, just ~3 GB disk)
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"
export VAGDHENU_DAILY_LIMIT=0 VAGDHENU_MAX_AKSHARAS=0
# uncomment the next line only if the box is headless and you'll reach the UI over SSH / LAN:
# export VAGDHENU_HOST=0.0.0.0
python demo/server.py
# -> Open http://127.0.0.1:7860 (or http://<box-ip>:7860 if VAGDHENU_HOST=0.0.0.0)
```

The Linux branch of `setup.sh` unconditionally uses PyTorch's `cu121` wheel index — that pulls
~3 GB of CUDA runtime libraries even if you have no GPU. Torch falls back to CPU at runtime with
no fuss; the extra disk is the only cost.

**CPU performance** on a 16-core x86 box, measured (`nfe 32`, the `demo/server.py` default): about
**~45 s per hemistich**, so a full 32-syllable anuṣṭubh takes ~90 s per request. Not real-time,
but usable for personal pārāyaṇa. Weaker CPUs are proportionally slower.

**Notes (both platforms):**
- Default bind is `127.0.0.1:7860`. To expose on your LAN or reach it over SSH, set
  `VAGDHENU_HOST=0.0.0.0` (or SSH-tunnel: `ssh -L 7860:localhost:7860 your-box`).
- Without `VAGDHENU_DAILY_LIMIT=0`, you'll hit the 10-render/day cap after 10 requests — all your
  local browser calls come from `127.0.0.1`, which counts as one "IP." The check exists to protect
  the public demo from abuse; it makes no sense when the server is just for you.
- `MAX_AKSHARAS=0` lets you paste longer input. Without it, anything over ~100 aksharas or with
  more than one full daṇḍa is rejected as "more than one shloka."

---

## Mode 2 — Dedicated GPU server

Same script as Mode 1, but on a machine with a dedicated NVIDIA GPU that stays warm 24/7. This is
what powers `prathosh.in/vagdhenu/` on Prof. Prathosh's on-prem A6000. It's also the natural fit
for a rented cloud GPU (AWS EC2, Lambda Labs, RunPod, etc.).

### On an on-prem GPU box (or any Linux GPU host you already own)

Setup is identical to the Linux block in Mode 1; the difference is that `bash scripts/setup.sh`'s
`cu121` torch wheel now actually uses the GPU, and you'll want to expose the UI on your LAN:

```bash
# ... same setup as Mode 1 Linux block ...
export VAGDHENU_HOST=0.0.0.0                 # bind on all interfaces
# keep the default limits (VAGDHENU_DAILY_LIMIT=10, VAGDHENU_MAX_AKSHARAS=100) for a public UI
# — or bump, e.g. VAGDHENU_DAILY_LIMIT=50 for a trusted user base
python demo/server.py &
```

Put a reverse proxy (Caddy, nginx, Cloudflare Tunnel) in front for TLS and DNS. `server.py` is
plain HTTP with no auth — don't expose it directly to the public internet.

### On AWS EC2 (or another cloud GPU rental)

The reference setup we've actually verified end-to-end. Every step below has been run against a
fresh instance; the CUDA verify session in this branch's development produced a working `.mp3`
from a `g4dn.4xlarge`.

**1. Pick an instance.**

| Instance | GPU | vCPU / RAM | On-demand price (us-east-1) | Notes |
|---|---|---|---|---|
| `g4dn.xlarge` | T4 (16 GB VRAM) | 4 / 16 GB | ~$0.53/hr | Minimum viable. Fine for single-verse renders. |
| `g4dn.2xlarge` | T4 | 8 / 32 GB | ~$0.75/hr | More headroom for concurrent requests. |
| `g4dn.4xlarge` | T4 | 16 / 64 GB | ~$1.20/hr | What we verified. Comfortable. |
| `g5.xlarge` | A10G (24 GB VRAM) | 4 / 16 GB | ~$1.00/hr | ~2× the T4's inference throughput. |

**~$390/month** if left running 24/7 at `g4dn.xlarge`. For occasional personal use, **stop** the
instance when idle (you pay ~$1/month for the EBS volume only) or terminate entirely and re-launch
when needed. Peak inference memory is ~2.5 GB, so any of the above has room.

**2. Pick the AMI.** Use the **AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 22.04)** —
comes with the NVIDIA driver pre-installed, saves you a driver install + reboot. Look up the
current AMI ID via SSM (the ID rolls forward as AWS publishes updates):

```bash
aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --region us-east-1 --query 'Parameter.Value' --output text
```

**3. Launch + SSH.** Standard EC2 launch. In your security group open **port 22 from your IP only**.
For the UI, either open port 7860 to your IP too, or SSH-tunnel from your laptop (safer).

**4. On the instance:**

```bash
# One-time system prep — ~2 min:
sudo apt-get update && sudo apt-get install -y python3.10-venv ffmpeg

# Clone + checkout (public repo, no auth needed):
git clone https://github.com/shyamsfo/vagdhenu.git && cd vagdhenu
git checkout mac-port

# Venv + setup — ~5-10 min (downloads torch cu121 + weights ~500 MB):
python3.10 -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh

# Configure runtime env:
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"
export VAGDHENU_HOST=0.0.0.0                 # so you can reach it from your laptop
# personal use: set both to 0 to disable the abuse guards; public: leave at defaults
# export VAGDHENU_DAILY_LIMIT=0 VAGDHENU_MAX_AKSHARAS=0

python demo/server.py
# -> "[boot] model warm on cuda (Tesla T4), ready."
# -> "Running on local URL: http://0.0.0.0:7860"
```

**On your laptop**, either SSH-tunnel and open the loopback URL:

```bash
ssh -L 7860:localhost:7860 ubuntu@<instance-public-ip>
# then in a browser: http://127.0.0.1:7860
```

...or hit `http://<instance-public-ip>:7860` directly (only if you opened 7860 in the SG).

**Performance on T4:** anuṣṭubh at `nfe 32` renders in a couple of seconds; `nfe 64` (locked
production config) is ~2× that. Model load at boot: ~30 s.

**Note on the cuDNN trap (auto-handled now, worth knowing about).** The Deep Learning AMI ships
CUDA 12.8 with cuDNN 9.10.2 on `ldconfig`, but torch 2.4.1+cu121 bundles cuDNN 9.24. Without
intervention, torch loads its own 9.24 `libcudnn.so.9` but the loader then finds the system's
9.10.2 `libcudnn_engines_runtime_compiled.so.9` — version mismatch, and the first convolution
fails with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. `src/device.py` prepends the venv's bundled
`nvidia/cudnn/lib` and `nvidia/cublas/lib` to `LD_LIBRARY_PATH` before torch is imported, so this
just works. If you ever see the cuDNN error and this file is missing that prepend logic, that's
your regression.

### Notes (both variants)

- `demo/server.py` loads the model *once at process boot* and holds it warm. Every request reuses
  the loaded model — no per-request cold start. This is why a dedicated GPU is worth the recurring
  cost for a public service.
- Rate limits (`limits.py`) are per-IP and per-day. In-memory, resets on process restart. Fine for
  a demo; not a durable quota system. If you need something stronger, put nginx / Cloudflare
  in front and rate-limit there.
- Add a reverse proxy (Caddy, nginx, Cloudflare Tunnel) for TLS and DNS. `server.py` speaks plain
  HTTP; don't expose it directly to the public internet without TLS termination.

---

## Mode 3 — Hugging Face Space (ZeroGPU)

Zero recurring cost. A pool of H100 GPUs is allocated on demand when a visitor triggers the
`@spaces.GPU`-decorated function; between requests, no GPU is held. The tradeoff is per-request
cold start latency (~10–20 s while ZeroGPU allocates).

```bash
huggingface-cli login    # once
huggingface-cli upload prathoshap/vagdhenu-demo . --repo-type space
```

The `demo/README.md` frontmatter above (`sdk: gradio`, `sdk_version`, `app_file: demo/app.py`) is
what HF Spaces reads to configure the runtime. HF Spaces provides gradio via its SDK, so
`demo/requirements.txt` (not the root `requirements.txt`) is what governs deps for the Space.

**Notes:**
- The Space imports `spaces` (the ZeroGPU decorator package), which only exists in HF's runtime —
  don't try to run `demo/app.py` locally, use `demo/server.py` instead.
- First request downloads the weights from `prathoshap/vagdhenu`; subsequent requests reuse them
  (they're cached in the Space's persistent storage).
- Visitors get a daily GPU-second quota from HF. HF Pro subscribers get more; free-tier visitors
  get less. If a visitor blows through, the widget shows a rate-limit error.
- The Space **also** honors `VAGDHENU_DAILY_LIMIT` and `VAGDHENU_MAX_AKSHARAS` via `limits.py` —
  set them in the Space's environment settings if you want to loosen the defaults.

---

## Environment variables

| Variable | Default | Applies to | Meaning |
|---|---|---|---|
| `VAGDHENU_HOST` | `127.0.0.1` (server.py) | `demo/server.py` | Bind address. Set to `0.0.0.0` to expose on LAN / public. |
| `VAGDHENU_PORT` | `7860` | `demo/server.py` | Port. |
| `VAGDHENU_NFE` | `32` | `demo/server.py` | Flow-matching steps (64 = locked production, 32 = serving). |
| `VAGDHENU_VOICE` | resolved from `models/` or `demo/weights/` | both | Path to the voice-steering `.pt`. |
| `VAGDHENU_VOC` | resolved from `models/` or the on-prem path | both | Path to the BigVGAN `.pth`. |
| `VAGDHENU_HF` | `prathoshap/vagdhenu` | `demo/app.py` | HF repo for weight downloads on the Space. |
| `VAGDHENU_DEVICE` | auto (`cuda`→`mps`→`cpu`) | both | Force a backend (`cpu`, `mps`, or `cuda`). |
| `VAGDHENU_DAILY_LIMIT` | `10` | `src/limits.py` (both) | Per-IP daily render cap. **`0` disables the cap.** |
| `VAGDHENU_MAX_AKSHARAS` | `100` | `src/limits.py` (both) | Reject requests with more aksharas than this. **`0` disables the check.** |

## Notes

- First request is slow on the Space (weight download + GPU allocation). Local warm servers pay
  the load cost once at boot and every request is fast after that.
- The synthesis pipeline mirrors the gold batch renderer (`src/render.py`); see `src/render_core.py`.
- For non-interactive batch conversion (text file → mp3), use `scripts/tts.py` instead of the UI.
