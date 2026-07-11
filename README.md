# Vāgdhenu — Sanskrit Chant TTS

*"The wish-cow of speech."* A production-grade, single-speaker **Sanskrit chant (pārāyaṇa) text-to-speech** system — it *chants* classical ślokas with metrically-aware durations and tradition-faithful melodic contour, not flat read-aloud.

> **MOS ~4.6** (expert listener). Conjuncts — including retroflex aspirates (ṣṭ, ḍḍh, …) — render 100% correctly, the class earlier architectures could not crack. Used to produce **MBTN** (32 YouTube videos, 17h 34m) and the **Śrīmad Bhāgavatam** (16,017 verses, audio app + 31 karaoke videos).

[ **[Project page + live demo](https://prathosh.in/vagdhenu/)** · [Model weights → HF](https://huggingface.co/prathoshap/vagdhenu) · [Demo → HF Space](https://huggingface.co/spaces/prathoshap/vagdhenu-demo) · Tech report → `docs/TECH_REPORT.md` ]

## Demos (rendered with this system)
- **Mahābhārata Tātparya Nirṇaya (MBTN)** — full chant series: [YouTube playlist](https://www.youtube.com/playlist?list=PLL1s8qiaGy0IP0G_PhlwaGA5EOfzoKrV_)
- **Śrīmad Bhāgavatam** — karaoke-video series: [YouTube playlist](https://www.youtube.com/playlist?list=PLDiYyVdyo2Sc)

Developed and maintained by **Prof. Prathosh, Indian Institute of Science, Bengaluru.**

## How it works
- **Backbone:** IndicF5 / F5-TTS — a flow-matching **DiT** (OT-CFM mel-infilling, ~337M params, *no* native duration or pitch head). Sanskrit is routed through **Kannada script** (Devanagari triggers Hindi schwa-deletion).
- **Vocoder:** NVIDIA **BigVGAN-v2**, fine-tuned on F5 vocos-mel (mandatory — vocos shivers on long vowels).
- **Prosody:** F5's content fidelity is bulletproof but its prosody is *text-driven, not designable*. The working levers are **the reference clip** (voice + swara + pace, via the *half-reference rule*) and a **voice-steering fine-tune**. (See `docs/TECH_REPORT.md` §14 for the full account — this is the central architectural finding.)
- **Text frontend (`src/prep_text.py`)** — the most reusable piece: Deva→SLP1→Kannada routing, internal visarga sandhi (utva/rutva/lopa/satva), homorganic anusvāra, vocalic-ṝ handling, daṇḍa-final rules, meter/gaṇa (L/G) detection.

## Layout
```
src/         text frontend, meter detection, inference, post-gate, reference bank
api/         FastAPI REST API + embedded web UI
demo/        Gradio app (HF ZeroGPU)
docs/        scrubbed technical report + frontend/pipeline references
examples/    sample inputs + rendered outputs
scripts/     env setup + weight download
Dockerfile   CUDA 12.1 + Python 3.10 + uv + ffmpeg + BigVGAN + weights
docker-compose.yml   GPU passthrough, healthcheck, HF cache volume
deploy.sh    one-shot deploy (clone → inject → build → launch → health check)
```

## Install & quickstart
Requires **Python 3.10** and a **CUDA 12.1 GPU**.
```bash
bash scripts/setup.sh    # torch+cu121, deps, BigVGAN, and downloads weights -> models/
# render a Devanagari verse (+ meter) to a chanted wav:
python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```
The batch renderer takes a shard JSON: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. For one-off single-verse renders see `src/render_production.py`. `CHAMP_ROOT` env overrides the weights dir (default `models/`).

## REST API + Docker deployment

A FastAPI REST API wraps the model — POST a Sanskrit verse, get back an MP3.
Includes an embedded web UI, Docker image, and a one-shot deploy script.

### Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/` | Web UI (HTML form + audio player) |
| GET | `/api/health` | `{"status":"ok","model_loaded":true,"meters":18}` |
| GET | `/api/meters` | List of available chandas meters |
| POST | `/api/chant` | JSON in → MP3 bytes (`audio/mpeg`) |
| POST | `/api/chant/json` | JSON in → JSON envelope (base64 MP3 + metadata) |

### One-shot deploy (Docker)

Prerequisites on the target server:
- NVIDIA GPU + driver ≥ 525 (CUDA 12.1 compatible)
- Docker + Docker Compose v2 + nvidia-container-toolkit

```bash
# Deploy on the GPU server itself:
bash deploy.sh

# Or from your local machine to a remote server:
bash deploy.sh ubuntu@host -i key.pem
```

The script clones the repo, injects the API files, builds the Docker image,
launches the container, and waits for the health check. First build takes
~5–10 min (downloads torch cu121 + ML deps + model weights). Subsequent
builds use Docker cache and are near-instant.

### Manual Docker deploy

```bash
docker compose up -d --build     # build + launch
docker compose logs -f           # watch logs
docker compose down              # stop
```

### Environment variables

All have sensible defaults — only override if needed. See `.env.example`.

| Variable | Default | Description |
|---|---|---|
| `VAGDHENU_HF` | `prathoshap/vagdhenu` | Hugging Face repo for weight download |
| `VAGDHENU_VOICE` | `models/voice_steer_ema_2026-06-17.pt` | Path to the DiT voice checkpoint |
| `VAGDHENU_VOC` | `models/voc_bigvgan_EMA_2026-06-11.pth` | Path to the BigVGAN vocoder checkpoint |

## Case studies
- **MBTN** (Mahābhārata Tātparya Nirṇaya) — 32-adhyāya *video* deliverable (Devanagari + Kannada karaoke, tanpura), shipped.
- **Śrīmad Bhāgavatam** — 12 skandhas, ~18k verses, *audio* app + a 31-video 3-script (Devanāgarī · Kannada · IAST) karaoke series. Sanskrit text gratefully acknowledged to **Poornaprajna Samshodhana Mandiram, Bengaluru**.

## Attribution & licenses
- Code: **Apache-2.0** (`LICENSE`).
- Built on **AI4Bharat IndicF5** (MIT), **NVIDIA BigVGAN-v2**, and **F5-TTS** — see their licenses; weights redistributed per those terms.
- Model weights + intended-use/ethics note: see the HF model card.

## Ethics / intended use
Single-speaker synthesis of sacred Sanskrit recitation, for pārāyaṇa/study/accessibility. The voice is the author's own. Please use responsibly; do not impersonate.

## Citation
*(BibTeX added with the arXiv report.)*
