# CLAUDE.md

Vāgdhenu: a single-speaker Sanskrit chant (pārāyaṇa) TTS. It is built on IndicF5/F5-TTS (a flow-matching DiT with no duration or pitch head) plus a fine-tuned BigVGAN-v2 vocoder. The full design rationale is in `docs/TECH_REPORT.md`; §14 explains why prosody is steered through the reference clip.

## Commands
```bash
bash scripts/setup.sh            # CUDA 12.1 / Python 3.10: torch, deps, BigVGAN clone, weights -> models/
bash scripts/setup_mlx.sh        # Apple silicon: mlx deps, weights, convert -> models/mlx (safetensors)

python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out   # torch batch
python src/render_mlx.py --text "<devanagari verse>" --out out.wav                           # MLX single
python src/render_mlx.py --shard examples/sample_shard.json --outdir out                     # MLX batch

pytest tests/                    # MLX unit tests (no weights or torch needed; skipped without mlx)
python tools/parity_mlx.py       # torch-vs-MLX parity (needs both envs + weights)
VAGDHENU_BACKEND=mlx VAGDHENU_MLX_MODEL=models/mlx python demo/server.py   # Gradio demo on MLX
```
Shard JSON format: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. `CHAMP_ROOT` overrides the weights dir (default `models/`).

## Architecture (`src/`)
- **Text frontend:** `prep_text.py` routes Devanagari → SLP1 → **Kannada script**, because Devanagari input triggers Hindi schwa-deletion. It also applies visarga sandhi, homorganic anusvāra, ṝ handling and daṇḍa-final rules. `chant_text.py` holds the per-hemistich text helpers and the post-gate (`gate`, `split_padas`, `detect_meter_key`, …), and both backends share it. Meter/gaṇa work lives in the `tts_*.py`, `gana_*.py` and `chandas_labeler.py` modules.
- **Reference bank:** `reference_bank/bank.json` maps each vṛtta to its wav and `ref_text`. **Half-reference rule:** `ref_text` must match the audio *exactly*, and the clip is usually one hemistich. The reference sets voice, swara and pace. An unknown meter falls back to `FALLBACK_METER = "vasantatilaka"`.
- **Torch backend:**
  - `render.py` is the *frozen* batch path. Its helpers are duplicated verbatim in other files, so do not refactor it casually.
  - `render_core.py` provides `Renderer.render_one(text, meter) -> (sr, audio)`. It loads the models once and is used by the demo.
  - `render_production.py` handles one-off single-verse renders.
- **MLX backend:** `vagdhenu_mlx/` (dit, cfm, bigvgan, audio, text, convert, renderer) has no torch at runtime. `from vagdhenu_mlx import Renderer` exposes the same `render_one()` API. The DiT runs in bf16, and BigVGAN's anti-aliased activations use a fused Metal kernel. See `docs/MLX_PORT.md`.
- **Locked parameters:** nfe 64 · cfg 3.0 · speed 0.90 · SR 24000. Keep the two backends in sync.

## Other dirs
- `demo/`: Gradio app (HF ZeroGPU: `app.py`; local: `server.py`).
- `training/`: corpus download, alignment, IndicF5 fine-tunes, BigVGAN vocoder training. These are research scripts, not a package.
- `docs/`: tech report, paper, and the project page (`index.html` with audio).
- `pipeline/`: placeholder for the data-prep and build/QC scripts, which have not been released yet. It contains only a README.

## Conventions
- Weights (`*.pt`, `*.safetensors`, `models/`) and most `*.wav` files are gitignored. The exceptions are `src/reference_bank/*.wav` and `examples/**/*.wav`.
- Code is licensed Apache-2.0. Upstream weights keep their own licenses (IndicF5 MIT, BigVGAN, F5-TTS).
