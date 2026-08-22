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
- **Text frontend (`src/prep_text.py`)** — the most reusable piece: Deva→SLP1→Kannada routing, internal visarga sandhi (utva/rutva/lopa/satva), homorganic anusvāra, vocalic-ṝ handling, daṇḍa-final rules, meter/gaṇa (L/G) detection. Vedic pitch marks (U+0952 anudātta, U+0951 RV svarita stroke, Vedic Extensions) are stripped before Kannada routing and returned as a parallel `accent_array` aligned 1:1 with gaṇa syllables (remapped through sandhi). **Audio remains classical śāstra chant** (`accents_drive_audio=False`); use `render_plan.plan_render` / `scripts/dry_run_render.py` for a GPU-free manifest.

## Layout
```
src/         text frontend, meter detection, render plan (noop), inference, reference bank
pipeline/    data-prep (cut→pair→train) + build/assemble/QC
demo/        Gradio app (HF ZeroGPU) — includes Plan only (dry-run)
docs/        scrubbed technical report + frontend/pipeline references
examples/    sample inputs (+ vedic_four_vedas_shard.json)
scripts/     env setup, weight download, dry_run_render, eval helpers
tests/       frontend unit tests (no GPU)
```

## Install & quickstart
Requires **Python 3.10** and a **CUDA 12.1 GPU** for audio render.
```bash
bash scripts/setup.sh    # torch+cu121, deps, BigVGAN, and downloads weights -> models/
# render a Devanagari verse (+ meter) to a chanted wav:
python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```
The batch renderer takes a shard JSON: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. For one-off single-verse renders see `src/render_production.py`. `CHAMP_ROOT` env overrides the weights dir (default `models/`).

### Text API note (breaking)
`model_text` / `model_text_sandhi` / `model_text_norm` return **`(kannada_text, accent_array)`**. Unpack or index `[0]` for the phoneme string used by IndicF5. `accent_array[i]` aligns with syllable `i` in `chandas_labeler.scan` / `syllabify_slp1`.

### Dry-run / noop (no GPU)
```bash
pip install indic-transliteration   # minimal frontend dep
python scripts/dry_run_render.py --veda-samples
python scripts/dry_run_render.py --text "अ॒ग्निमी॑ळे पुरोहितं ..." -o plan.json
python -m pytest tests/test_svara_prep.py -q
```

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
