---
title: Vāgdhenu — Sanskrit Chant TTS
emoji: 🕉️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: demo/app.py
pinned: false
license: apache-2.0
short_description: Metered Sanskrit/Vedic chant text-to-speech (Devanagari in, audio out)
---

# Vāgdhenu — Sanskrit chant TTS (demo)

A Hugging Face **ZeroGPU** Space for metered Sanskrit/Purāṇic chant synthesis. Enter Devanagari,
pick the meter (chandas), and render. Weights are pulled from
[`prathoshap/vagdhenu`](https://huggingface.co/prathoshap/vagdhenu); the reference bank ships in this repo.

## Deploying

This Space expects the **full `vagdhenu` repo** layout (the app imports `src/render_core.py`,
`src/prep_text.py`, and `src/reference_bank/`). Push the repo to the Space and keep
`app_file: demo/app.py` in this header so the relative imports resolve.

```bash
huggingface-cli upload prathoshap/vagdhenu-demo . --repo-type space
```

Environment variables (optional):

- `VAGDHENU_HF` — weights repo (default `prathoshap/vagdhenu`)
- `VAGDHENU_VOICE` / `VAGDHENU_VOC` — weight filenames

## Notes

- First request is slow: it downloads the weights and loads DiT + BigVGAN onto the GPU. Subsequent
  requests reuse the loaded models.
- The synthesis pipeline mirrors the gold batch renderer (`src/render.py`); see `src/render_core.py`.
