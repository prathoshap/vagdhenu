"""Vāgdhenu — Sanskrit/chant TTS demo (Hugging Face ZeroGPU Space).

Loads the released DiT voice + BigVGAN vocoder (from prathoshap/vagdhenu) and the reference
bank shipped in this repo, then synthesizes metered Devanagari chant. The heavy model load +
synthesis happen inside an @spaces.GPU function so ZeroGPU allocates a GPU only on demand.

Local run (with a real GPU + the weights downloaded):
    VAGDHENU_HF=prathoshap/vagdhenu python demo/app.py
"""
import os, sys, json

import gradio as gr
import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
# works both in-repo (demo/app.py -> ../src) and in a flattened Space (app.py + ./src)
SRC = next((p for p in (os.path.join(os.path.dirname(HERE), "src"), os.path.join(HERE, "src"))
            if os.path.exists(os.path.join(p, "render_core.py"))), os.path.join(HERE, "src"))
sys.path.insert(0, SRC)

from huggingface_hub import hf_hub_download

# ── config ───────────────────────────────────────────────────────────────────────────────
WEIGHTS_REPO = os.environ.get("VAGDHENU_HF", "prathoshap/vagdhenu")
VOICE_FILE   = os.environ.get("VAGDHENU_VOICE", "voice_steer_ema_2026-06-17.pt")
VOC_FILE     = os.environ.get("VAGDHENU_VOC", "voc_bigvgan_EMA_2026-06-11.pth")
INDICF5_REPO = "ai4bharat/IndicF5"
BANK_PATH    = os.path.join(SRC, "reference_bank", "bank.json")

# meter dropdown — read the bank at startup (no GPU needed)
_bank = json.load(open(BANK_PATH, encoding="utf-8"))
METERS = [k for k, v in _bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
# friendly default + a couple of common meters first
_PREFERRED = [m for m in ("anuṣṭubh", "upajāti", "śārdūlavikrīḍita", "vasantatilakā") if m in METERS]
METERS = _PREFERRED + [m for m in METERS if m not in _PREFERRED]

EXAMPLE = "वासुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥"

_RENDERER = None


def _ensure_assets():
    """Download the 2 release weights + IndicF5 vocab into the HF cache (CPU-only)."""
    voice = hf_hub_download(WEIGHTS_REPO, VOICE_FILE)
    voc   = hf_hub_download(WEIGHTS_REPO, VOC_FILE)
    try:
        vocab = hf_hub_download(INDICF5_REPO, "checkpoints/vocab.txt")
    except Exception:
        vocab = None  # render_core will fall back to globbing the cache
    return voice, voc, vocab


def _get_renderer():
    global _RENDERER
    if _RENDERER is None:
        from render_core import Renderer
        voice, voc, vocab = _ensure_assets()
        _RENDERER = Renderer(voice, voc, BANK_PATH, device="cuda", vocab_file=vocab)
    return _RENDERER


@spaces.GPU(duration=120)
def synthesize(text, meter, seed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some Devanagari text.")
    r = _get_renderer()
    sr, audio = r.render_one(text, meter, seed=int(seed))
    return (sr, audio)


with gr.Blocks(title="Vāgdhenu — Sanskrit chant TTS") as demo:
    gr.Markdown(
        "# Vāgdhenu — Sanskrit chant TTS\n"
        "Metered Vedic/Purāṇic chant synthesis. Enter **Devanagari** (separate hemistichs with "
        "`।`/`॥` or newlines), pick the **meter** (chandas) that matches the verse, and render.\n\n"
        "Weights: [`prathoshap/vagdhenu`](https://huggingface.co/prathoshap/vagdhenu) · "
        "Apache-2.0 code / CC-BY-4.0 data."
    )
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="Devanagari text", value=EXAMPLE, lines=4)
        with gr.Column(scale=2):
            meter = gr.Dropdown(METERS, value=METERS[0], label="Meter (chandas)")
            seed = gr.Slider(0, 1000, value=60, step=1, label="Seed")
            btn = gr.Button("Synthesize", variant="primary")
    out = gr.Audio(label="Output", type="numpy")
    btn.click(synthesize, inputs=[txt, meter, seed], outputs=out)
    gr.Examples(
        examples=[
            [EXAMPLE, "anuṣṭubh", 60],
            ["शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।\nप्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥", "anuṣṭubh", 60],
        ],
        inputs=[txt, meter, seed],
    )

if __name__ == "__main__":
    demo.launch()
