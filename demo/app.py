"""Vāgdhenu — Sanskrit/chant TTS demo (Hugging Face ZeroGPU Space).

Loads the released DiT voice + BigVGAN vocoder (from prathoshap/vagdhenu) and the reference bank
shipped in this repo, then synthesizes metered chant from a verse in any Indic script. The model
load + synthesis run inside an @spaces.GPU function so ZeroGPU allocates a GPU only on demand.

Designed to be usable by a non-technical user: paste a verse, press one button. The meter is
auto-detected; the only knob (a random seed) is hidden under "Advanced".

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


def _ensure_bigvgan():
    """NVIDIA BigVGAN ships as a repo, not a pip package — clone it once and put it on sys.path so
    `import bigvgan` works (render_core uses the torch path, use_cuda_kernel=False, so no build)."""
    try:
        import bigvgan  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    dst = os.path.join(HERE, "BigVGAN")
    if not os.path.isdir(os.path.join(dst, ".git")):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/NVIDIA/BigVGAN.git", dst], check=True)
    if dst not in sys.path:
        sys.path.insert(0, dst)


_ensure_bigvgan()

# ── config ───────────────────────────────────────────────────────────────────────────────
WEIGHTS_REPO = os.environ.get("VAGDHENU_HF", "prathoshap/vagdhenu")
VOICE_FILE   = os.environ.get("VAGDHENU_VOICE", "voice_steer_ema_2026-06-17.pt")
VOC_FILE     = os.environ.get("VAGDHENU_VOC", "voc_bigvgan_EMA_2026-06-11.pth")
VOCAB_FILE   = os.environ.get("VAGDHENU_VOCAB", "vocab.txt")  # tokenizer vocab, shipped in the weights repo
BANK_PATH    = os.path.join(SRC, "reference_bank", "bank.json")
# vocab.txt is bundled with the repo (IndicF5's MIT tokenizer vocab) so there's NO runtime dependency
# on the gated ai4bharat/IndicF5 repo. Falls back to the weights repo, then None (render_core globs).
BUNDLED_VOCAB = os.path.join(SRC, "reference_bank", VOCAB_FILE)

AUTO = "__auto__"

# meter list + display aliases — read the bank at startup (no GPU needed)
_bank = json.load(open(BANK_PATH, encoding="utf-8"))
METERS = [k for k, v in _bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
_PREFERRED = [m for m in ("anuṣṭubh", "upajāti", "śārdūlavikrīḍita", "vasantatilakā") if m in METERS]
METERS = _PREFERRED + [m for m in METERS if m not in _PREFERRED]
_FALLBACK_DISPLAY = "vasantatilakā" if "vasantatilakā" in METERS else METERS[0]
# detected ascii/diacritic name -> pretty bank key
_ALIAS = {}
for _k, _v in _bank.items():
    if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v:
        continue
    _ALIAS[_k.lower()] = _k
    _ALIAS[_v["wav"].replace(".wav", "").lower()] = _k

# meter dropdown: Auto first (most users never touch it), then the named meters
METER_CHOICES = [("✨ Auto-detect (recommended)", AUTO)] + [(m, m) for m in METERS]

# ── sample shlokas (a ready menu for users who don't have a verse handy) ────────────────────
# Authored in Devanagari and transliterated at load to showcase multiple scripts. All are COMPLETE
# verses so meter auto-detect works. (source-label, devanagari, display script)
from indic_transliteration import sanscript as _S  # noqa: E402


def _tx(deva, scheme):
    return deva if scheme == _S.DEVANAGARI else _S.transliterate(deva, _S.DEVANAGARI, scheme)


_SAMPLES = [
    ("Kṛṣṇa — Vāsudevasutaṃ (Devanagari)",
     "वासुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥", _S.DEVANAGARI),
    ("Viṣṇu — Śuklāmbaradharaṃ (Devanagari)",
     "शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।\nप्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥", _S.DEVANAGARI),
    ("Gaṇeśa — Vakratuṇḍa (Devanagari)",
     "वक्रतुण्ड महाकाय सूर्यकोटिसमप्रभ ।\nनिर्विघ्नं कुरु मे देव सर्वकार्येषु सर्वदा ॥", _S.DEVANAGARI),
    ("Maṅgala-śloka — Maṅgalaṃ bhagavān Viṣṇuḥ (Devanagari)",
     "मङ्गलं भगवान् विष्णुः मङ्गलं गरुडध्वजः ।\nमङ्गलं पुण्डरीकाक्षः मङ्गलायतनो हरिः ॥", _S.DEVANAGARI),
    ("Sarasvatī — Yā kundendu · śārdūlavikrīḍita (Devanagari)",
     "या कुन्देन्दुतुषारहारधवला या शुभ्रवस्त्रावृता\nया वीणावरदण्डमण्डितकरा या श्वेतपद्मासना ।\n"
     "या ब्रह्माच्युतशङ्करप्रभृतिभिर्देवैः सदा वन्दिता\nसा मां पातु सरस्वती भगवती निःशेषजाड्यापहा ॥", _S.DEVANAGARI),
    ("Guru — Gururbrahmā (Kannada script)",
     "गुरुर्ब्रह्मा गुरुर्विष्णुः गुरुर्देवो महेश्वरः ।\nगुरुः साक्षात् परं ब्रह्म तस्मै श्रीगुरवे नमः ॥", _S.KANNADA),
    ("Sarasvatī — Namastubhyaṃ (Telugu script)",
     "सरस्वति नमस्तुभ्यं वरदे कामरूपिणि ।\nविद्यारम्भं करिष्यामि सिद्धिर्भवतु मे सदा ॥", _S.TELUGU),
    ("Morning — Karāgre vasate Lakṣmīḥ (Malayalam script)",
     "कराग्रे वसते लक्ष्मीः करमध्ये सरस्वती ।\nकरमूले तु गोविन्दः प्रभाते करदर्शनम् ॥", _S.MALAYALAM),
]
EXAMPLES = [[_tx(d, sc), AUTO, 60] for _, d, sc in _SAMPLES]
EXAMPLE_LABELS = [name for name, _, _ in _SAMPLES]
EX_DEFAULT = EXAMPLES[0][0]

_RENDERER = None


def _ensure_assets():
    """Fetch the 2 release weights (CPU-only) + resolve the tokenizer vocab. vocab.txt is bundled
    locally (no gated-IndicF5 dependency); if missing, try the weights repo."""
    voice = hf_hub_download(WEIGHTS_REPO, VOICE_FILE)
    voc   = hf_hub_download(WEIGHTS_REPO, VOC_FILE)
    vocab = BUNDLED_VOCAB if os.path.exists(BUNDLED_VOCAB) else None
    if vocab is None:
        try:
            vocab = hf_hub_download(WEIGHTS_REPO, VOCAB_FILE)
        except Exception:
            vocab = None  # last resort: render_core globs the IndicF5 cache (local dev only)
    return voice, voc, vocab


def _get_renderer():
    global _RENDERER
    if _RENDERER is None:
        from render_core import Renderer
        voice, voc, vocab = _ensure_assets()
        _RENDERER = Renderer(voice, voc, BANK_PATH, device="cuda", vocab_file=vocab)
    return _RENDERER


def _resolve_display(name):
    """detected meter name -> (pretty bank key, recognized?)."""
    k = _ALIAS.get((name or "").lower())
    return (k, True) if k else (_FALLBACK_DISPLAY, False)


@spaces.GPU(duration=120)
def synthesize(text, meter_choice, seed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please paste a verse first 🙏")
    from render_core import detect_meter_key
    if meter_choice == AUTO or not meter_choice:
        used, recognized = _resolve_display(detect_meter_key(text))
        status = (f"🪔 Detected meter: **{used}**" if recognized
                  else f"🪔 Couldn't pin the meter — chanting with **{used}** (a good general fit). "
                       f"Tip: paste the *complete* verse, or pick the meter under Advanced.")
    else:
        used, status = meter_choice, f"🪔 Meter: **{meter_choice}**"
    try:
        sr, audio = _get_renderer().render_one(text, used, seed=int(seed))
    except Exception as e:
        raise gr.Error(f"Sorry, rendering failed: {e}")
    return (sr, audio), status


INTRO = """\
# 🕉️ Vāgdhenu — Sanskrit chant
Turn a **Sanskrit verse into traditional chant**. Just paste a verse and press **Chant it**.

- Works with **any Indian script** — Devanagari (Hindi/Sanskrit), Kannada, Telugu, Malayalam,
  Bengali, Gujarati, Gurmukhi, Oriya, Grantha. It's detected automatically.
- The **meter is auto-detected** — you don't need to know it. (You can set it yourself under *Advanced*.)
- Put each line/half-verse on its own line, or separate them with `।` / `॥`.
"""

FOOTER = """\
---
The first chant takes ~30–60s (the model loads onto the GPU); after that it's quick.
Voice & method: [`prathoshap/vagdhenu`](https://huggingface.co/prathoshap/vagdhenu) ·
code [github.com/prathoshap/vagdhenu](https://github.com/prathoshap/vagdhenu) · Apache-2.0.
"""

with gr.Blocks(title="Vāgdhenu — Sanskrit chant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(
                label="Your verse",
                placeholder="Paste a Sanskrit verse in any Indian script…",
                value=EX_DEFAULT, lines=4,
            )
            with gr.Accordion("⚙️ Advanced (optional)", open=False):
                meter = gr.Dropdown(METER_CHOICES, value=AUTO, label="Meter (chandas)",
                                    info="Leave on Auto-detect unless you know the meter.")
                seed = gr.Slider(0, 1000, value=60, step=1, label="Seed",
                                 info="Change for a different take of the same verse.")
            btn = gr.Button("🎧 Chant it", variant="primary", size="lg")
        with gr.Column(scale=2):
            out = gr.Audio(label="Chant", type="numpy", autoplay=False)
            status = gr.Markdown("")
    btn.click(synthesize, inputs=[txt, meter, seed], outputs=[out, status])
    gr.Markdown("### 📜 Sample shlokas — click one to load it, then press **Chant it**")
    gr.Examples(
        examples=EXAMPLES,
        example_labels=EXAMPLE_LABELS,
        inputs=[txt, meter, seed],
        label="",
    )
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch()
