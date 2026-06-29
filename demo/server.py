"""Vāgdhenu standalone warm server for a dedicated GPU (ece A6000).
Loads the model ONCE at startup and serves it — no ZeroGPU, no per-visitor quota wall.
Guards: one shloka per request + 10 renders/IP/day (src/limits.py). Run in the `indicf5` env."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
import gradio as gr
import limits
from render_core import Renderer, detect_meter_key
from indic_transliteration import sanscript as _S

BANK  = os.path.join(SRC, "reference_bank", "bank.json")
VOCAB = os.path.join(SRC, "reference_bank", "vocab.txt")
VOICE = os.environ.get("VAGDHENU_VOICE", os.path.join(HERE, "weights", "voice_steer.pt"))
VOC   = os.environ.get("VAGDHENU_VOC",   "/home/ece/Prathosh/CHAMPION_2026-06-11/voc_bigvgan_EMA_2026-06-11.pth")
AUTO  = "__auto__"
NFE   = int(os.environ.get("VAGDHENU_NFE", "32"))

print(f"[boot] loading model once  voice={VOICE}  nfe={NFE} …", flush=True)
RENDERER = Renderer(VOICE, VOC, BANK, device="cuda", vocab_file=VOCAB, nfe=NFE)
print("[boot] model warm, ready.", flush=True)

_bank = json.load(open(BANK, encoding="utf-8"))
METERS = [k for k,v in _bank.items() if not k.startswith("_") and isinstance(v,dict) and "wav" in v]
_PREF = [m for m in ("anuṣṭubh","upajāti","śārdūlavikrīḍita","vasantatilakā","mālinī") if m in METERS]
METERS = _PREF + [m for m in METERS if m not in _PREF]
_FALLBACK = "vasantatilakā" if "vasantatilakā" in METERS else METERS[0]
_ALIAS = {}
for _k,_v in _bank.items():
    if _k.startswith("_") or not isinstance(_v,dict) or "wav" not in _v: continue
    _ALIAS[_k.lower()] = _k; _ALIAS[_v["wav"].replace(".wav","").lower()] = _k
METER_CHOICES = [("✨ Auto-detect (recommended)", AUTO)] + [(m,m) for m in METERS]

def _tx(d,sch): return d if sch==_S.DEVANAGARI else _S.transliterate(d,_S.DEVANAGARI,sch)
# (label, devanagari, display script, meter)  — meter=AUTO unless detection needs help
_MAL = ("हठलुठ दल घिष्टोत्कण्ठदष्टोष्ठ विद्युत्\nसटशठ कठिनोरः पीठभित्सुष्ठुनिष्ठाम् ।\n"
        "पठतिनुतव कण्ठाधिष्ठ घोरान्त्रमाला\nदह दह नरसिंहासह्यवीर्याहितं मे ॥")
_KAR = "कराग्रे वसते लक्ष्मीः करमध्ये सरस्वती ।\nकरमूले तु गोविन्दः प्रभाते करदर्शनम् ॥"
_SAMPLES = [
 ("Kṛṣṇa — Vasudevasutaṃ (Devanagari)","वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥",_S.DEVANAGARI,AUTO),
 ("Viṣṇu — Śuklāmbaradharaṃ (Devanagari)","शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।\nप्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥",_S.DEVANAGARI,AUTO),
 ("Narasiṃha — retroflex tongue-twister (mālinī)", _MAL, _S.DEVANAGARI, "mālinī"),
 ("Karāgre vasate — jihvāmūlīya + upadhmānīya", _KAR, _S.DEVANAGARI, AUTO),
 ("Guru — Gururbrahmā (Kannada script)","गुरुर्ब्रह्मा गुरुर्विष्णुः गुरुर्देवो महेश्वरः ।\nगुरुः साक्षात् परं ब्रह्म तस्मै श्रीगुरवे नमः ॥",_S.KANNADA,AUTO),
 ("Sarasvatī — Namastubhyaṃ (Telugu script)","सरस्वति नमस्तुभ्यं वरदे कामरूपिणि ।\nविद्यारम्भं करिष्यामि सिद्धिर्भवतु मे सदा ॥",_S.TELUGU,AUTO),
]
EXAMPLES = [[_tx(d,sc), m, 60] for _,d,sc,m in _SAMPLES]
EXAMPLE_LABELS = [n for n,_,_,_ in _SAMPLES]
EX_DEFAULT = EXAMPLES[0][0]

def _resolve(name):
    k=_ALIAS.get((name or "").lower()); return (k,True) if k else (_FALLBACK,False)

def synthesize(text, meter_choice, seed, request: gr.Request):
    text=(text or "").strip()
    if not text: raise gr.Error("Please paste a verse first 🙏")
    msg = limits.validate_one_shloka(text)
    if msg: raise gr.Error(msg)
    if not limits.check_and_count(limits.client_ip(request)):
        raise gr.Error(f"You've reached today's limit of {limits.DAILY_LIMIT} chants from this network. "
                       f"Please come back tomorrow 🙏")
    if meter_choice==AUTO or not meter_choice:
        used,ok=_resolve(detect_meter_key(text))
        status=(f"🪔 Detected meter: **{used}**" if ok else
                f"🪔 Couldn't pin the meter — chanting with **{used}** (a good general fit).")
    else:
        used,status=meter_choice,f"🪔 Meter: **{meter_choice}**"
    try:
        sr,audio=RENDERER.render_one(text, used, seed=int(seed))
    except Exception as e:
        raise gr.Error(f"Sorry, rendering failed: {e}")
    return (sr,audio), status

INTRO=("# Vāgdhenu — Sanskrit chant\n"
 "Turn a **Sanskrit verse into traditional chant**. Paste a verse and press **Chant it**.\n\n"
 "- Works with **any Indian script** (auto-detected & transliterated).\n"
 "- The **meter is auto-detected** — you don't need to know it.\n\n"
 "*Developed and maintained by Prof. Prathosh, Indian Institute of Science, Bengaluru.*")

with gr.Blocks(title="Vāgdhenu — Sanskrit chant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            txt=gr.Textbox(label="Your verse — one shloka at a time", value=EX_DEFAULT, lines=4,
                           placeholder="Paste a single Sanskrit verse in any Indian script…")
            gr.Markdown("<sub>One shloka per chant · up to 10 chants per day.</sub>")
            with gr.Accordion("⚙️ Advanced (optional)", open=False):
                meter=gr.Dropdown(METER_CHOICES, value=AUTO, label="Meter (chandas)",
                                  info="Leave on Auto-detect unless you know the meter.")
                seed=gr.Slider(0,1000,value=60,step=1,label="Seed")
            btn=gr.Button("🎧 Chant it", variant="primary", size="lg")
        with gr.Column(scale=2):
            out=gr.Audio(label="Chant", type="numpy")
            status=gr.Markdown("")
    btn.click(synthesize, inputs=[txt,meter,seed], outputs=[out,status])
    gr.Markdown("### 📜 Sample shlokas — click one, then press **Chant it**")
    gr.Examples(examples=EXAMPLES, example_labels=EXAMPLE_LABELS, inputs=[txt,meter,seed], label="")

if __name__=="__main__":
    demo.queue(max_size=64, default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=7860, show_api=False)
