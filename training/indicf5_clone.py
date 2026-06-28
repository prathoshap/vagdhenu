"""IndicF5 zero-shot voice-cloning feasibility test for Sanskrit.
Clone pilot_reciter (chant reference) reading NEW Sanskrit shlokas. MIT model, Devanagari input."""
import os, numpy as np, soundfile as sf, torch
from transformers import AutoModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__)
model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
try: model = model.to(dev)
except Exception as e: print("note: .to(dev) skipped:", e)

ref_wav = "<PROD>/sanskrit-tts/data/wavs/Anuvyakhyana_A2_064.wav"
ref_text = "युक्त्यागमविरोधेन प्राप्तमत्राभिधीयते बालरूढिम् विनैवापि विद्वद्रूढिसमाश्रयात्"
print("ref_audio:", ref_wav)
print("ref_text :", ref_text)

targets = {
 "clone_clean": "वासुदेवं परित्यज्य यो धर्मो नैव विद्यते। तं धर्मं कलिमायाख्यं न वदन्ति मनीषिणः॥",
 "clone_hatha": "हठलुठ दल घिष्टोत्कण्ठदष्टोष्ठ विद्युत् सटशठ कठिनोरः पीठभित्सुष्ठुनिष्ठाम्। पठतिनुतव कण्ठाधिष्ठ घोरान्त्रमाला दह दह नरसिंहासह्यवीर्याहितंमे॥",
 "clone_gadya": "द्वापरे सर्वत्र ज्ञान आकुलीभूते तन्निर्णयाय ब्रह्मरुद्रेन्द्रादिभिरर्थितो भगवान् नारायणो व्यासत्वेनावततार।",
}
os.makedirs("/tmp/indicf5_out", exist_ok=True)
for name, tgt in targets.items():
    try:
        audio = model(tgt, ref_audio_path=ref_wav, ref_text=ref_text)
        audio = np.array(audio, dtype=np.float32)
        if audio.dtype == np.int16 or audio.max() > 1.5: audio = audio.astype(np.float32) / 32768.0
        sf.write(f"/tmp/indicf5_out/{name}.wav", audio, samplerate=24000)
        print(f"wrote {name}  ({len(audio)/24000:.1f}s)")
    except Exception as e:
        print(f"FAILED {name}: {type(e).__name__}: {e}")
print("DONE")
