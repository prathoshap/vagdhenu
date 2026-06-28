"""Fix: monkeypatch GRN to weight/bias (matching the IndicF5 checkpoint) BEFORE loading,
so the trained text-encoder GRN loads instead of zero-init identity. Then re-test Sanskrit."""
import os, numpy as np, soundfile as sf, torch, torch.nn as nn
import f5_tts.model.modules as mod

class GRN_wb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, dim))   # was self.gamma
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))     # was self.beta
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (x * Nx) + self.bias + x
mod.GRN = GRN_wb   # patch BEFORE model instantiation

from transformers import AutoModel
from indic_transliteration import sanscript
m = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to("cuda")

# verify the text-encoder GRN actually loaded (non-zero => from checkpoint)
for n, p in m.named_parameters():
    if "text_blocks.0.grn.weight" in n:
        print(f"text GRN[0].weight |sum|={float(p.abs().sum()):.4f}  (LOADED if >0, was 0=identity)"); break

ref_wav = "<PROD>/sanskrit-tts/data/wavs/Anuvyakhyana_A2_064.wav"
ref_dev = "युक्त्यागमविरोधेन प्राप्तमत्राभिधीयते बालरूढिम् विनैवापि विद्वद्रूढिसमाश्रयात्"
def kn(d): return sanscript.transliterate(d, sanscript.DEVANAGARI, sanscript.KANNADA)
items = {
 "hindi":          ("नमस्ते, संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए।", ref_dev),
 "skt_devanagari": ("वासुदेवं परित्यज्य यो धर्मो नैव विद्यते", ref_dev),
 "skt_kannada":    (kn("वासुदेवं परित्यज्य यो धर्मो नैव विद्यते"), kn(ref_dev)),
}
os.makedirs("/tmp/indicf5_grnfix", exist_ok=True)
for name, (tgt, rt) in items.items():
    try:
        a = np.array(m(tgt, ref_audio_path=ref_wav, ref_text=rt), dtype=np.float32)
        if np.abs(a).max() > 1.5: a = a / 32768.0
        sf.write(f"/tmp/indicf5_grnfix/{name}.wav", a, 24000)
        print(f"wrote {name}  {len(a)/24000:.1f}s")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
print("DONE")
