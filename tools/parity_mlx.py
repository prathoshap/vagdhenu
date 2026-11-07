"""Numerical parity: MLX port vs the torch reference (IndicF5 commit 13f7c4d + NVIDIA BigVGAN).

Runs in the torch (CUDA/CPU) env with mlx installed; BIGVGAN_DIR = a clone of NVIDIA/BigVGAN; needs HF_HOME with nvidia/bigvgan_v2_24khz_100band_256x config.json. Everything loads from the converted safetensors,
so no pickle is opened here.

  usage: parity.py MODEL_DIR UPSTREAM_SRC BIGVGAN_DIR [--cpu] [--steps N]

--cpu runs MLX on the CPU stream, where fp32 matmuls are exact, with tight tolerances: this
checks the port is structurally identical. Without it MLX runs on the GPU, whose fp32 matmul
on Apple silicon is ~1e-3 relative (tensor-core style), so tolerances are looser.
"""
import json, os, sys, time
import numpy as np
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL_DIR, SRC, BVDIR = args[:3]
CPU = "--cpu" in sys.argv
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 6
sys.path.insert(0, SRC); sys.path.insert(0, BVDIR)
import mlx.core as mx
if CPU:
    mx.set_default_device(mx.cpu)
from safetensors.numpy import load_file
from vagdhenu_mlx import audio as A, text as T, dit as D, bigvgan as BV, cfm as C
from vagdhenu_mlx.model import load_dit, load_bigvgan

torch.manual_seed(0); np.random.seed(0)
BANK = os.path.join(SRC, "reference_bank")
bank = json.load(open(os.path.join(BANK, "bank.json"), encoding="utf-8"))
ok = True
TIGHT = CPU
def check(name, a, b, tol_cpu, tol_gpu):
    global ok
    tol = tol_cpu if TIGHT else tol_gpu
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.abs(a - b); scale = np.abs(b).max() + 1e-12
    rel, mrel = err.max() / scale, err.mean() / scale
    good = rel <= tol
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] {name:34s} max rel={rel:.2e} mean rel={mrel:.2e} (tol {tol:.0e}) shape={a.shape}")

# ---------------------------------------------------------------- tokenizer
from f5_tts.model.utils import convert_char_to_pinyin
import prep_text as PT
texts = []
for k, v in bank.items():
    if isinstance(v, dict) and "ref_text" in v:
        texts.append(v["ref_text"] + ".  ")
        for p in v.get("full_verse", v["ref_text"]).replace("॥", "।").split("।"):
            if p.strip(): texts.append(v["ref_text"] + ".  " + PT.model_text(p.strip()))
texts += ["abc def, ghi 12.5% x:y 'q' “z”; ok", "नमः 123 abc", "a b  c"]
bad = [t for t in texts if convert_char_to_pinyin([t])[0] != T.to_char_list(t)]
print(f"[{'ok' if not bad else 'FAIL'}] tokenizer: {len(texts)-len(bad)}/{len(texts)} strings identical")
ok &= not bad
for t in bad[:3]: print("   mismatch:", repr(t), convert_char_to_pinyin([t])[0][:20], T.to_char_list(t)[:20])

# ---------------------------------------------------------------- reference preprocessing + mel
from f5_tts.infer.utils_infer import preprocess_ref_audio_text
from f5_tts.model.modules import MelSpec
import torchaudio
mel_ref = MelSpec()
real_mel = None
for key in ["anuṣṭubh", "śārdūlavikrīḍita", "gadya"]:
    e = bank[key]; wav = os.path.join(BANK, e["wav"])
    ra_path, rt = preprocess_ref_audio_text(wav, e["ref_text"], clip_short=True, show_info=lambda *a: None)
    ta, sr = torchaudio.load(ra_path)
    ours, ours_t = A.preprocess_ref(wav, e["ref_text"])
    assert rt == ours_t, (rt, ours_t)
    check(f"preprocess_ref {key}", ours, ta[0].numpy(), 1e-6, 1e-6)
    m_t = mel_ref(ta)[0].T.numpy()
    check(f"log_mel {key}", A.log_mel(ours), m_t, 1e-4, 1e-4)
    real_mel = real_mel if real_mel is not None else m_t

# ---------------------------------------------------------------- DiT single step
from f5_tts.model import DiT as TDiT
vocab = T.load_vocab(os.path.join(MODEL_DIR, "vocab.txt"))
sd = load_file(os.path.join(MODEL_DIR, "dit.safetensors"))
tdit = TDiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=len(vocab), mel_dim=100)
missing, unexpected = tdit.load_state_dict({k.replace("transformer.", "", 1): torch.from_numpy(v) for k, v in sd.items()}, strict=False)
assert not [m for m in missing if "freqs_cis" not in m and "inv_freq" not in m], missing
tdit.eval()
mdit, _ = load_dit(MODEL_DIR, dtype=mx.float32)

N = 300
x = np.random.randn(1, N, 100).astype(np.float32)
cond = np.random.randn(1, N, 100).astype(np.float32); cond[:, 150:] = 0
ids = np.array([[vocab.get(c, 0) for c in "सन्ततं चिन्तयेत् कण्ठं.  ಸಂತತಂ ಚಿಂತಯೇತ್"]], dtype=np.int64)
tt = 0.37
with torch.no_grad():
    ref_out = tdit(torch.from_numpy(x), torch.from_numpy(cond), torch.from_numpy(ids), torch.tensor(tt), drop_audio_cond=False, drop_text=False).numpy()
    ref_null = tdit(torch.from_numpy(x), torch.from_numpy(cond), torch.from_numpy(ids), torch.tensor(tt), drop_audio_cond=True, drop_text=True).numpy()
tok = np.zeros((1, N), np.int32); tok[0, :ids.shape[1]] = ids[0] + 1
ctx = mdit.context(mx.array(cond), mx.array(tok))
ctx_n = mdit.context(mx.zeros_like(mx.array(cond)), mx.zeros_like(mx.array(tok)))
mods, fin = mdit.time_modulations(mx.array([tt], dtype=mx.float32))
out = mdit.step(mx.array(x), ctx, mods[0], fin[0])
out_n = mdit.step(mx.array(x), ctx_n, mods[0], fin[0])
check("DiT cond step fp32", np.asarray(out), ref_out, 1e-4, 1e-2)
check("DiT null  step fp32", np.asarray(out_n), ref_null, 1e-4, 1e-2)

# batched-vs-single (different lengths, padding masked)
x2 = np.random.randn(1, 220, 100).astype(np.float32); cond2 = cond[:, :220].copy()
tok2 = np.zeros((1, 220), np.int32); tok2[0, :ids.shape[1]] = ids[0] + 1
single2 = mdit.step(mx.array(x2), mdit.context(mx.array(cond2), mx.array(tok2)), mods[0], fin[0])
xb = np.zeros((2, N, 100), np.float32); xb[0] = x[0]; xb[1, :220] = x2[0]
cb = np.zeros((2, N, 100), np.float32); cb[0] = cond[0]; cb[1, :220] = cond2[0]
tb = np.zeros((2, N), np.int32); tb[0] = tok[0]; tb[1] = np.pad(tok2[0], (0, N - 220))
valid = np.zeros((2, N, 1), np.float32); valid[0] = 1; valid[1, :220] = 1
am = mx.where(mx.array(valid[:, None, :, 0]) > 0, 0.0, -1e9)[:, None]
ctxb = mdit.context(mx.array(cb), mx.array(tb), mx.array(valid))
outb = np.asarray(mdit.step(mx.array(xb), ctxb, mods[0], fin[0], mx.array(valid), am))
check("DiT batched==single (item 0)", outb[0], np.asarray(out)[0], 1e-5, 1e-4)
check("DiT batched==single (item 1)", outb[1, :220], np.asarray(single2)[0], 1e-5, 1e-2)

# ---------------------------------------------------------------- BigVGAN
import bigvgan as NB
from env import AttrDict
cfg_path = [os.path.join(r, f) for r, _, fs in os.walk(os.environ["HF_HOME"]) for f in fs
            if f == "config.json" and "bigvgan_v2_24khz_100band_256x" in r][0]
h = AttrDict(json.load(open(cfg_path)))
tv = NB.BigVGAN(h, use_cuda_kernel=False); tv.remove_weight_norm()
vsd = load_file(os.path.join(MODEL_DIR, "bigvgan.safetensors"))
tv.load_state_dict({k: torch.from_numpy(v) for k, v in vsd.items()}, strict=True); tv.eval()
mv = load_bigvgan(MODEL_DIR, dtype=mx.float32)
mel = real_mel[None, 100:160].astype(np.float32)
with torch.no_grad(): ref_w = tv(torch.from_numpy(mel).transpose(1, 2))[0, 0].numpy()
t0 = time.time(); w = np.asarray(mv(mx.array(mel)))[0]; print(f"   bigvgan {mel.shape[1]} frames: {time.time()-t0:.2f}s")
check("BigVGAN fp32 (real mel)", w, ref_w, 1e-4, 5e-2)  # GPU: 6 stages of reduced-precision fp32 convs

# ---------------------------------------------------------------- end-to-end sample (same noise)
from f5_tts.model import CFM
cfm = CFM(transformer=tdit, mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos"), odeint_kwargs=dict(method="euler"), vocab_char_map=vocab).eval()
e = bank["anuṣṭubh"]
ref = C.Reference.load(os.path.join(BANK, e["wav"]), e["ref_text"])
gen = PT.model_text("वैकुण्ठस्याखिला वेदा उद्गीर्यन्तेऽनिशं यतः")
dur = C.Synth.fixed_duration(ref.duration_s + 0.326 * 16)
text_list = convert_char_to_pinyin([ref.text + gen])
steps = STEPS
torch.manual_seed(1); noise = torch.randn(dur, 100).numpy()
t0 = time.time()
with torch.no_grad():
    o, _ = cfm.sample(cond=torch.from_numpy(ref.wave)[None], text=text_list, duration=dur, steps=steps, cfg_strength=3.0, sway_sampling_coef=-1.0, seed=1)
print(f"   torch cpu {steps} steps: {time.time()-t0:.2f}s")
ref_mel = o[0, ref.frames:].numpy()
synth = C.Synth(mdit, vocab, dtype=mx.float32)
t0 = time.time()
m = synth.generate([C.Request(ref, gen, duration=dur, noise=noise)], steps=steps, cfg_strength=3.0)[0]
print(f"   mlx fp32 {steps} steps: {time.time()-t0:.2f}s")
check(f"CFM.sample {steps} steps fp32", m, ref_mel, 1e-3, 3e-2)
if not CPU:
    synth16 = C.Synth(load_dit(MODEL_DIR, dtype=mx.bfloat16)[0], vocab, dtype=mx.bfloat16)
    t0 = time.time(); m16 = synth16.generate([C.Request(ref, gen, duration=dur, noise=noise)], steps=steps, cfg_strength=3.0)[0]
    print(f"   mlx bf16 {steps} steps: {time.time()-t0:.2f}s")
    check(f"CFM.sample {steps} steps bf16", m16, ref_mel, 1e-3, 1e-1)
print("ALL OK" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
