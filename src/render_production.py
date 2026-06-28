"""PRODUCTION renderer — Prathosh Sanskrit chant TTS.
Gold pipeline: voice_armA_ema (F5/IndicF5 DiT) + nvidia BigVGAN-v2 vocoder (fine-tuned EMA).
BigVGAN-v2 is the production vocoder (vocos left only the long-vowel phase 'shiver'; BigVGAN removes it).
All paths durable (CHAMPION_2026-06-11), no /tmp dependency. See CHAMPION MANIFEST for provenance."""
import os, sys, glob, json, argparse, numpy as np, soundfile as sf, torch
PROD = "<PROD>"
sys.path.insert(0, PROD)
import prep_text as PT, bigvgan
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text
from f5_tts.model import DiT
CHAMP = f"{PROD}/CHAMPION_2026-06-11"

def n_aksharas(s):
    """Count syllable nuclei (Devanagari/Kannada): independent vowels + consonants not followed by virama."""
    n = 0; L = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons  = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i+1] if i+1 < L else ""
            if nxt not in ("\u094D", "\u0CCD"):  # not a virama -> carries a vowel
                n += 1
    return n

def _aksharas(s):
    out=[]; cur=""
    for i,c in enumerate(s):
        o=ord(c); base=(0x0C85<=o<=0x0C94) or (0x0905<=o<=0x0914) or (0x0C95<=o<=0x0CB9) or (0x0915<=o<=0x0939)
        prev=s[i-1] if i>0 else ""
        if base and prev not in ("\u0CCD","\u094D"):
            if cur: out.append(cur)
            cur=c
        else: cur+=c
    if cur: out.append(cur)
    return out

def _rep_depths(aks):
    n=len(aks); mono=1; i=0
    while i<n:
        j=i+1
        while j<n and aks[j]==aks[i]: j+=1
        mono=max(mono,j-i); i=j if j>i+1 else i+1
    di=1; i=0
    while i+1<n:
        if aks[i]!=aks[i+1]:
            cnt=1; j=i+2
            while j+1<n and aks[j]==aks[i] and aks[j+1]==aks[i+1]: cnt+=1; j+=2
            di=max(di,cnt); i=j if cnt>1 else i+1
        else: i+=1
    return mono, di

def _resplit_padawise(pieces, max_syll=24):
    out=[]
    for p in pieces:
        words=p.split(); cur=[]; cs=0
        for w in words:
            ws=n_aksharas(w)
            if cur and cs+ws>max_syll: out.append(" ".join(cur)); cur=[w]; cs=ws
            else: cur.append(w); cs+=ws
        if cur: out.append(" ".join(cur))
    return out

_VMATRA = set("ಾಿೀುೂೃೄೆೇೈೊೋೌ")   # Kannada dependent vowel signs
# Visarga half-echo ONLY after a SHORT vowel (i, u, ṛ via matra; inherent short "a" handled in _danda_fix).
# LONG vowels / diphthongs keep the BARE visarga — it elongates the vowel naturally (guṇaiḥ -> "guṇaa-ii-ḥ").
_VECHO_SHORT = {"ಿ": "ಹಿ", "ು": "ಹು", "ೃ": "ಹೃ"}   # short i, u, ṛ -> hi/hu/hṛ
_VLONG = set("ಾೀೂೄೆೇೈೊೋೌ")                          # ā ī ū ṝ e ai o au -> bare visarga
def _danda_fix(s):
    """At a daṇḍa (segment end): anusvara -> m (always). Visarga -> ha/hi/hu echo ONLY after a SHORT vowel
    (inherent a / i / u / ṛ). After a LONG vowel or diphthong (ā/ī/ū/e/ai/o/au) the visarga is kept BARE —
    it elongates the vowel naturally (guṇaiḥ -> "guṇaa-ii-ḥ"), better than a clipped explicit echo. Short i/u
    over-render bare (ಭೀತಿಃ -> "ti-hi-hi") so they get the echo. GLOBAL, every render (disable: --no_danda_fix)."""
    s = s.rstrip()
    if not s: return s
    if s.endswith("ಃ"):
        core = s[:-1]; pv = core[-1] if core else ""
        if pv in _VECHO_SHORT:      # short i/u/ṛ matra -> echo
            s = core + _VECHO_SHORT[pv]
        elif pv in _VLONG:          # long/diphthong matra -> keep bare visarga
            pass
        else:                        # no matra = inherent short "a" -> ha  (namaḥ -> nama-ha)
            s = core + "ಹ"
    elif s.endswith("ಂ"):
        s = s[:-1] + "ಮ್"
    return s

_AN_KA=set("ಕಖಗಘಙ"); _AN_CA=set("ಚಛಜಝಞ"); _AN_TTA=set("ಟಠಡಢಣ"); _AN_TA=set("ತಥದಧನ")
def _anusvara_m(s):
    """Anusvara ಂ -> homorganic nasal of the FOLLOWING consonant (looks past spaces / word boundary):
    ka-varga->ಙ್, ca->ಞ್, ṭa->ಣ್, ta->ನ್; pa-varga & y/r/l/v/ś/ṣ/s/h -> ಮ್. Segment-final ಂ left for
    _danda_fix. Fixes bare-anusvara conjunct drops (तारतम्यं -> ತಾರತಮ್ಯನ್, not "taratan"). Global."""
    res=[]; n=len(s)
    for i,c in enumerate(s):
        if c=="ಂ":
            j=i+1
            while j<n and s[j]==" ": j+=1
            nxt=s[j] if j<n else ""
            if   not nxt:        res.append("ಂ")
            elif nxt in _AN_KA:  res.append("ಙ್")
            elif nxt in _AN_CA:  res.append("ಞ್")
            elif nxt in _AN_TTA: res.append("ಣ್")
            elif nxt in _AN_TA:  res.append("ನ್")
            else:                res.append("ಮ್")
        else: res.append(c)
    return "".join(res)

_SATVA = {"ಚ": "ಶ್", "ಛ": "ಶ್", "ಟ": "ಷ್", "ಠ": "ಷ್", "ತ": "ಸ್", "ಥ": "ಸ್"}  # visarga + c/ch->ś, ṭ/ṭh->ṣ, t/th->s (satva, joined)
def _satva(s):
    """Internal visarga before c/ch, ṭ/ṭh, t/th -> ś/ṣ/s + the consonant (satva, joined conjunct).
    Visarga before ś/ṣ/s, k/kh (jihvāmūlīya), p/ph (upadhmānīya) stays PLAIN -- the model learned those
    acoustically. Segment-final visarga has no following consonant so it is left untouched (for _danda_fix)."""
    out = []; n = len(s); i = 0
    while i < n:
        c = s[i]
        if c == "ಃ":
            j = i + 1
            while j < n and s[j] == " ": j += 1
            nxt = s[j] if j < n else ""
            if nxt in _SATVA:
                out.append(_SATVA[nxt]); i = j; continue   # drop visarga+space, join sibilant to the next akshara
        out.append(c); i += 1
    return "".join(out)

def _hna_metathesis(s):
    """h + retroflex/dental nasal conjunct -> nasal + h (ಹ್ಣ->ಣ್ಹ, ಹ್ನ->ನ್ಹ). F5 struggles with the
    ह्ण/ह्न onset (e.g. गृह्णन्ति); the metathesis is also a legitimate chant pronunciation (breath
    follows the nasal closure). Vowel matra rides along. (disable with --raw)"""
    return s.replace("ಹ್ಣ", "ಣ್ಹ").replace("ಹ್ನ", "ನ್ಹ")

def _vocalic_l(s):
    """Vocalic ḷ/ḹ -> 'lṛ' rendering: ೢ->್ಲೃ, ೣ->್ಲೄ, ಌ->ಲೃ, ೡ->ಲೄ. Rare (√क्लृप्, अचीकॢपत्); model
    renders कॢ like कृ otherwise. Confirmed v043. (disable with --raw)"""
    return s.replace("ೢ", "್ಲೃ").replace("ೣ", "್ಲೄ").replace("ಌ", "ಲೃ").replace("ೡ", "ಲೄ")

def _warp_bands(mel, delta):
    """Shift F5 mel along the 100-mel-band axis by `delta` bands (>0 = lower/bigger voice). Re-vocoded by BigVGAN -> clean phase."""
    a = np.asarray(mel, dtype=np.float32)
    if 100 not in a.shape: return a
    ax = a.shape.index(100); a2 = np.moveaxis(a, ax, -1); F = a2.shape[-1]; floor = float(a2.min())
    src = np.arange(F, dtype=np.float32) + delta
    i0 = np.floor(src).astype(int); fr = (src - i0).astype(np.float32)
    out = np.take(a2, np.clip(i0,0,F-1), -1)*(1-fr) + np.take(a2, np.clip(i0+1,0,F-1), -1)*fr
    out = np.where((src>=0)&(src<=F-1), out, floor)
    return np.moveaxis(out, -1, ax)

def _self_double(y, sr, semis, voice=0.78, dbl=0.72, depth_ms=12.0, rate_hz=0.3, center_ms=18.0):
    """Chorus 2-voice: phase-aligned pitch-shifted copy + LFO-modulated fractional delay
    (moving comb -> reads as 2 voices, not a static echo). Same swara as the take."""
    import librosa
    from scipy.signal import correlate
    y = np.asarray(y, np.float32)
    b = librosa.effects.pitch_shift(y, sr=sr, n_steps=semis).astype(np.float32) if semis != 0 else y.copy()
    c = len(y)//2; w = min(3*sr, max(1, len(y)//3)); ax = y[c-w//2:c+w//2]; bx = b[c-w//2:c+w//2]
    lag = int(np.argmax(correlate(bx, ax, mode="full")) - (len(ax)-1))   # kill pitch-shift latency
    b = np.roll(b, -lag)
    n = len(b); t = np.arange(n)
    d = (center_ms/1000.0 + (depth_ms/1000.0)*np.sin(2*np.pi*rate_hz*t/sr)) * sr   # time-varying delay (samples)
    idx = t - d; i0 = np.floor(idx).astype(int); fr = (idx - i0).astype(np.float32)
    i0c = np.clip(i0, 0, n-1); i1c = np.clip(i0+1, 0, n-1)
    bc = b[i0c]*(1-fr) + b[i1c]*fr; bc[idx < 0] = 0.0
    m2 = min(len(y), len(bc)); ym = voice*y[:m2] + dbl*bc[:m2]
    m = np.abs(ym).max(); print(f"[double] chorus {semis:+.2f}st depth±{depth_ms:.0f}ms rate{rate_hz}Hz", flush=True)
    return (ym/m*0.97 if m > 0 else ym).astype(np.float32)

def _tanpura(y, sr, level):
    """Synthesized tanpura (Pa-Sa-Sa-Sa, jawari buzz) tuned to the chant's Sa, mixed under the voice."""
    import librosa
    y = np.asarray(y, np.float32)
    f0, _, _ = librosa.pyin(y, fmin=90, fmax=350, sr=sr, frame_length=2048, hop_length=256)
    f0 = f0[np.isfinite(f0)]; sa = 170.0
    if len(f0):
        cents = 1200*np.log2(f0/110.0); hist, edges = np.histogram(cents, bins=240, range=(0,2400))
        pk = (edges[np.argmax(hist)] + edges[np.argmax(hist)+1])/2; sa = 110.0*2**(pk/1200.0)
    rng = np.random.RandomState(7)
    def pluck(f, dur):
        n = int(dur*sr); t = np.arange(n)/sr; out = np.zeros(n, np.float32); k = 1
        while f*k < sr/2 and k <= 40:
            amp = (1.0/k)*(1.0 + 1.8*np.exp(-((k-9)**2)/(2*5.0**2)))
            out += (amp*np.sin(2*np.pi*f*k*(1+0.0003*k)*t + rng.rand()*2*np.pi)).astype(np.float32); k += 1
        env = np.exp(-t/(dur*0.55)).astype(np.float32); at = int(0.006*sr); env[:at] *= np.linspace(0,1,at)
        return out*env
    notes = [0.75*sa, sa, sa, 0.5*sa]; spacing = 1.15; ring = 2.6
    buf = np.zeros(len(y) + int(ring*sr), np.float32); pos = 0; ni = 0
    while pos < len(y):
        g = pluck(notes[ni % 4], ring); e = min(pos+len(g), len(buf)); buf[pos:e] += g[:e-pos]
        pos += int(spacing*sr); ni += 1
    tan = buf[:len(y)]; tan = tan/(np.abs(tan).max()+1e-9)
    fi = int(1.2*sr); fo = int(1.5*sr); tan[:fi] *= np.linspace(0,1,fi); tan[-fo:] *= np.linspace(1,0,fo)
    mix = 0.92*y + level*tan; m = np.abs(mix).max(); print(f"[tanpura] Sa~{sa:.1f}Hz level={level}", flush=True)
    return (mix/m*0.97 if m > 0 else mix).astype(np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--ref_wav"); ap.add_argument("--ref_text_file")
ap.add_argument("--meter", help="auto-load ref wav+text from reference bank by meter name/slug")
ap.add_argument("--bank", default=f"{PROD}/production/reference_bank/bank.json")
ap.add_argument("--padas", required=True, help="JSON list of text segments (split at daṇḍa/half-verse)")
ap.add_argument("--out", required=True)
ap.add_argument("--voice", default=f"{CHAMP}/voice_steer_ema_2026-06-17.pt")  # E79 steering voice; gold fallback: voice_armA_ema_2026-06-11.pt
ap.add_argument("--voc", default=f"{CHAMP}/voc_bigvgan_EMA_2026-06-11.pth")
ap.add_argument("--speed", type=float, default=0.90); ap.add_argument("--nfe", type=int, default=64)
ap.add_argument("--cfg", type=float, default=3.0); ap.add_argument("--gap", type=float, default=0.55)
ap.add_argument("--gap_halant", type=float, default=0.20, help="extra silence (s) added AFTER a chunk that ends in a pure consonant (virama/halant), to equalize PERCEIVED pause vs vowel-final chunks")
ap.add_argument("--no_gap_halant", action="store_true", help="disable halant-aware gap compensation")
ap.add_argument("--seed", type=int, default=50)
ap.add_argument("--also_vocos", action="store_true")
ap.add_argument("--no_clip", action="store_true")
ap.add_argument("--xfade", type=float, default=0.0, help="seconds; >0 = seamless equal-power crossfade between chunks instead of silence gap")
ap.add_argument("--raw", action="store_true", help="feed padas to the model AS-IS (skip model_text frontend); for injecting vocab tokens like ZWNJ")
ap.add_argument("--no_autoprime", action="store_true", help="disable auto repeat-prime selection for verses with deep syllable repeats")
ap.add_argument("--no_danda_fix", action="store_true", help="disable daṇḍa-final visarga->echo / anusvara->m conversion (applied to ALL renders by default)")
ap.add_argument("--no_sandhi", action="store_true", help="disable internal word-boundary visarga sandhi (utva/rutva/lopa + satva); default ON, matches the training texts")
ap.add_argument("--sec_per_syll", type=float, default=-1.0, help="per-chunk dur = ref_len + n_syll*sec_per_syll. -1 = use meter's baked sec_per_syll (calibrated to ref pace); 0 = speed-based; >0 = explicit")
ap.add_argument("--n_syll", help="JSON list of per-pada syllable counts (required with --sec_per_syll)")
ap.add_argument("--double_bands", type=float, default=0.0, help="overlay a 2nd reciter: F5-mel band-shifted by this many mel bands (>0 lower) + BigVGAN re-vocode (clean phase)")
ap.add_argument("--double", type=float, default=0.0, help="self-double: phase-aligned pitch-shifted copy (semitones, e.g. -0.08) for a tight 2-voice sound")
ap.add_argument("--double_mix", type=float, default=0.72, help="level of the doubled 2nd voice (0..1). lower = subtler shadow, higher = co-equal reciter")
ap.add_argument("--tanpura", type=float, default=0.0, help="mix a synthesized tanpura drone tuned to the chant Sa at this level (e.g. 0.08)")
a = ap.parse_args(); SR = 24000
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
vocab = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"))[0]
cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=vocab, device="cuda")
ck = torch.load(a.voice, map_location="cpu", weights_only=True)
ema = {k.replace("ema_model.", ""): v for k, v in ck["ema_model_state_dict"].items() if k not in ("initted", "step")}
cfm.load_state_dict(ema, strict=False); cfm.eval()
real_voc = load_vocoder("vocos")
class Cap:
    def __init__(s, r): s.r = r; s.last = None
    def decode(s, m): s.last = m.detach().cpu().numpy(); return s.r.decode(m)
cap = Cap(real_voc)
# --- PRODUCTION VOCODER: nvidia BigVGAN-v2 (NOT the VITS2 Generator) ---
g = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
bsd = torch.load(a.voc, map_location="cpu"); bsd = bsd.get("model", bsd)
g.load_state_dict(bsd); g.remove_weight_norm(); g = g.cuda().eval()
for p in g.parameters(): p.requires_grad = False
def bvgan(mel):
    m = torch.from_numpy(mel).cuda()
    with torch.no_grad():
        if m.dim()==3 and m.shape[1]!=100 and m.shape[2]==100: m = m.transpose(1,2)
        return g(m).squeeze().cpu().numpy().astype(np.float32)
def gate(au, voice=0.08, sil=0.012, fin=0.015, fout=0.040, lead=0.03, keep=0.06, fade=True, fric=False):
    """Trim F5 padding-silence/edge-artifacts to tight speech bounds (+ click-fades) for rhythmic
    concat & ASS sync. fric=True (clip starts with ś/ṣ/s/h): a leading sibilant is a low-energy
    fricative, often split from the voicing by a stop closure -> the voiced-onset detector eats it
    (e.g. स्तुहि -> 'tuhi'). For these, set the onset at the first window above a low fricative floor
    and SKIP the fade-in so the (faint) fricative attack survives at full level."""
    win = int(0.02*SR); r = [float(np.sqrt((au[i:i+win]**2).mean())) for i in range(0, len(au)-win, win)]; n = len(r)
    if n == 0: return au
    if fric:
        FR = 0.006
        s = next((i for i in range(n-1) if r[i] > FR and r[i+1] > FR), int(np.argmax(r)))
        while s > 0 and r[s-1] > FR: s -= 1
        ve = max((i for i in range(n) if r[i] > 0.035), default=s)
    else:
        vs = next((i for i in range(n-1) if r[i] > voice and r[i+1] > sil), int(np.argmax(r))); s = vs
        while s > 0 and r[s-1] > sil: s -= 1
        ve = max((i for i in range(n) if r[i] > 0.035), default=vs)
    start = max(0, s*win - int(lead*SR))
    end = min(len(au), ve*win + int(keep*SR)); out = au[start:end].copy()
    if fade:
        fi, fo = (0 if fric else int(fin*SR)), int(fout*SR)
        if fi and len(out) > fi: out[:fi] *= np.linspace(0, 1, fi)
        if fo and len(out) > fo: out[-fo:] *= (np.cos(np.linspace(0, np.pi, fo))*0.5 + 0.5)
    return out
if a.meter:
    _bank = json.load(open(a.bank, encoding="utf-8"))
    _bdir = os.path.dirname(a.bank)
    _lut = {}
    for _k, _v in _bank.items():
        if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v: continue
        _lut[_k.lower()] = _v
        _lut[_v["wav"].replace(".wav", "").lower()] = _v
    _q = a.meter.lower().replace(".wav", "")
    if _q not in _lut:
        # NEAREST-METER fallback: pick the bank meter whose ref hemistich is closest in syllable count
        _meters = {k: v for k, v in _bank.items() if isinstance(v, dict) and "wav" in v and v.get("class") != "gadya"}
        try:
            _pads = json.load(open(a.padas)); _vs = n_aksharas(PT.model_text(_pads[0]))
            _k, _e = min(_meters.items(), key=lambda kv: abs(n_aksharas(kv[1].get("ref_text", "")) - _vs))
            print(f"[meter] '{a.meter}' not in bank -> NEAREST meter '{_k}' ({_vs} vs {n_aksharas(_e.get('ref_text',''))} syll/segment)", flush=True)
        except Exception as _ex:
            avail = ", ".join(sorted(set(e["wav"].replace(".wav","") for e in _bank.values() if isinstance(e, dict) and "wav" in e)))
            sys.exit(f"[meter] '{a.meter}' not in bank and nearest-fallback failed ({_ex}). Available: {avail}")
    else:
        _e = _lut[_q]
    if not a.ref_wav: a.ref_wav = os.path.join(_bdir, _e["wav"])
    ref_text = _e["ref_text"]
    if a.sec_per_syll < 0: a.sec_per_syll = float(_e.get("sec_per_syll", 0.26))
    print(f"[meter] {a.meter} -> {_e['wav']} (mode={_e.get('mode')}, {_e.get('dur_s')}s) ref_text={ref_text[:40]}...", flush=True)
elif a.ref_text_file:
    ref_text = open(a.ref_text_file, encoding="utf-8").read().strip()
else:
    sys.exit("Provide --meter, OR both --ref_wav and --ref_text_file")
if not a.ref_wav:
    sys.exit("no --ref_wav resolved (use --meter or --ref_wav)")
ref_audio, ref_t = preprocess_ref_audio_text(a.ref_wav, ref_text, clip_short=not a.no_clip)
def _basetext(p):
    if a.raw: return p
    # internal word-boundary visarga sandhi (utva/rutva/lopa) matches the training texts; segment-final
    # visarga preserved (echo_final=False) for _danda_fix. No-op on already-resolved inputs.
    return PT.model_text_sandhi(p, echo_final=False) if not a.no_sandhi else PT.model_text(p)
_RAWPADAS = json.load(open(a.padas))
FRIC_ONSET = bool(_RAWPADAS) and (PT.align_slp1(_RAWPADAS[0])[:1] in ("S", "z", "s", "h"))   # ś/ṣ/s/h clip onset
PIECES = [_basetext(p) for p in _RAWPADAS]
if not a.no_sandhi and not a.raw:
    PIECES = [_satva(x) for x in PIECES]
if not a.no_danda_fix and not a.raw:
    PIECES = [_danda_fix(_anusvara_m(x)) for x in PIECES]
if not a.raw:
    PIECES = [_hna_metathesis(x) for x in PIECES]
    PIECES = [_vocalic_l(x) for x in PIECES]
print("PIECES:", [repr(x) for x in PIECES], flush=True)
if not a.no_autoprime:
    _mono = max((_rep_depths(_aksharas(x))[0] for x in PIECES), default=1)
    _di   = max((_rep_depths(_aksharas(x))[1] for x in PIECES), default=1)
    if _di >= 3:
        try:
            _bk = json.load(open(a.bank, encoding="utf-8")); _bdir = os.path.dirname(a.bank)
            _primes = _bk.get("repeat_primes", {})
            _pick = next((k for k in ["prime_jaya","prime_chata"] if k in _primes and _primes[k].get("di_max",0)>=_di), None) \
                    or next((k for k,v in _primes.items() if isinstance(v,dict) and v.get("di_max",0)>=_di), None)
            if _pick:
                _pv=_primes[_pick]; a.ref_wav=os.path.join(_bdir,_pv["wav"]); ref_text=_pv["ref_text"]
                ref_audio, ref_t = preprocess_ref_audio_text(a.ref_wav, ref_text, clip_short=not a.no_clip)
                PIECES = _resplit_padawise(PIECES, 24)
                print(f"[autoprime] di x{_di} repeat -> prime '{_pick}' ({_pv['wav']}); pada-wise ({len(PIECES)} chunks)", flush=True)
            else:
                print(f"[autoprime] di x{_di} but no prime with di_max>={_di} in bank", flush=True)
        except Exception as _e:
            print(f"[autoprime] skipped: {_e}", flush=True)
    if _mono >= 3:
        print(f"[autoprime] WARNING: mono x{_mono} repeat has NO in-distribution prime (known gap); syllables may drop", flush=True)
NSYLL = json.load(open(a.n_syll)) if a.n_syll else [n_aksharas(x) for x in PIECES]
if a.sec_per_syll < 0: a.sec_per_syll = 0.26   # fallback when no --meter
REF_LEN_SEC = 0.0
if a.sec_per_syll > 0:
    import torchaudio as _ta
    _ra, _sr = _ta.load(ref_audio)
    REF_LEN_SEC = _ra.shape[-1] / _sr
    print(f"[fixdur] ref_len={REF_LEN_SEC:.2f}s sec_per_syll={a.sec_per_syll} n_syll={NSYLL}", flush=True)
_VIRAMA = "्್"  # devanagari + kannada virama
def _ends_halant(txt):
    t = txt.rstrip(" ।॥|.,;:!?‌‍")
    return len(t) > 0 and t[-1] in _VIRAMA
GAPS = [np.zeros(int(a.gap*SR) + (int(a.gap_halant*SR) if (not a.no_gap_halant and _ends_halant(_p)) else 0), dtype=np.float32) for _p in PIECES]
if not a.no_gap_halant:
    _hl = [i for i, _p in enumerate(PIECES) if _ends_halant(_p)]
    if _hl: print(f"[gap_halant] +{a.gap_halant:.2f}s after halant-final chunks {_hl}", flush=True)
gap = np.zeros(int(a.gap*SR), dtype=np.float32); bseg = []; vseg = []
for i, p in enumerate(PIECES):
    au = None
    for att in range(4):
        torch.manual_seed(a.seed + att)   # all chunks start from the vetted base seed; att only on retry (was seed+i*5 -> chunk1 landed on bad seed 55)
        _fixd = (REF_LEN_SEC + NSYLL[i]*a.sec_per_syll) if (a.sec_per_syll > 0 and NSYLL) else None
        w, sr, _ = infer_process(ref_audio, ref_t, p, cfm, cap, mel_spec_type="vocos", speed=a.speed, nfe_step=a.nfe, cfg_strength=a.cfg, device="cuda", fix_duration=_fixd)
        w = np.array(w, dtype=np.float32)
        if np.abs(w).max() > 1.5: w = w/32768.0
        if float(np.sqrt((w**2).mean())) > 0.04: au = w; break
    if au is None: au = w
    y = bvgan(cap.last); mx = np.abs(y).max(); y = y/mx*0.97 if mx > 1 else y
    if a.double_bands != 0:
        y2 = bvgan(_warp_bands(cap.last, a.double_bands)); mx2 = np.abs(y2).max(); y2 = y2/mx2*0.97 if mx2 > 1 else y2
        m2 = min(len(y), len(y2)); ym = 0.72*y[:m2] + 0.72*y2[:m2]; mm = np.abs(ym).max(); y = ym/mm*0.97 if mm > 0 else ym
    bseg.append(y)
    if a.also_vocos: vseg.append(au)
    print(f"pada{i}: {len(y)/SR:.1f}s", flush=True)
def _stitch(segs):
    if a.xfade > 0 and len(segs) > 1:
        X = int(a.xfade*SR)
        pieces = [gate(s, fade=False, fric=(FRIC_ONSET and i == 0)) for i, s in enumerate(segs)]
        out = pieces[0].astype(np.float32).copy()
        for nxt in pieces[1:]:
            nxt = nxt.astype(np.float32); o = min(X, len(out), len(nxt))
            if o <= 0: out = np.concatenate([out, nxt]); continue
            wa = np.sqrt(np.linspace(1, 0, o)); wb = np.sqrt(np.linspace(0, 1, o))
            out = np.concatenate([out[:-o], out[-o:]*wa + nxt[:o]*wb, nxt[o:]])
        fi, fo = int(0.015*SR), int(0.040*SR)
        if len(out) > fi: out[:fi] *= np.linspace(0, 1, fi)
        if len(out) > fo: out[-fo:] *= (np.cos(np.linspace(0, np.pi, fo))*0.5 + 0.5)
        return out
    b = []
    for i, s in enumerate(segs): b += [gate(s, fric=(FRIC_ONSET and i == 0)), GAPS[i] if i < len(GAPS) else gap]
    return np.concatenate(b[:-1])
_final = _stitch(bseg)
if a.double:  _final = _self_double(_final, SR, a.double, dbl=a.double_mix)
if a.tanpura: _final = _tanpura(_final, SR, a.tanpura)
sf.write(a.out, _final, SR); print("RENDERED", a.out, flush=True)
if a.also_vocos:
    vp = a.out.rsplit(".",1)[0] + "_vocos.wav"; sf.write(vp, _stitch(vseg), SR); print("also", vp, flush=True)
