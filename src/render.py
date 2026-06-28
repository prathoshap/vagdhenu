"""PERSISTENT BATCH renderer — load DiT+BigVGAN ONCE, loop over a shard of per-hemistich clips.
Faithful port of render_production.py's gold per-piece pipeline (helpers copied verbatim); the only
change is structural: model loaded once, render_clip() called per clip with a per-clip seed.
Writes DRY hemistich wavs (tanpura/assembly happen later in postaudio).
shard JSON: [{"id","meter","padas":[deva,...],"seed":60,"no_sandhi":true,"out":"/abs/clip.wav"}]"""
import os, sys, glob, json, argparse, numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__))     # src/  (prep_text.py sits beside this file)
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import prep_text as PT, bigvgan
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text
from f5_tts.model import DiT
CHAMP = os.environ.get("CHAMP_ROOT", os.path.join(REPO, "models"))   # weights land here (scripts/download_weights.py)
SR = 24000
FALLBACK_METER = "vasantatilaka"   # unknown/unmatched vṛtta -> this reference instead of erroring (see get_ref)

# ── helpers copied VERBATIM from render_production.py ──────────────────────────────────
def n_aksharas(s):
    n = 0; L = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons  = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i+1] if i+1 < L else ""
            if nxt not in ("्", "್"):
                n += 1
    return n

def _aksharas(s):
    out=[]; cur=""
    for i,c in enumerate(s):
        o=ord(c); base=(0x0C85<=o<=0x0C94) or (0x0905<=o<=0x0914) or (0x0C95<=o<=0x0CB9) or (0x0915<=o<=0x0939)
        prev=s[i-1] if i>0 else ""
        if base and prev not in ("್","्"):
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

_VMATRA = set("ಾಿೀುೂೃೄೆೇೈೊೋೌ")
_VECHO_SHORT = {"ಿ": "ಹಿ", "ು": "ಹು", "ೃ": "ಹೃ"}
_VLONG = set("ಾೀೂೄೆೇೈೊೋೌ")
def _danda_fix(s):
    s = s.rstrip()
    if not s: return s
    if s.endswith("ಃ"):
        core = s[:-1]; pv = core[-1] if core else ""
        if pv in _VECHO_SHORT:      s = core + _VECHO_SHORT[pv]
        elif pv in _VLONG:          pass
        else:                        s = core + "ಹ"
    elif s.endswith("ಂ"):
        s = s[:-1] + "ಮ್"
    return s

_AN_KA=set("ಕಖಗಘಙ"); _AN_CA=set("ಚಛಜಝಞ"); _AN_TTA=set("ಟಠಡಢಣ"); _AN_TA=set("ತಥದಧನ")
def _anusvara_m(s):
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

_SATVA = {"ಚ": "ಶ್", "ಛ": "ಶ್", "ಟ": "ಷ್", "ಠ": "ಷ್", "ತ": "ಸ್", "ಥ": "ಸ್"}
def _satva(s):
    out = []; n = len(s); i = 0
    while i < n:
        c = s[i]
        if c == "ಃ":
            j = i + 1
            while j < n and s[j] == " ": j += 1
            nxt = s[j] if j < n else ""
            if nxt in _SATVA:
                out.append(_SATVA[nxt]); i = j; continue
        out.append(c); i += 1
    return "".join(out)

def _hna_metathesis(s):
    """h + retroflex/dental nasal conjunct -> nasal + h (ಹ್ಣ->ಣ್ಹ, ಹ್ನ->ನ್ಹ). F5 struggles with the
    ह्ण/ह्न onset (e.g. गृह्णन्ति); the metathesis is also a legitimate chant pronunciation (the breath
    follows the nasal closure). The vowel matra rides along (ಹ್ಣಿ->ಣ್ಹಿ). Confirmed by ear on v090 गृह्णन्ति."""
    return s.replace("ಹ್ಣ", "ಣ್ಹ").replace("ಹ್ನ", "ನ್ಹ")

def _vocalic_l(s):
    """Vocalic ḷ/ḹ (ऌ, कॢ) -> traditional 'lṛ' rendering: matra ೢ->್ಲೃ, ೣ->್ಲೄ; independent ಌ->ಲೃ, ೡ->ಲೄ.
    Vanishingly rare (essentially only √क्लृप्, e.g. अचीकॢपत्); the model never learned it and renders कॢ
    like कृ ('kru'). Confirmed by ear on v043 अचीकॢपत् -> अचीक्लृपत्."""
    return s.replace("ೢ", "್ಲೃ").replace("ೣ", "್ಲೄ").replace("ಌ", "ಲೃ").replace("ೡ", "ಲೄ")

def gate(au, voice=0.08, sil=0.012, fin=0.015, fout=0.040, lead=0.03, keep=0.06, fade=True, fric=False, halant=False):
    """Trim F5 padding-silence/edge-artifacts to tight speech bounds (+ click-fades). fric=True (clip
    starts with ś/ṣ/s/h): keep the low-energy leading fricature (low onset floor, no fade-in). halant=True
    (clip ENDS in a pure consonant त्/क्/प्): the final unvoiced stop is a weak burst after a closure, so
    the standard 0.035 offset detector cuts it -> low trailing floor + more keep + short fade-out."""
    win = int(0.02*SR); r = [float(np.sqrt((au[i:i+win]**2).mean())) for i in range(0, len(au)-win, win)]; n = len(r)
    if n == 0: return au
    if fric:
        FR = 0.006   # fricative floor: catch the quiet ś/ṣ/s onset, not pure silence/noise
        s = next((i for i in range(n-1) if r[i] > FR and r[i+1] > FR), int(np.argmax(r)))
        while s > 0 and r[s-1] > FR: s -= 1
        _vdef = s
    else:
        vs = next((i for i in range(n-1) if r[i] > voice and r[i+1] > sil), int(np.argmax(r))); s = vs
        while s > 0 and r[s-1] > sil: s -= 1
        _vdef = vs
    ve_thr = 0.012 if halant else 0.035            # halant: catch the weak final stop burst after the closure
    ve = max((i for i in range(n) if r[i] > ve_thr), default=_vdef)
    keep_s = 0.12 if halant else keep
    start = max(0, s*win - int(lead*SR))
    end = min(len(au), ve*win + int(keep_s*SR)); out = au[start:end].copy()
    if fade:
        fi = (0 if fric else int(fin*SR)); fo = int((0.018 if halant else fout)*SR)   # no fade-in over fricative; short fade-out on stop
        if fi and len(out) > fi: out[:fi] *= np.linspace(0, 1, fi)
        if fo and len(out) > fo: out[-fo:] *= (np.cos(np.linspace(0, np.pi, fo))*0.5 + 0.5)
    return out

_VIRAMA = "्್"
def _ends_halant(txt):
    t = txt.rstrip(" ।॥|.,;:!?‌‍")
    return len(t) > 0 and t[-1] in _VIRAMA

# ── args + model load (ONCE) ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--outdir", default="", help="if set, write <outdir>/<id>.wav (overrides per-clip out)")
ap.add_argument("--dump_raw", default="", help="debug: also write un-gated concat to <dump_raw>/<id>_raw.wav")
ap.add_argument("--bank", default=os.path.join(HERE, "reference_bank", "bank.json"))
ap.add_argument("--voice", default=f"{CHAMP}/voice_steer_ema_2026-06-17.pt")
ap.add_argument("--voc", default=f"{CHAMP}/voc_bigvgan_EMA_2026-06-11.pth")
ap.add_argument("--speed", type=float, default=0.90); ap.add_argument("--nfe", type=int, default=64)
ap.add_argument("--cfg", type=float, default=3.0); ap.add_argument("--gap", type=float, default=0.55)
ap.add_argument("--gap_halant", type=float, default=0.20)
a = ap.parse_args()

CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
# vocab.txt ships in the weights repo (-> models/ via download_weights.py); fall back to the IndicF5 cache for legacy local setups
_vocab_cands = [os.path.join(CHAMP, "vocab.txt"), os.path.join(HERE, "reference_bank", "vocab.txt")] \
    + glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"))
vocab = next((v for v in _vocab_cands if v and os.path.exists(v)), None)
if vocab is None:
    raise SystemExit("vocab.txt not found — run scripts/download_weights.py")
cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=vocab, device="cuda")
ck = torch.load(a.voice, map_location="cpu", weights_only=True)
ema = {k.replace("ema_model.", ""): v for k, v in ck["ema_model_state_dict"].items() if k not in ("initted", "step")}
cfm.load_state_dict(ema, strict=False); cfm.eval()
real_voc = load_vocoder("vocos")
class Cap:
    def __init__(s, r): s.r = r; s.last = None
    def decode(s, m): s.last = m.detach().cpu().numpy(); return s.r.decode(m)
cap = Cap(real_voc)
g = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
bsd = torch.load(a.voc, map_location="cpu"); bsd = bsd.get("model", bsd)
g.load_state_dict(bsd); g.remove_weight_norm(); g = g.cuda().eval()
for p in g.parameters(): p.requires_grad = False
def bvgan(mel):
    m = torch.from_numpy(mel).cuda()
    with torch.no_grad():
        if m.dim()==3 and m.shape[1]!=100 and m.shape[2]==100: m = m.transpose(1,2)
        return g(m).squeeze().cpu().numpy().astype(np.float32)

# ── reference bank (loaded once, ref preprocessing cached per meter) ───────────────────
import torchaudio as _ta
_bank = json.load(open(a.bank, encoding="utf-8"))
_bdir = os.path.dirname(a.bank)
_lut = {}
for _k, _v in _bank.items():
    if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v: continue
    _lut[_k.lower()] = _v
    _lut[_v["wav"].replace(".wav", "").lower()] = _v
_refcache = {}
def get_ref(meter):
    key = meter.lower().replace(".wav", "")
    if key in _refcache: return _refcache[key]
    if key not in _lut:
        if FALLBACK_METER not in _lut:
            raise SystemExit(f"[meter] '{meter}' not in bank (and fallback '{FALLBACK_METER}' missing)")
        print(f"[meter] unknown vṛtta '{meter}' -> fallback '{FALLBACK_METER}'", flush=True)
        key = FALLBACK_METER
    e = _lut[key]
    ref_wav = os.path.join(_bdir, e["wav"]); ref_text = e["ref_text"]
    sps = float(e.get("sec_per_syll", 0.26))
    ref_audio, ref_t = preprocess_ref_audio_text(ref_wav, ref_text, clip_short=True)
    ra, sr = _ta.load(ref_audio); ref_len = ra.shape[-1] / sr
    val = (ref_audio, ref_t, sps, ref_len)
    _refcache[key] = val
    print(f"[meter] {meter} -> {e['wav']} sps={sps} ref_len={ref_len:.2f}s", flush=True)
    return val

_primes = _bank.get("repeat_primes", {})
def _stitch(segs, GAPS, fric=False, halant=False):
    if len(segs) == 1: return gate(segs[0], fric=fric, halant=halant)
    b = []; last = len(segs) - 1
    for i, s in enumerate(segs): b += [gate(s, fric=(fric and i == 0), halant=(halant and i == last)), GAPS[i] if i < len(GAPS) else GAPS[-1]]
    return np.concatenate(b[:-1])

def render_clip(clip):
    meter = clip["meter"]; seed = int(clip["seed"]); no_sandhi = bool(clip["no_sandhi"])
    out = os.path.join(a.outdir, clip["id"] + ".wav") if a.outdir else clip["out"]
    ref_audio, ref_t, sps, ref_len = get_ref(meter)
    if "sps" in clip: sps = float(clip["sps"])   # per-clip duration override (0 = speed-based, no fix_duration)
    spd = float(clip.get("speed", a.speed))      # per-clip pace override (lower = slower/elongated chant)
    if "ref_wav" in clip:                         # per-clip reference override (A/B reference experiments)
        ref_audio, ref_t = preprocess_ref_audio_text(clip["ref_wav"], clip.get("ref_text", ""), clip_short=True)
        _rab, _srb = _ta.load(ref_audio); ref_len = _rab.shape[-1] / _srb
    def _basetext(p):
        return PT.model_text_sandhi(p, echo_final=False) if not no_sandhi else PT.model_text(p)
    PIECES = [_basetext(p) for p in clip["padas"]]
    if not no_sandhi:
        PIECES = [_satva(x) for x in PIECES]
    PIECES = [_danda_fix(_anusvara_m(x)) for x in PIECES]
    PIECES = [_hna_metathesis(x) for x in PIECES]
    PIECES = [_vocalic_l(x) for x in PIECES]
    # autoprime: di-repeat >=3 -> di-prime (jaya/chata); mono-repeat >=2 -> prime_mono (in-distribution
    # ta-ta-ta from sumadhwa_10_44, fixes satata-class merges). Swap ref + pada-wise resplit.
    _ra, _rt = ref_audio, ref_t
    _mono = max((_rep_depths(_aksharas(x))[0] for x in PIECES), default=1)
    _di   = max((_rep_depths(_aksharas(x))[1] for x in PIECES), default=1)
    _pick = None
    if clip.get("no_autoprime"):
        _di = 0; _mono = 0   # skip autoprime (clean single-piece render, no pada-resplit pause)
    if _di >= 3:
        _pick = next((k for k in ["prime_jaya","prime_chata"] if k in _primes and _primes[k].get("di_max",0)>=_di), None) \
                or next((k for k,v in _primes.items() if isinstance(v,dict) and v.get("di_max",0)>=_di), None)
    if _pick is None and _mono >= 2 and "prime_mono" in _primes and _primes["prime_mono"].get("mono_max",0) >= _mono:
        _pick = "prime_mono"
    if _pick:
        _pv=_primes[_pick]; _ra,_rt = preprocess_ref_audio_text(os.path.join(_bdir,_pv["wav"]), _pv["ref_text"], clip_short=True)
        _prb,_psr = _ta.load(_ra); ref_len = _prb.shape[-1]/_psr   # FIX: fix_duration is TOTAL (ref+gen); use the
        # PRIME's actual length so generated = ref_len+NSYLL*sps − prime_len stays right. Without this the meter's
        # ref_len was used against the (longer) prime ref -> generated collapsed (~1.5s). (2026-06-22)
        # HEMISTICH-WISE: no pada-resplit. The resplit inserted a mid-hemistich pause AND disrupted the
        # in-context priming (confirmed v186 satata: prime+resplit FAILED, prime alone CLEAR).
        print(f"[autoprime] {clip['id']} mono x{_mono} di x{_di} -> prime '{_pick}' (hemistich-wise, ref_len={ref_len:.2f}s)", flush=True)
    NSYLL = [n_aksharas(x) for x in PIECES]
    GAPS = [np.zeros(int(a.gap*SR) + (int(a.gap_halant*SR) if _ends_halant(_p) else 0), dtype=np.float32) for _p in PIECES]
    bseg = []
    for i, p in enumerate(PIECES):
        au = None
        for att in range(4):
            torch.manual_seed(seed + att)
            _fixd = (ref_len + NSYLL[i]*sps) if (sps > 0 and NSYLL) else None
            w, sr, _ = infer_process(_ra, _rt, p, cfm, cap, mel_spec_type="vocos", speed=spd, nfe_step=a.nfe, cfg_strength=a.cfg, device="cuda", fix_duration=_fixd)
            w = np.array(w, dtype=np.float32)
            if np.abs(w).max() > 1.5: w = w/32768.0
            if float(np.sqrt((w**2).mean())) > 0.04: au = w; break
        if au is None: au = w
        y = bvgan(cap.last); mx = np.abs(y).max(); y = y/mx*0.97 if mx > 1 else y
        bseg.append(y)
    if a.dump_raw:
        os.makedirs(a.dump_raw, exist_ok=True)
        sf.write(os.path.join(a.dump_raw, clip["id"] + "_raw.wav"), np.concatenate(bseg), SR)
    _slp = PT.align_slp1(clip["padas"][0])
    fric = bool(_slp) and _slp[0] in ("S", "z", "s", "h")   # ś/ṣ/s/h onset -> fricative-aware gate
    halant = _ends_halant(PIECES[-1])                       # त्/क्/प् final -> preserve stop burst
    final = _stitch(bseg, GAPS, fric=fric, halant=halant)
    sf.write(out, final, SR)
    return {"id": clip["id"], "dur": round(len(final)/SR, 3), "pieces": len(PIECES), "seed": seed, "out": out}

clips = json.load(open(a.shard, encoding="utf-8"))
print(f"[batch] {len(clips)} clips, model loaded", flush=True)
results = []
for clip in clips:
    try:
        r = render_clip(clip); results.append(r); print(f"OK {r['id']} {r['dur']}s seed{r['seed']}", flush=True)
    except Exception as e:
        results.append({"id": clip["id"], "error": str(e)}); print(f"FAIL {clip['id']} {e}", flush=True)
json.dump(results, open(a.results, "w"), ensure_ascii=False, indent=1)
ok = sum(1 for r in results if "error" not in r)
print(f"[batch] DONE {ok}/{len(clips)} FAIL={len(clips)-ok}", flush=True)
