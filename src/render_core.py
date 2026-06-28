"""Reusable render core for Vāgdhenu — the gold per-hemistich pipeline as a callable.

This is a faithful extraction of render.py's render_clip(): the helper functions are copied
verbatim and the model-load + per-piece synthesis live in a `Renderer` class whose `render_one()`
RETURNS audio (sr, np.float32) instead of writing a wav. render.py remains the frozen batch path;
this module exists so the Gradio demo (and any interactive caller) can load the models once and
render single inputs without argparse / file I/O.

Usage:
    r = Renderer(voice_path, voc_path, bank_path, device="cuda")
    sr, audio = r.render_one("तस्मै नमः ...", meter="anuṣṭubh")
"""
import os, sys, glob, json, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_text as PT  # noqa: E402

SR = 24000

# ── helpers copied VERBATIM from render.py ───────────────────────────────────────────────
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
    return s.replace("ಹ್ಣ", "ಣ್ಹ").replace("ಹ್ನ", "ನ್ಹ")

def _vocalic_l(s):
    return s.replace("ೢ", "್ಲೃ").replace("ೣ", "್ಲೄ").replace("ಌ", "ಲೃ").replace("ೡ", "ಲೄ")

def gate(au, voice=0.08, sil=0.012, fin=0.015, fout=0.040, lead=0.03, keep=0.06, fade=True, fric=False, halant=False):
    win = int(0.02*SR); r = [float(np.sqrt((au[i:i+win]**2).mean())) for i in range(0, len(au)-win, win)]; n = len(r)
    if n == 0: return au
    if fric:
        FR = 0.006
        s = next((i for i in range(n-1) if r[i] > FR and r[i+1] > FR), int(np.argmax(r)))
        while s > 0 and r[s-1] > FR: s -= 1
        _vdef = s
    else:
        vs = next((i for i in range(n-1) if r[i] > voice and r[i+1] > sil), int(np.argmax(r))); s = vs
        while s > 0 and r[s-1] > sil: s -= 1
        _vdef = vs
    ve_thr = 0.012 if halant else 0.035
    ve = max((i for i in range(n) if r[i] > ve_thr), default=_vdef)
    keep_s = 0.12 if halant else keep
    start = max(0, s*win - int(lead*SR))
    end = min(len(au), ve*win + int(keep_s*SR)); out = au[start:end].copy()
    if fade:
        fi = (0 if fric else int(fin*SR)); fo = int((0.018 if halant else fout)*SR)
        if fi and len(out) > fi: out[:fi] *= np.linspace(0, 1, fi)
        if fo and len(out) > fo: out[-fo:] *= (np.cos(np.linspace(0, np.pi, fo))*0.5 + 0.5)
    return out

_VIRAMA = "्್"
def _ends_halant(txt):
    t = txt.rstrip(" ।॥|.,;:!?‌‍")
    return len(t) > 0 and t[-1] in _VIRAMA

_DANDAS = "।॥|"
def split_padas(text):
    """Split a free-text shloka into hemistich/pada pieces: newlines first, then dandas. Empty drop."""
    pieces = []
    for line in text.replace("॥", "।").replace("|", "।").splitlines():
        for seg in line.split("।"):
            seg = seg.strip()
            if seg: pieces.append(seg)
    return pieces or ([text.strip()] if text.strip() else [])


class Renderer:
    """Loads DiT + vocos + BigVGAN + the reference bank ONCE; render_one() synthesizes a single input."""

    def __init__(self, voice_path, voc_path, bank_path, device="cuda", vocab_file=None,
                 speed=0.90, nfe=64, cfg=3.0, gap=0.55, gap_halant=0.20):
        import bigvgan
        from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
        from f5_tts.model import DiT
        self.device = device
        self.speed = speed; self.nfe = nfe; self.cfg = cfg
        self.gap = gap; self.gap_halant = gap_halant
        self._preprocess = preprocess_ref_audio_text
        import torchaudio as ta
        self._ta = ta

        CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        vocab = vocab_file or glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"))[0]
        self.cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=vocab, device=device)
        ck = torch.load(voice_path, map_location="cpu", weights_only=True)
        ema = {k.replace("ema_model.", ""): v for k, v in ck["ema_model_state_dict"].items()
               if k not in ("initted", "step")}
        self.cfm.load_state_dict(ema, strict=False); self.cfm.eval()

        real_voc = load_vocoder("vocos")
        class Cap:
            def __init__(s, r): s.r = r; s.last = None
            def decode(s, m): s.last = m.detach().cpu().numpy(); return s.r.decode(m)
        self.cap = Cap(real_voc)

        g = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
        bsd = torch.load(voc_path, map_location="cpu"); bsd = bsd.get("model", bsd)
        g.load_state_dict(bsd); g.remove_weight_norm(); g = g.to(device).eval()
        for p in g.parameters(): p.requires_grad = False
        self.g = g

        self._bank = json.load(open(bank_path, encoding="utf-8"))
        self._bdir = os.path.dirname(bank_path)
        self._lut = {}
        for _k, _v in self._bank.items():
            if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v: continue
            self._lut[_k.lower()] = _v
            self._lut[_v["wav"].replace(".wav", "").lower()] = _v
        self._primes = self._bank.get("repeat_primes", {})
        self._refcache = {}

    def meters(self):
        return [k for k, v in self._bank.items()
                if not k.startswith("_") and isinstance(v, dict) and "wav" in v]

    def _bvgan(self, mel):
        m = torch.from_numpy(mel).to(self.device)
        with torch.no_grad():
            if m.dim() == 3 and m.shape[1] != 100 and m.shape[2] == 100: m = m.transpose(1, 2)
            return self.g(m).squeeze().cpu().numpy().astype(np.float32)

    def _get_ref(self, meter):
        key = meter.lower().replace(".wav", "")
        if key in self._refcache: return self._refcache[key]
        if key not in self._lut:
            raise ValueError(f"meter '{meter}' not in bank")
        e = self._lut[key]
        ref_wav = os.path.join(self._bdir, e["wav"]); ref_text = e["ref_text"]
        sps = float(e.get("sec_per_syll", 0.26))
        ref_audio, ref_t = self._preprocess(ref_wav, ref_text, clip_short=True)
        ra, sr = self._ta.load(ref_audio); ref_len = ra.shape[-1] / sr
        val = (ref_audio, ref_t, sps, ref_len)
        self._refcache[key] = val
        return val

    def _stitch(self, segs, GAPS, fric=False, halant=False):
        if len(segs) == 1: return gate(segs[0], fric=fric, halant=halant)
        b = []; last = len(segs) - 1
        for i, s in enumerate(segs):
            b += [gate(s, fric=(fric and i == 0), halant=(halant and i == last)),
                  GAPS[i] if i < len(GAPS) else GAPS[-1]]
        return np.concatenate(b[:-1])

    def render_one(self, text, meter, seed=60, no_sandhi=True, speed=None, sps=None):
        """Synthesize one shloka. text = free Devanagari (split into padas on newline/danda).
        Returns (sr, audio float32). Pipeline is identical to render.py's render_clip()."""
        padas = text if isinstance(text, list) else split_padas(text)
        if not padas: raise ValueError("empty text")
        ref_audio, ref_t, ref_sps, ref_len = self._get_ref(meter)
        if sps is not None: ref_sps = float(sps)
        spd = float(speed) if speed is not None else self.speed

        def _basetext(p):
            return PT.model_text_sandhi(p, echo_final=False) if not no_sandhi else PT.model_text(p)
        PIECES = [_basetext(p) for p in padas]
        if not no_sandhi:
            PIECES = [_satva(x) for x in PIECES]
        PIECES = [_danda_fix(_anusvara_m(x)) for x in PIECES]
        PIECES = [_hna_metathesis(x) for x in PIECES]
        PIECES = [_vocalic_l(x) for x in PIECES]

        _ra, _rt = ref_audio, ref_t
        _mono = max((_rep_depths(_aksharas(x))[0] for x in PIECES), default=1)
        _di   = max((_rep_depths(_aksharas(x))[1] for x in PIECES), default=1)
        _pick = None
        if _di >= 3:
            _pick = next((k for k in ["prime_jaya", "prime_chata"]
                          if k in self._primes and self._primes[k].get("di_max", 0) >= _di), None) \
                    or next((k for k, v in self._primes.items()
                             if isinstance(v, dict) and v.get("di_max", 0) >= _di), None)
        if _pick is None and _mono >= 2 and "prime_mono" in self._primes \
                and self._primes["prime_mono"].get("mono_max", 0) >= _mono:
            _pick = "prime_mono"
        if _pick:
            _pv = self._primes[_pick]
            _ra, _rt = self._preprocess(os.path.join(self._bdir, _pv["wav"]), _pv["ref_text"], clip_short=True)
            _prb, _psr = self._ta.load(_ra); ref_len = _prb.shape[-1] / _psr

        NSYLL = [n_aksharas(x) for x in PIECES]
        GAPS = [np.zeros(int(self.gap*SR) + (int(self.gap_halant*SR) if _ends_halant(_p) else 0),
                         dtype=np.float32) for _p in PIECES]
        from f5_tts.infer.utils_infer import infer_process
        bseg = []
        for i, p in enumerate(PIECES):
            au = None
            for att in range(4):
                torch.manual_seed(seed + att)
                _fixd = (ref_len + NSYLL[i]*ref_sps) if (ref_sps > 0 and NSYLL) else None
                w, sr, _ = infer_process(_ra, _rt, p, self.cfm, self.cap, mel_spec_type="vocos",
                                         speed=spd, nfe_step=self.nfe, cfg_strength=self.cfg,
                                         device=self.device, fix_duration=_fixd)
                w = np.array(w, dtype=np.float32)
                if np.abs(w).max() > 1.5: w = w/32768.0
                if float(np.sqrt((w**2).mean())) > 0.04: au = w; break
            if au is None: au = w
            y = self._bvgan(self.cap.last); mx = np.abs(y).max(); y = y/mx*0.97 if mx > 1 else y
            bseg.append(y)

        _slp = PT.align_slp1(padas[0])
        fric = bool(_slp) and _slp[0] in ("S", "z", "s", "h")
        halant = _ends_halant(PIECES[-1])
        final = self._stitch(bseg, GAPS, fric=fric, halant=halant)
        return SR, final
