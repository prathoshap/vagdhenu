"""Chandas / gaṇa labeler — per-akṣara laghu(L,1 mātrā)/guru(G,2 mātrā) scansion + meter ID.
Phase-1 component for vṛtta-conditioning. Operates on SLP1 (single-char phonemes).
"""
import re
from indic_transliteration import sanscript

SHORT = set("aiufx")          # a i u ṛ ḷ   (short vowels)
LONG  = set("AIUFXeEoO")       # ā ī ū ṝ ḹ e ai o au (always guru by length)
VOWELS = SHORT | LONG
MARKS = set("MH~")             # anusvāra, visarga, candrabindu (close the syllable -> guru)

def to_slp1(text, src="devanagari"):
    scheme = {"devanagari": sanscript.DEVANAGARI, "kannada": sanscript.KANNADA, "slp1": sanscript.SLP1}[src]
    return sanscript.transliterate(text, scheme, sanscript.SLP1)

def _clean(slp1):
    # keep only phoneme-relevant chars: vowels, consonants, M H ~ ; drop spaces/daṇḍa/digits/avagraha
    return "".join(c for c in slp1 if c.isalpha() or c in MARKS)

def scan(slp1_text, pada_final_guru=True):
    """Return (weights list 'L'/'G', n_syllables) for a clean SLP1 string."""
    s = _clean(slp1_text)
    vpos = [i for i, c in enumerate(s) if c in VOWELS]
    w = []
    for k, i in enumerate(vpos):
        v = s[i]
        nxt = vpos[k + 1] if k + 1 < len(vpos) else len(s)
        between = s[i + 1:nxt]                       # chars after this vowel, before next vowel
        if v in LONG:
            w.append("G")
        elif any(m in between for m in MARKS):        # short vowel + anusvāra/visarga -> guru
            w.append("G")
        else:
            cons = [c for c in between if c not in VOWELS and c not in MARKS]
            w.append("G" if len(cons) >= 2 else "L")  # short vowel + conjunct -> guru (position)
    if pada_final_guru and w:
        w[-1] = "G"                                   # pādānta convention (anceps -> guru)
    return w, len(w)

# --- known sama-vṛtta signatures (one pāda; '.' = anceps) for meter ID ---
SIGNATURES = {
    "GGLGGLLGLGG": "indravajrā(11)",
    "LGLGGLLGLGG": "upendravajrā(11)",
    "LGLGGLLGLGLG": "vaṃśastha(12)",
    "GGLGLLLGLLGLGG": "vasantatilakā(14)",
    "LLLLLLGGGLGGLGG": "mālinī(15)",
    "GGGGLLLLLGGLGGLGG": "mandākrāntā(17)",
    "LGGGGGLLLLLGGLLLG": "śikhariṇī(17)",
    "LGLLLGLGLLLGLGGLG": "pṛthvī(17)",
    "GGGLLGLGLLLGGGLGGLG": "śārdūlavikrīḍita(19)",
    "GGGGLGGLLLLLLGGLGGLGG": "sragdharā(21)",
}
def identify(pada_weights):
    return SIGNATURES.get("".join(pada_weights), f"({len(pada_weights)})")

# yati (caesura): within-pāda akṣara positions AFTER which the caesura falls
YATI = {
    "mālinī(15)": [8], "śikhariṇī(17)": [6], "mandākrāntā(17)": [4, 10],
    "śārdūlavikrīḍita(19)": [12], "sragdharā(21)": [7, 14],
}

def syllabify_slp1(slp1):
    """Split clean SLP1 into orthographic akṣaras: [onset consonants][vowel][M/H/~]*. Count == #vowels."""
    s = _clean(slp1); aks = []; start = 0; j = 0; n = len(s)
    while j < n:
        if s[j] in VOWELS:
            e = j + 1
            while e < n and s[e] in MARKS: e += 1
            aks.append(s[start:e]); start = e; j = e
        else:
            j += 1
    if start < n:                       # trailing consonant (e.g. pāda-final halanta)
        aks[-1] = aks[-1] + s[start:] if aks else s[start:]
    return aks

def label_verse(text, src="devanagari", pada_len=None, meter=None):
    """Return per-pāda list of per-akṣara feature dicts:
       {aksara, w(L/G), matra, pada_idx, pos, pos_norm, yati_end}."""
    slp1 = to_slp1(text, src)
    aks = syllabify_slp1(slp1)
    w, _ = scan(slp1, pada_final_guru=False)
    N = len(aks)
    if pada_len is None: pada_len = N                 # single pāda fallback
    ypos = set(YATI.get(meter or "", []))
    padas = []
    for p0 in range(0, N, pada_len):
        seg = []
        plen = min(pada_len, N - p0)
        for k in range(plen):
            i = p0 + k
            seg.append(dict(
                aksara=aks[i], w=w[i], matra=2 if w[i] == "G" else 1,
                pada_idx=p0 // pada_len, pos=k,
                pos_norm=round(k / max(1, plen - 1), 3),
                yati_end=(k + 1) in ypos,
            ))
        padas.append(seg)
    return padas

# --- Kannada char-broadcast (the model tokenizes Kannada chars; need L/G per char) ---
def _kn_class(c):
    o = ord(c)
    if 0x0C85 <= o <= 0x0C94: return "base"                         # independent vowels
    if 0x0C95 <= o <= 0x0CB9 or o == 0x0CDE: return "base"          # consonants
    if o == 0x0CCD: return "virama"
    if (0x0CBE <= o <= 0x0CCC) or o in (0x0CD5, 0x0CD6, 0x0C82, 0x0C83): return "sign"  # matra/anusvāra/visarga
    return "other"

def kannada_aksharas(text):
    """Segment Kannada into akṣaras → list of (kind, [char_indices]); kind 'aks' or 'fill'.
       Vowel-less coda (trailing virama) merges into the previous akṣara."""
    segs = []; cur = []; prev_vir = False
    for i, c in enumerate(text):
        cl = _kn_class(c)
        if cl == "other":
            if cur: segs.append(["aks", cur]); cur = []
            segs.append(["fill", [i]]); prev_vir = False; continue
        if cl == "base" and not prev_vir and cur:
            segs.append(["aks", cur]); cur = []
        cur.append(i); prev_vir = (cl == "virama")
    if cur: segs.append(["aks", cur])
    # merge trailing-virama (vowel-less) akṣaras into the previous akṣara
    out = []
    for kind, idxs in segs:
        if kind == "aks" and _kn_class(text[idxs[-1]]) == "virama" and out and out[-1][0] == "aks":
            out[-1][1].extend(idxs)
        else:
            out.append([kind, idxs])
    return out

def char_gana_ids(kn_text):
    """Per-char L/G id aligned to kn_text: 0=filler, 1=laghu, 2=guru. Returns (ids, ok, n_aks, n_lg)."""
    segs = kannada_aksharas(kn_text)
    aks = [idxs for k, idxs in segs if k == "aks"]
    w, _ = scan(to_slp1(kn_text, "kannada"), pada_final_guru=False)
    ids = [0] * len(kn_text)
    ok = (len(aks) == len(w))
    if ok:
        for k, idxs in enumerate(aks):
            g = 2 if w[k] == "G" else 1
            for i in idxs: ids[i] = g
    return ids, ok, len(aks), len(w)


if __name__ == "__main__":
    import json, sys
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                       "<REPO>/Sarvamoola/Texts/sumadhvavijayah_mula.json"))
    exact = total = 0
    diffs = {}
    sample = {}
    for sg in d["sargas"]:
        for v in sg["verses"]:
            txt = " ".join(v.get("padas") or [v["text"]])
            _, n = scan(to_slp1(txt, "devanagari"), pada_final_guru=False)
            total += 1
            st = v.get("syll_total")
            if st is None: continue
            if n == st: exact += 1
            else: diffs[n - st] = diffs.get(n - st, 0) + 1
            m = v.get("meter", "?")
            if m not in sample and v.get("pada_len"):
                pl = v["pada_len"]
                wv, _ = scan(to_slp1(v["padas"][0], "devanagari"))
                sample[m] = (v["padas"][0][:40], "".join(wv[:pl]), identify(wv[:pl]))
    print(f"syllable-count match: {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"mismatch distribution (mine - annotated): {dict(sorted(diffs.items()))}")
    print("\nmeter L/G signatures (my scan -> identification):")
    for m, (snip, sig, ident) in sorted(sample.items()):
        print(f"  {m:32s} pada1='{snip}...'  {sig} -> {ident}")
