"""Text prep for the Prathosh voice fine-tune.
Two outputs per verse:
  - model_text : Kannada-routed (champion path), daṇḍa/number-stripped, NO phonetic conversion.
                 Returns (kannada_text, accent_array) — accents are Vedic svara metadata aligned
                 1:1 with gaṇa syllables (chandas_labeler.scan / syllabify_slp1).
  - mfa_text   : phonetic Devanagari for MFA alignment — visarga sandhi (jihvāmūlīya/upadhmānīya/
                 sibilant-gemination, shloka-final visarga preserved) + anusvāra→homorganic nasal.
All processing is done in Devanagari (Kannada sources transliterated in first).

Vedic pārāyaṇa (future-proofing): Devanagari pitch-accent marks (anudātta U+0952, svarita glyph
U+0951 in RV print tradition, Vedic Extensions U+1CD0–U+1CFF) are stripped *before*
Deva→SLP1→Kannada so IndicF5's Kannada route never sees OOV combining marks. Accents are reattached
as a parallel syllable-aligned metadata array for a future pitch / pārāyaṇa conditioner. Unmarked
syllables are implicit udātta in RV notation (stored as None).
"""
import re
from difflib import SequenceMatcher
from indic_transliteration import sanscript

VIRAMA = "्"        # ्
VISARGA = "ः"       # ः
ANUSVARA = "ं"      # ं
JIHVA = "ᳵ"         # ᳵ  jihvāmūlīya
UPADH = "ᳶ"         # ᳶ  upadhmānīya
NUKTA = "़"         # ़
CANDRABINDU = "ँ"   # ँ

# ── Vedic pitch accents (structural; not phonemes) ───────────────────────────────────
# Labels follow Rigvedic / pārāyaṇa print convention, not raw Unicode character names:
#   U+0951 DEVANAGARI STRESS SIGN UDATTA is painted on *svarita* syllables in RV saṃhitā;
#   true udātta is typically unmarked (None in accent_array).
#   U+0952 = anudātta. Vedic Extensions carry explicit svarita / anudātta variants.
_SVARA_LABEL = {
    "\u0951": "svarita",         # RV saṃhitā svarita stroke (Unicode name: UDATTA)
    "\u0952": "anudatta",        # DEVANAGARI STRESS SIGN ANUDATTA (॒)
    "\u0953": "grave",           # DEVANAGARI GRAVE ACCENT
    "\u0954": "acute",           # DEVANAGARI ACUTE ACCENT
    "\u1CD0": "karshana",
    "\u1CD1": "shara",
    "\u1CD2": "prenka",
    "\u1CD4": "svarita",         # YAJURVEDIC MIDLINE SVARITA
    "\u1CD5": "svarita",         # YAJURVEDIC AGGRAVATED INDEPENDENT SVARITA
    "\u1CD6": "svarita",         # YAJURVEDIC INDEPENDENT SVARITA
    "\u1CD7": "svarita",
    "\u1CD8": "svarita",
    "\u1CD9": "svarita",
    "\u1CDA": "double_svarita",  # VEDIC TONE DOUBLE SVARITA
    "\u1CDB": "double_svarita",
    "\u1CDC": "anudatta",
    "\u1CDD": "anudatta",
    "\u1CDE": "anudatta",
    "\u1CDF": "anudatta",
    "\u1CE0": "svarita",         # ATHARVAVEDIC INDEPENDENT SVARITA
    "\u1CE1": "double_svarita",  # ATHARVAVEDIC DOUBLE SVARITA
}
# Syllable-closing marks that ride with the vowel nucleus (not new syllables).
_SYLLABLE_CLOSERS = {ANUSVARA, VISARGA, CANDRABINDU, JIHVA, UPADH}
# SLP1 vowels — same inventory as chandas_labeler (gaṇa detection).
_SLP1_VOWELS = set("aAiIuUfFxXeEoO")

KA_V = set("कखगघङ"); CA_V = set("चछजझञ"); TTA_V = set("टठडढण")
TA_V = set("तथदधन");  PA_V = set("पफबभम")
STOP_NASAL = {**{c:"ङ" for c in KA_V}, **{c:"ञ" for c in CA_V}, **{c:"ण" for c in TTA_V},
              **{c:"न" for c in TA_V}, **{c:"म" for c in PA_V}}
K_UNVOICED = set("कख"); P_UNVOICED = set("पफ")
# visarga → sibilant+halant assimilation (classical, by following stop/sibilant):
#   → स्  before स/त/थ   |   → श्  before श/च/छ   |   → ष्  before ष/ट/ठ
VIS_SIB = {**{c:"स" for c in "सतथ"}, **{c:"श" for c in "शचछ"}, **{c:"ष" for c in "षटठ"}}
PUNCT_DROP = set("।॥|/\\—–\"'“”‘’„«»‹›*•·().,;!?‌‍")   # daṇḍas, pipe/slash, quotes, parens, ZWJ/ZWNJ
SKIP = set(" \t\n-") | PUNCT_DROP | set("0123456789०१२३४५६७८९")

# Unicode block -> sanscript scheme, so a shloka in ANY Brahmic script is accepted: it is detected
# here and transliterated to Devanagari in to_deva(), after which the whole pipeline (which works in
# Devanagari) is unchanged. First in-block char wins. Roman input (IAST/ITRANS/HK) is NOT auto-detected
# — pass it pre-transliterated. (Tamil lacks distinct Sanskrit varga letters, so Tamil-script Sanskrit
# is inherently lossy; Grantha is the faithful Tamil-region script for Sanskrit and IS supported.)
_SCRIPT_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
    (0x11300, 0x1137F, sanscript.GRANTHA),
]
def detect_script(t):
    for c in t:
        o = ord(c)
        for lo, hi, scheme in _SCRIPT_BLOCKS:
            if lo <= o <= hi:
                return scheme
    return sanscript.DEVANAGARI

def to_deva(t):
    src = detect_script(t)
    return t if src == sanscript.DEVANAGARI else sanscript.transliterate(t, src, sanscript.DEVANAGARI)

def fix_colon(deva):
    """Stray Latin colon used as visarga: 'गुरु:-' / 'गुरु:' → 'गुरुः'."""
    deva = deva.replace(":-", VISARGA)
    return deva.replace(":", VISARGA)

def strip_punct(deva):
    """Colon→visarga, remove daṇḍas/pipes/slashes/quotes/digits, hyphen→space; avagraha & ॐ kept."""
    deva = fix_colon(deva)
    out = []
    for c in deva:
        if c in PUNCT_DROP or c.isdigit() or ("०" <= c <= "९") or c in "-–—":
            continue                         # hyphen → JOIN (compounds must stay continuous; space breaks alignment)
        out.append(c)
    return re.sub(r"\s+", " ", "".join(out)).strip()

# ── Svara (Vedic pitch accent) extraction ────────────────────────────────────────────

def _is_svara_mark(c):
    """True for Devanagari / Vedic Extensions pitch-accent combining marks."""
    if c in _SVARA_LABEL:
        return True
    o = ord(c)
    return 0x1CD0 <= o <= 0x1CFF          # Vedic Extensions block (tone marks + related)

def _svara_label(c):
    """Map a combining mark to a stable metadata label (None if not an accent)."""
    if c in _SVARA_LABEL:
        return _SVARA_LABEL[c]
    o = ord(c)
    if 0x1CD0 <= o <= 0x1CFF:
        return "vedic"                    # unknown Vedic Extensions tone mark
    return None

def _is_deva_consonant(c):
    o = ord(c)
    # Main consonants क–ह, ळ, rare/extended letters; exclude nukta/virama/matras.
    return ((0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)
            or (0x0978 <= o <= 0x097F) or o == 0x0931)

def _is_deva_ivowel(c):
    o = ord(c)
    # Independent vowels अ–औ, ॠ/ॡ, ॲ–ॿ subset used as vowels; ॐ handled separately.
    return ((0x0904 <= o <= 0x0914) or o in (0x0960, 0x0961)
            or (0x0972 <= o <= 0x0977))

def _is_deva_matra(c):
    o = ord(c)
    return ((0x093E <= o <= 0x094C) or o in (0x094E, 0x094F)
            or (0x0955 <= o <= 0x0957) or o in (0x0962, 0x0963))

def extract_svara(deva):
    """Split base phonemes from Vedic pitch accents on a (preferably punct-stripped) Devanagari string.

    Returns (clean_deva, accent_array) where accent_array[i] is the pitch label for the i-th
    vowel-bearing syllable — the same indexing as chandas_labeler.syllabify_slp1 / scan after
    Deva→SLP1. Labels: None (implicit udātta in RV) | 'anudatta' | 'svarita' | …
    Accents attach to the most recent open syllable (standard RV/YV encoding).
    """
    out = []
    accents = []                          # one entry per vowel nucleus
    pending = None                        # orphan accent before first syllable
    i, n = 0, len(deva)

    def _apply_accent(lab):
        nonlocal pending
        if accents:
            accents[-1] = lab             # last mark on the syllable wins
        else:
            pending = lab

    def _open_syllable():
        nonlocal pending
        accents.append(pending)
        pending = None

    def _absorb_closers_and_accents():
        """Consume anusvāra/visarga/candrabindu and svara marks that trail a nucleus."""
        nonlocal i
        while i < n:
            ch = deva[i]
            if _is_svara_mark(ch):
                _apply_accent(_svara_label(ch)); i += 1
            elif ch in _SYLLABLE_CLOSERS:
                out.append(ch); i += 1
            else:
                break

    while i < n:
        c = deva[i]
        if _is_svara_mark(c):
            _apply_accent(_svara_label(c)); i += 1
            continue
        if c == "ॐ":
            # Deva ॐ → SLP1 "AUM" (two vowel nuclei: A + U). Reserve two accent
            # slots so gaṇa alignment is exact; any mark on ॐ rides the first nucleus.
            out.append(c); _open_syllable(); i += 1
            _absorb_closers_and_accents()
            accents.append(None)          # second nucleus of AUM (U in UM)
            continue
        if _is_deva_ivowel(c):
            out.append(c); _open_syllable(); i += 1
            _absorb_closers_and_accents()
            continue
        if _is_deva_consonant(c):
            out.append(c); i += 1
            if i < n and deva[i] == NUKTA:
                out.append(deva[i]); i += 1
            # virāma → conjunct; syllable nucleus not yet closed
            if i < n and deva[i] == VIRAMA:
                out.append(deva[i]); i += 1
                continue
            # optional dependent vowel (matra); else inherent a
            if i < n and _is_deva_matra(deva[i]):
                out.append(deva[i]); i += 1
            _open_syllable()
            _absorb_closers_and_accents()
            continue
        if _is_deva_matra(c):
            # Lone matra (malformed / editorial) — still a nucleus for alignment safety
            out.append(c); _open_syllable(); i += 1
            _absorb_closers_and_accents()
            continue
        # spaces, avagraha, pass-through
        out.append(c); i += 1
    return "".join(out), accents

def slp1_syllable_count(slp):
    """Number of vowel nuclei in SLP1 — equals len(chandas_labeler.scan(...)[0])."""
    return sum(1 for c in slp if c in _SLP1_VOWELS)

def slp1_vowel_positions(slp):
    """Character offsets of each SLP1 vowel nucleus (gaṇa syllable index order)."""
    return [i for i, c in enumerate(slp) if c in _SLP1_VOWELS]

def align_accent_array(accents, n_syll):
    """Pad/truncate accent metadata so it is exactly 1:1 with gaṇa syllables.
    Prefer remap_accents_slp1 when the SLP1 string itself changed (sandhi/echo/F→rU)."""
    if len(accents) == n_syll:
        return list(accents)
    if len(accents) < n_syll:
        return list(accents) + [None] * (n_syll - len(accents))
    return list(accents[:n_syll])

def _merge_accent(a, b):
    """Combine two labels on one surviving nucleus; later non-None wins."""
    return b if b is not None else a

def _scatter_accents(src_accents, pre_idxs, out, post_idxs):
    """Map accents from pre syllable indices onto post indices (merge/split aware)."""
    if not post_idxs:
        return
    if not pre_idxs:
        return
    n_pre, n_post = len(pre_idxs), len(post_idxs)
    if n_pre == n_post:
        for pi, pj in zip(pre_idxs, post_idxs):
            out[pj] = _merge_accent(out[pj], src_accents[pi])
        return
    for k, pj in enumerate(post_idxs):
        start = k * n_pre // n_post
        end = max(start + 1, (k + 1) * n_pre // n_post) if n_pre >= n_post else start + 1
        end = min(end, n_pre)
        if n_pre < n_post:
            start = min(k, n_pre - 1)
            end = start + 1
        for pi in pre_idxs[start:end]:
            out[pj] = _merge_accent(out[pj], src_accents[pi])

def remap_accents_slp1(pre_slp, post_slp, accents):
    """Reattach accent_array after an SLP1 transform (sandhi, echo, F→rU, …).

    Uses SequenceMatcher on the full SLP1 strings and maps vowel nuclei inside each opcode
    block so mid-string merges/deletes (utva/lopa) keep labels on the surviving syllable
    instead of shifting via end-padding.
    """
    pre_v = slp1_vowel_positions(pre_slp)
    post_v = slp1_vowel_positions(post_slp)
    src = align_accent_array(accents, len(pre_v))
    if pre_slp == post_slp:
        return src
    if not post_v:
        return []
    if not pre_v:
        return [None] * len(post_v)

    out = [None] * len(post_v)
    pre_at = {pos: k for k, pos in enumerate(pre_v)}
    post_at = {pos: k for k, pos in enumerate(post_v)}
    sm = SequenceMatcher(a=pre_slp, b=post_slp, autojunk=False)

    orphan = []  # (pre_syll_index, accent) deleted with no in-block post target
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        pre_idxs = [pre_at[p] for p in pre_v if i1 <= p < i2]
        post_idxs = [post_at[p] for p in post_v if j1 <= p < j2]
        if tag == "equal":
            for pi, pj in zip(pre_idxs, post_idxs):
                out[pj] = _merge_accent(out[pj], src[pi])
        elif tag == "replace":
            if not post_idxs:
                # e.g. utva elides following 'a' into avagraha — keep label as orphan
                for pi in pre_idxs:
                    if src[pi] is not None:
                        orphan.append((pi, src[pi]))
            elif not pre_idxs:
                pass  # inserted nuclei stay None
            else:
                _scatter_accents(src, pre_idxs, out, post_idxs)
        elif tag == "delete":
            for pi in pre_idxs:
                if src[pi] is not None:
                    orphan.append((pi, src[pi]))
        # insert: new nuclei stay None

    for pi, lab in orphan:
        # Map deleted pre nucleus to closest surviving post nucleus (prefer left).
        frac = pi / max(1, len(pre_v) - 1) if len(pre_v) > 1 else 0.0
        target = int(round(frac * (len(post_v) - 1))) if len(post_v) > 1 else 0
        target = max(0, min(target, len(post_v) - 1))
        for cand in (target, target - 1, target + 1):
            if 0 <= cand < len(post_v):
                out[cand] = _merge_accent(out[cand], lab)
                break
    return out

def prep_deva(src_text):
    """to_deva → strip punct → extract svara. Returns (clean_deva, accent_array)."""
    return extract_svara(strip_punct(to_deva(src_text)))

def _apply_rU(slp):
    """long vocalic ṝ (ॄ/ॠ) → repha+ū: IndicF5 mispronounces Kannada ೄ (U+0CC4)."""
    return slp.replace("F", "rU")

def _deva_to_slp1(clean_deva):
    """Phoneme-only Devanagari → SLP1 with the IndicF5 ṝ workaround."""
    return _apply_rU(sanscript.transliterate(clean_deva, sanscript.DEVANAGARI, sanscript.SLP1))

def _slp1_to_kannada(slp):
    return sanscript.transliterate(slp, sanscript.SLP1, sanscript.KANNADA)

def _accents_for_slp1(clean_deva, accents, final_slp):
    """Map Devanagari-extracted accents onto final SLP1 nuclei (handles F→rU)."""
    pre = sanscript.transliterate(clean_deva, sanscript.DEVANAGARI, sanscript.SLP1)
    accents = align_accent_array(accents, slp1_syllable_count(pre))
    if pre == final_slp:
        return accents
    return remap_accents_slp1(pre, final_slp, accents)

def _next_real(s, i):
    """Index of next non-skip char after position i, or None."""
    j = i + 1
    while j < len(s) and s[j] in SKIP:
        j += 1
    return j if j < len(s) else None

def phonetic_mfa(deva, kannada_safe=False):
    """Apply visarga + anusvāra conversions on a daṇḍa/number-stripped Devanagari string.
    kannada_safe=True keeps plain ः before k/p (skips jihvāmūlīya ᳵ / upadhmānīya ᳶ, which are
    out-of-vocab for the Kannada-routed IndicF5) — used for the A/B 'normalized' arm."""
    s = strip_punct(deva)
    # locate the shloka-final visarga (last visarga with no real char after it) -> preserve
    last_vis_final = None
    for i, c in enumerate(s):
        if c == VISARGA and _next_real(s, i) is None:
            last_vis_final = i
    out = []
    for i, c in enumerate(s):
        if c == VISARGA:
            if i == last_vis_final:           # shloka-final → keep ः
                out.append(VISARGA); continue
            j = _next_real(s, i)
            nxt = s[j] if j is not None else None
            if nxt in K_UNVOICED:   out.append(VISARGA if kannada_safe else JIHVA)
            elif nxt in P_UNVOICED: out.append(VISARGA if kannada_safe else UPADH)
            elif nxt in VIS_SIB:    out.append(VIS_SIB[nxt] + VIRAMA)   # s/ś/ṣ/c/ch/ṭ/ṭh/t/th
            else:                   out.append(VISARGA)     # voiced/vowel/semivowel/h → leave
        elif c == ANUSVARA:
            j = _next_real(s, i)
            nxt = s[j] if j is not None else None
            if nxt in STOP_NASAL:   out.append(STOP_NASAL[nxt] + VIRAMA)
            else:                   out.append(ANUSVARA)    # before sibilant/semivowel/h/end → keep
        elif _is_svara_mark(c):
            continue                  # defensive: accents must not reach Kannada/MFA paths
        else:
            out.append(c)
    return "".join(out)

def model_text(src_text):
    """PLAIN champion path (A/B Arm A): strip punct, extract Vedic svara, Deva→SLP1→Kannada, NO sandhi.

    Returns (kannada_text, accent_array) where accent_array aligns 1:1 with gaṇa syllables
    (one entry per SLP1 vowel nucleus — same count as chandas_labeler.scan / syllabify_slp1).

    This is exactly what the 4.6-MOS pilot_reciter/Prathosh champions trained on — visarga ः / anusvāra ं
    kept plain (both in IndicF5 vocab); the model learns jihvāmūlīya/upadhmānīya/homorganic acoustically.
    Pitch accents never enter the Kannada string (IndicF5-safe).
    """
    clean, accents = prep_deva(src_text)
    slp = _deva_to_slp1(clean)
    accents = _accents_for_slp1(clean, accents, slp)
    return _slp1_to_kannada(slp), accents

# ── word-boundary visarga sandhi (SLP1) ──────────────────────────────────────────────
_VS_VOICED = set("gGjJqQdDbBNYRnmyrlvh"); _VS_OTHERV = set("iIuUfFxXeEoO")
_VS_ALLV = set("aAiIuUfFxXeEoO"); _VS_LEN = {"a":"A","i":"I","u":"U","f":"F","A":"A","I":"I","U":"U"}
# satva (ḥ→ś/ṣ/s before c/ṭ/t & sibilants) and jihvāmūlīya/upadhmānīya (ḥ before k/kh/p/ph) are
# DELIBERATELY NOT applied — the training texts left these as PLAIN ः and the model learned them
# acoustically (A/B 2026-06-15: plain > resolved for satva). Only utva/rutva/lopa are applied.

def visarga_sandhi(slp):
    """Word-boundary visarga sandhi — utva/rutva/lopa ONLY (the sandhi that WAS resolved in the
    training texts). On a space-separated SLP1 string:
      1 utva : aH + a → o ' (avagraha) ; aH + voiced-cons → o
      2 rutva: (i/u/e/o…)H + vowel/voiced-cons → r
      3 lopa : āH + vowel/voiced → ā ; aH + (vowel≠a) → a ; saḥ/eṣaḥ + (≠a) → sa/eṣa ; H + r → drop + lengthen
    ḥ before any UNVOICED consonant or sibilant (satva / jihvāmūlīya / upadhmānīya contexts) → KEPT PLAIN.
    Segment-final visarga preserved (echo handled separately)."""
    ws = slp.split(" "); i = 0; out = []
    while i < len(ws):
        w = ws[i]
        if w.endswith("H") and i < len(ws) - 1 and len(w) >= 2:
            V = w[-2]; base = w[:-1]; nxt = ws[i + 1]; F = nxt[0] if nxt else ""
            if F == "r":                                       out.append(base[:-1] + _VS_LEN.get(V, V)); i += 1; continue   # H+r: drop+lengthen
            if w in ("saH", "ezaH") and F != "a":              out.append(base); i += 1; continue                            # saḥ/eṣaḥ
            if F not in _VS_ALLV and F not in _VS_VOICED:      out.append(w); i += 1; continue                              # satva/sibilant/k/p → KEEP plain
            if V == "a":
                if F == "a":                                   out.append(base[:-1] + "o"); ws[i + 1] = "'" + nxt[1:]; i += 1; continue  # utva aH+a
                if F in _VS_VOICED:                             out.append(base[:-1] + "o"); i += 1; continue               # utva aH+voiced
                out.append(base); i += 1; continue                                                                         # lopa aH+vowel
            if V == "A":                                       out.append(base); i += 1; continue                          # lopa āH
            if V in _VS_OTHERV:                                 out.append(base + "r"); i += 1; continue                    # rutva
            out.append(w); i += 1
        else:
            out.append(w); i += 1
    return " ".join(out)

_VS_VOWELS = "aAiIuUfFxXeEoO"
def visarga_echo_final(slp):
    """Chant echo-vowel for the segment-final visarga: ḥ → h + the preceding vowel.
    rāmaḥ→rāmaha, śrīpatiḥ→śrīpatihi, guruḥ→guruhu, …aiḥ(E)→…aihai. Only the LAST word's
    visarga (the chant pause) — internal/boundary visargas are handled by visarga_sandhi."""
    ws = slp.split(" ")
    if ws and ws[-1].endswith("H") and len(ws[-1]) >= 2 and ws[-1][-2] in _VS_VOWELS:
        ws[-1] = ws[-1][:-1] + "h" + ws[-1][-2]
    return " ".join(ws)

def model_text_sandhi(src_text, echo_final=True):
    """PRODUCTION normalizer: extract svara → Deva→SLP1 → visarga sandhi (utva/rutva/lopa; satva &
    jihvāmūlīya/upadhmānīya left PLAIN — the model learned those acoustically) → echo-vowel on the
    segment-final visarga (ḥ→ha/hi/hu/hai…) → SLP1→Kannada.

    Returns (kannada_text, accent_array). Accents are remapped through each SLP1 transform
    (visarga sandhi, optional echo, F→rU) via remap_accents_slp1 so mid-string merges keep
    labels on the surviving nucleus (1:1 with final gaṇa syllables).
    """
    clean, accents = prep_deva(src_text)
    slp0 = sanscript.transliterate(clean, sanscript.DEVANAGARI, sanscript.SLP1)
    accents = align_accent_array(accents, slp1_syllable_count(slp0))
    slp1 = visarga_sandhi(slp0)
    accents = remap_accents_slp1(slp0, slp1, accents)
    if echo_final:
        slp2 = visarga_echo_final(slp1)
        accents = remap_accents_slp1(slp1, slp2, accents)
        slp1 = slp2
    slp = _apply_rU(slp1)
    accents = remap_accents_slp1(slp1, slp, accents)
    return _slp1_to_kannada(slp), accents

def model_text_norm(src_text):
    """Kannada-safe NORMALIZED path (A/B Arm B): E48 sandhi minus jihvāmūlīya/upadhmānīya
    (plain ः kept before k/p, since ೱ/ೲ are OOV). Applies anusvāra→homorganic nasal +
    visarga→sibilant gemination, shloka-final ः preserved. All output chars are in the Kannada vocab.

    Returns (kannada_text, accent_array) remapped through phonetic_mfa + F→rU onto final nuclei.
    """
    clean, accents = prep_deva(src_text)
    phon = phonetic_mfa(clean, kannada_safe=True)
    pre = sanscript.transliterate(clean, sanscript.DEVANAGARI, sanscript.SLP1)
    accents = align_accent_array(accents, slp1_syllable_count(pre))
    mid = sanscript.transliterate(phon, sanscript.DEVANAGARI, sanscript.SLP1)
    accents = remap_accents_slp1(pre, mid, accents)
    final = _apply_rU(mid)
    accents = remap_accents_slp1(mid, final, accents)
    return sanscript.transliterate(phon, sanscript.DEVANAGARI, sanscript.KANNADA), accents

def mfa_text(src_text):
    """Phonetic Devanagari (visarga/anusvāra conversions) — annotation for a future phonetic model.
    Vedic accent marks are stripped (they are not MFA phones)."""
    clean, _ = prep_deva(src_text)
    return phonetic_mfa(clean)

def align_slp1(src_text):
    """Plain SLP1 for MFA forced-alignment (model-native convention: visarga=H, anusvāra=M, no
    phonetic conversion). Avagraha dropped (not a phone). Words space-separated.
    Vedic accent marks are stripped before transliteration."""
    clean, _ = prep_deva(src_text)
    slp = sanscript.transliterate(clean, sanscript.DEVANAGARI, sanscript.SLP1)
    slp = slp.replace("'", "").replace("’", "")          # avagraha → drop
    slp = slp.replace("L", "l").replace("|", "")          # ḻ (retroflex l) → l for the model's phone set
    slp = slp.replace("F", "rU")          # long vocalic ṝ → repha+ū — keep MFA text == audio
    return re.sub(r"\s+", " ", slp).strip()

# phones MFA/the acoustic model knows (SLP1 inventory); every align_slp1 char must be one of these
PHONES = set("aAiIuUfFxXeEoO kKgGN cCjJY wWqQR tTdDn pPbBm yrlv Szs h M H ~".split()) | set(
    "aAiIuUfFxXeEoOkKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshMH~")

def word_phones(word):
    """SLP1 word → space-joined phone list (SLP1 is phonemic: 1 char = 1 phone)."""
    return " ".join(ch for ch in word if ch in PHONES)
