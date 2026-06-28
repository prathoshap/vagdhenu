"""Text prep for the Prathosh voice fine-tune.
Two outputs per verse:
  - model_text : Kannada-routed (champion path), daṇḍa/number-stripped, NO phonetic conversion.
  - mfa_text   : phonetic Devanagari for MFA alignment — visarga sandhi (jihvāmūlīya/upadhmānīya/
                 sibilant-gemination, shloka-final visarga preserved) + anusvāra→homorganic nasal.
All processing is done in Devanagari (Kannada sources transliterated in first).
"""
import re
from indic_transliteration import sanscript

VIRAMA = "्"        # ्
VISARGA = "ः"       # ः
ANUSVARA = "ं"      # ं
JIHVA = "ᳵ"         # ᳵ  jihvāmūlīya
UPADH = "ᳶ"         # ᳶ  upadhmānīya

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

def detect_script(t):
    for c in t:
        o = ord(c)
        if 0x0C80 <= o <= 0x0CFF: return sanscript.KANNADA
        if 0x0900 <= o <= 0x097F: return sanscript.DEVANAGARI
    return sanscript.DEVANAGARI

def to_deva(t):
    return sanscript.transliterate(t, detect_script(t), sanscript.DEVANAGARI)

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
        else:
            out.append(c)
    return "".join(out)

def model_text(src_text):
    """PLAIN champion path (A/B Arm A): strip punct, transliterate Deva→Kannada, NO sandhi.
    This is exactly what the 4.6-MOS pilot_reciter/Prathosh champions trained on — visarga ः / anusvāra ं
    kept plain (both in IndicF5 vocab); the model learns jihvāmūlīya/upadhmānīya/homorganic acoustically."""
    slp = sanscript.transliterate(strip_punct(to_deva(src_text)),
                                  sanscript.DEVANAGARI, sanscript.SLP1)
    slp = slp.replace("F", "rU")   # long vocalic ṝ (ॄ/ॠ) → repha+ū: IndicF5 mispronounces Kannada ೄ (U+0CC4). Fix at SLP1 so tF→trU→ತ್ರೂ (2026-06-22)
    return sanscript.transliterate(slp, sanscript.SLP1, sanscript.KANNADA)

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
    """PRODUCTION normalizer: strip punct → Deva→SLP1 → visarga sandhi (utva/rutva/lopa; satva &
    jihvāmūlīya/upadhmānīya left PLAIN — the model learned those acoustically) → echo-vowel on the
    segment-final visarga (ḥ→ha/hi/hu/hai…) → SLP1→Kannada. Normalizes utva/rutva for inputs that
    lack them (matching the training texts) + fixes the clip-final visarga garble. Per render-unit."""
    slp = sanscript.transliterate(strip_punct(to_deva(src_text)), sanscript.DEVANAGARI, sanscript.SLP1)
    slp = visarga_sandhi(slp)
    if echo_final:
        slp = visarga_echo_final(slp)
    slp = slp.replace("F", "rU")   # long vocalic ṝ (ॄ/ॠ) → repha+ū: IndicF5 mispronounces Kannada ೄ (U+0CC4). Fix at SLP1 so tF→trU→ತ್ರೂ (2026-06-22) (incl. sandhi-generated F)
    return sanscript.transliterate(slp, sanscript.SLP1, sanscript.KANNADA)

def model_text_norm(src_text):
    """Kannada-safe NORMALIZED path (A/B Arm B): E48 sandhi minus jihvāmūlīya/upadhmānīya
    (plain ः kept before k/p, since ೱ/ೲ are OOV). Applies anusvāra→homorganic nasal +
    visarga→sibilant gemination, shloka-final ः preserved. All output chars are in the Kannada vocab."""
    return sanscript.transliterate(phonetic_mfa(to_deva(src_text), kannada_safe=True),
                                   sanscript.DEVANAGARI, sanscript.KANNADA)

def mfa_text(src_text):
    """Phonetic Devanagari (visarga/anusvāra conversions) — annotation for a future phonetic model."""
    return phonetic_mfa(to_deva(src_text))

def align_slp1(src_text):
    """Plain SLP1 for MFA forced-alignment (model-native convention: visarga=H, anusvāra=M, no
    phonetic conversion). Avagraha dropped (not a phone). Words space-separated."""
    slp = sanscript.transliterate(strip_punct(to_deva(src_text)),
                                  sanscript.DEVANAGARI, sanscript.SLP1)
    slp = slp.replace("'", "").replace("’", "")          # avagraha → drop
    slp = slp.replace("L", "l").replace("|", "")          # ḻ (retroflex l) → l for the model's phone set
    slp = slp.replace("F", "rU")          # long vocalic ṝ (ॄ/ॠ) → repha+ū: IndicF5 mispronounces Kannada ೄ (U+0CC4). Fix at SLP1 so tF→trU→ತ್ರೂ (2026-06-22) — keep MFA text == audio
    return re.sub(r"\s+", " ", slp).strip()

# phones MFA/the acoustic model knows (SLP1 inventory); every align_slp1 char must be one of these
PHONES = set("aAiIuUfFxXeEoO kKgGN cCjJY wWqQR tTdDn pPbBm yrlv Szs h M H ~".split()) | set(
    "aAiIuUfFxXeEoOkKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshMH~")

def word_phones(word):
    """SLP1 word → space-joined phone list (SLP1 is phonemic: 1 char = 1 phone)."""
    return " ".join(ch for ch in word if ch in PHONES)
