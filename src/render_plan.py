# -*- coding: utf-8 -*-
"""Torch-free render planner (dry-run / noop).

Exercises the full text frontend — pada split, svara extraction, Deva→SLP1→Kannada,
gaṇa alignment, meter detection, ref-bank resolution — and returns a JSON-serializable
manifest without loading DiT/BigVGAN or touching the GPU.

Used by:
  scripts/dry_run_render.py
  demo "Plan only" path
  CI checks for Vedic saṃhitā ingestion
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import prep_text as PT

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANK = os.path.join(HERE, "reference_bank", "bank.json")
FALLBACK_METER = "vasantatilaka"

# Vedic pitch marks (must not affect akṣara counts or meter detection).
_VEDIC_MARK_RANGES = ((0x0951, 0x0954), (0x1CD0, 0x1CFF))


def is_vedic_mark(c: str) -> bool:
    if not c:
        return False
    o = ord(c[0])
    return any(lo <= o <= hi for lo, hi in _VEDIC_MARK_RANGES)


def strip_vedic_marks(text: str) -> str:
    return "".join(c for c in text if not is_vedic_mark(c))


def has_vedic_marks(text: str) -> bool:
    return any(is_vedic_mark(c) for c in text)


def n_aksharas(s: str) -> int:
    """Count orthographic akṣaras in Devanagari/Kannada; skip Vedic marks; count ॐ."""
    n = 0
    L = len(s)
    for i, c in enumerate(s):
        if is_vedic_mark(c):
            continue
        o = ord(c)
        if o == 0x0950:  # ॐ
            n += 1
            continue
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i + 1] if i + 1 < L else ""
            if nxt not in ("्", "್") and not is_vedic_mark(nxt):
                n += 1
    return n


def split_padas(text: str) -> list[str]:
    pieces = []
    for line in text.replace("॥", "।").replace("|", "।").splitlines():
        for seg in line.split("।"):
            seg = seg.strip()
            if seg:
                pieces.append(seg)
    return pieces or ([text.strip()] if text.strip() else [])


def detect_meter_key(text: str) -> str:
    """Best-effort vṛtta detection. Strips Vedic accents first (they are not phonemes)."""
    try:
        from indic_transliteration import sanscript
        from tts_syllabify import syllabify
        from tts_weight import tag_weights
        from tts_meter import detect_meter
    except Exception:
        return ""
    try:
        d = PT.to_deva(text)
        d = strip_vedic_marks(d)
        d = d.replace("॥", "|").replace("।", "|").replace("\n", " | ")
        d = "".join(
            c for c in d
            if not (c.isdigit() or ("०" <= c <= "९")) and c not in "\"'“”‘’()"
        )
        slp = re.sub(
            r"\s+", " ",
            sanscript.transliterate(d, sanscript.DEVANAGARI, sanscript.SLP1),
        ).strip()
        # ॐ → AUM is fine for syllabify; drop residual non-phones
        syls = syllabify(slp)
        tag_weights(syls)
        name = detect_meter(syls).get("name", "unknown")
    except Exception:
        return ""
    if name in ("anushtubh_half", "anushtubh"):
        return "anushtubh"
    if name in ("unknown", None, ""):
        return ""
    return name


def _load_bank(bank_path: str) -> tuple[dict, dict]:
    bank = json.load(open(bank_path, encoding="utf-8"))
    lut = {}
    for k, v in bank.items():
        if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
            continue
        lut[k.lower()] = (k, v)
        stem = v["wav"].replace(".wav", "").lower()
        lut[stem] = (k, v)
    return bank, lut


def resolve_meter(text: str, meter: str | None, bank_path: str = DEFAULT_BANK) -> dict[str, Any]:
    """Resolve meter name against the reference bank; report fallback explicitly."""
    _, lut = _load_bank(bank_path)
    detected = detect_meter_key(text) if not meter or meter in ("", "__auto__", "auto") else ""
    requested = (meter or "").lower().replace(".wav", "")
    if requested in ("", "__auto__", "auto"):
        key = detected.lower() if detected else ""
        source = "detected" if key else "fallback"
    else:
        key = requested
        source = "user"
    entry = None
    resolved = None
    if key and key in lut:
        resolved, entry = lut[key]
        recognized = True
    else:
        recognized = False
        fb = FALLBACK_METER.lower()
        if fb in lut:
            resolved, entry = lut[fb]
            source = "fallback"
        else:
            resolved = FALLBACK_METER
    return {
        "requested": meter or "__auto__",
        "detected": detected or None,
        "resolved": resolved,
        "recognized": recognized,
        "source": source,
        "ref_wav": entry.get("wav") if entry else None,
        "ref_text": entry.get("ref_text") if entry else None,
        "sec_per_syll": float(entry.get("sec_per_syll", 0.26)) if entry else None,
        "vedic_input": has_vedic_marks(text),
        "warning": (
            "Vedic saṃhitā accents are stripped for phonemes; audio would be classical "
            "śāstra chant (no udātta/anudātta/svarita pitch). Meter bank is classical vṛtta only."
            if has_vedic_marks(text) else
            (None if recognized else
             f"Unrecognized vṛtta — would fall back to '{resolved}' reference.")
        ),
    }


def _finalize_piece(kn: str, no_sandhi: bool) -> str:
    """Mirror render_core post-Kannada fixes without importing that module."""
    # local copies of the small pure-string transforms
    _SATVA = {"ಚ": "ಶ್", "ಛ": "ಶ್", "ಟ": "ಷ್", "ಠ": "ಷ್", "ತ": "ಸ್", "ಥ": "ಸ್"}
    _VMATRA = set("ಾಿೀುೂೃೄೆೇೈೊೋೌ")
    _VECHO_SHORT = {"ಿ": "ಹಿ", "ು": "ಹು", "ೃ": "ಹೃ"}
    _VLONG = set("ಾೀೂೄೆೇೈೊೋೌ")

    def satva(s):
        out = []; n = len(s); i = 0
        while i < n:
            c = s[i]
            if c == "ಃ":
                j = i + 1
                while j < n and s[j] == " ":
                    j += 1
                nxt = s[j] if j < n else ""
                if nxt in _SATVA:
                    out.append(_SATVA[nxt]); i = j; continue
            out.append(c); i += 1
        return "".join(out)

    def anusvara_m(s):
        _AN_KA = set("ಕಖಗಘಙ"); _AN_CA = set("ಚಛಜಝಞ")
        _AN_TTA = set("ಟಠಡಢಣ"); _AN_TA = set("ತಥದಧನ")
        res = []; n = len(s)
        for i, c in enumerate(s):
            if c == "ಂ":
                j = i + 1
                while j < n and s[j] == " ":
                    j += 1
                nxt = s[j] if j < n else ""
                if not nxt:
                    res.append("ಂ")
                elif nxt in _AN_KA:
                    res.append("ಙ್")
                elif nxt in _AN_CA:
                    res.append("ಞ್")
                elif nxt in _AN_TTA:
                    res.append("ಣ್")
                elif nxt in _AN_TA:
                    res.append("ನ್")
                else:
                    res.append("ಮ್")
            else:
                res.append(c)
        return "".join(res)

    def danda_fix(s):
        s = s.rstrip()
        if not s:
            return s
        if s.endswith("ಃ"):
            core = s[:-1]; pv = core[-1] if core else ""
            if pv in _VECHO_SHORT:
                s = core + _VECHO_SHORT[pv]
            elif pv in _VLONG:
                pass
            else:
                s = core + "ಹ"
        elif s.endswith("ಂ"):
            s = s[:-1] + "ಮ್"
        return s

    if not no_sandhi:
        kn = satva(kn)
    kn = danda_fix(anusvara_m(kn))
    kn = kn.replace("ಹ್ಣ", "ಣ್ಹ").replace("ಹ್ನ", "ನ್ಹ")
    kn = kn.replace("ೢ", "್ಲೃ").replace("ೣ", "್ಲೄ").replace("ಌ", "ಲೃ").replace("ೡ", "ಲೄ")
    return kn


def plan_piece(pada: str, no_sandhi: bool = True) -> dict[str, Any]:
    """Plan one pāda/hemistich: phonemes + accent_array aligned to gaṇa syllables."""
    from chandas_labeler import scan, syllabify_slp1

    if no_sandhi:
        kn_raw, accents = PT.model_text(pada)
    else:
        kn_raw, accents = PT.model_text_sandhi(pada, echo_final=False)
    kn = _finalize_piece(kn_raw, no_sandhi=no_sandhi)
    slp = PT.align_slp1(pada)
    aks = syllabify_slp1(slp)
    weights, n = scan(slp, pada_final_guru=False)
    oov = [f"U+{ord(c):04X}" for c in kn if is_vedic_mark(c)]
    align_ok = len(accents) == n == len(aks)
    return {
        "source": pada,
        "kannada": kn,
        "kannada_raw": kn_raw,
        "slp1": slp,
        "accent_array": accents,
        "syllables": aks,
        "gana": weights,
        "gana_str": "".join(weights),
        "n_syll": n,
        "n_aksharas": n_aksharas(kn),
        "n_accents_marked": sum(1 for a in accents if a is not None),
        "align_ok": align_ok,
        "kannada_oov_accents": oov,
        "syllable_map": [
            {"syl": a, "gana": g, "accent": s}
            for a, g, s in zip(aks, weights, accents)
        ],
        "ends_halant": bool(kn.rstrip()) and kn.rstrip()[-1] in "्್",
        "noop": True,
    }


def plan_render(
    text: str,
    meter: str | None = None,
    no_sandhi: bool = True,
    bank_path: str = DEFAULT_BANK,
    seed: int = 60,
) -> dict[str, Any]:
    """Full dry-run manifest for one input. Never loads models or writes audio."""
    text = (text or "").strip()
    padas = split_padas(text)
    meter_info = resolve_meter(text, meter, bank_path=bank_path)
    pieces = [plan_piece(p, no_sandhi=no_sandhi) for p in padas]
    total_syll = sum(p["n_syll"] for p in pieces)
    sps = meter_info.get("sec_per_syll") or 0.26
    # Reference duration unknown without loading wav; estimate chunk durs from sps only.
    est_chunk_s = [round(p["n_syll"] * sps, 3) for p in pieces]
    all_align = all(p["align_ok"] for p in pieces)
    any_oov = any(p["kannada_oov_accents"] for p in pieces)
    ok = bool(padas) and all_align and not any_oov
    warnings = [w for w in [meter_info.get("warning")] if w]
    if not padas:
        warnings.append("Empty text.")
    if not all_align:
        warnings.append("accent_array length mismatch vs gaṇa syllables on at least one pāda.")
    if any_oov:
        warnings.append("Vedic accent codepoints leaked into Kannada (IndicF5 OOV risk).")

    return {
        "noop": True,
        "ok": ok,
        "seed": seed,
        "no_sandhi": no_sandhi,
        "source_text": text,
        "vedic_input": has_vedic_marks(text),
        "n_padas": len(padas),
        "total_syllables": total_syll,
        "total_aksharas": sum(p["n_aksharas"] for p in pieces),
        "meter": meter_info,
        "pieces": pieces,
        "estimate": {
            "sec_per_syll": sps,
            "chunk_dur_s": est_chunk_s,
            "total_dur_s_est": round(sum(est_chunk_s) + 0.55 * max(0, len(pieces) - 1), 3),
            "note": "Estimate ignores reference clip length (unavailable in noop).",
        },
        "would_render_audio": False,
        "audio_prosody": "classical_sastra_chant",
        "accents_drive_audio": False,
        "warnings": warnings,
    }
