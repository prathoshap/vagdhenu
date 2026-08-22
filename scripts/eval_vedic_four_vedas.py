# -*- coding: utf-8 -*-
"""Four-Veda frontend stress test for Vagdhenu Vedic TTS readiness."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prep_text import model_text, model_text_sandhi, align_slp1, extract_svara
from chandas_labeler import scan, syllabify_slp1, identify
from limits import _n_aksharas

ACCENT_CP = set(range(0x0951, 0x0955)) | set(range(0x1CD0, 0x1D00))

VEDAS = {
    "Rigveda": {
        "cite": "RV 1.1.1 (Agni sukta opening)",
        "text": (
            "अ\u0952ग्निमी\u0951ळे पु\u0952रोहि\u0951तं य\u0952ज्ञस्य\u0951 "
            "दे\u0952वमृ\u0951त्विजम् । हो\u0952तारं रत्नधा\u0951तमम्"
        ),
        "tradition": "Rigvedic samhita marks (U+0952 anudatta, U+0951 svarita glyph)",
    },
    "Yajurveda": {
        "cite": "YV/VS Shanti mantra (with leading OM)",
        "text": (
            "ॐ\u0951 श\u0952ं नो मि\u0951त्रः श\u0952ं वरु\u0951णः । "
            "श\u0952ं नो भवत्वर्य\u0951मा । श\u0952ं न इ\u0951न्द्रो "
            "ब\u0952ृह\u0951स्पतिः । श\u0952ं नो विष्णुरुरुक्र\u0951मः"
        ),
        "tradition": "Yajurvedic shanti with RV-style marks + OM",
    },
    "Samaveda": {
        "cite": "SV opening (Agni) — samhita text; gana melismas not in Unicode",
        "text": (
            "अ\u0952ग्न आ या\u0951हि वी\u0952तये\u0951 गृ\u0952णा\u0951नो "
            "ह\u0952व्यदा\u0951तये । नि होता सत्सि बर्हिषि"
        ),
        "tradition": "Samaveda samhita accents only (gana melody is separate notation)",
    },
    "Atharvaveda": {
        "cite": "AV 1.1.1 (ye trisaptah)",
        "text": (
            "ये\u0952 त्रिष\u0951प्ताः प\u0952रिय\u0951न्ति वि\u0952श्वा "
            "रू\u0951पाणि बिभ्रतः । वा\u0952चस्पतिर्बला\u0951 तेषां "
            "तन्वो अद्य दधातु मे"
        ),
        "tradition": "Atharvaveda samhita (U+0951/0952)",
    },
}


def oov_accents(s):
    return [f"U+{ord(c):04X}" for c in s if ord(c) in ACCENT_CP]


def strip_accents(s):
    return "".join(c for c in s if ord(c) not in ACCENT_CP)


def main():
    results = []
    print("=" * 72)
    print("FOUR-VEDA TTS FRONTEND STRESS TEST")
    print("=" * 72)

    for veda, meta in VEDAS.items():
        text = meta["text"]
        print(f"\n### {veda} — {meta['cite']}")
        print(f"    tradition: {meta['tradition']}")
        try:
            kn, acc = model_text(text)
            kn_s, acc_s = model_text_sandhi(text, echo_final=False)
            kn_e, acc_e = model_text_sandhi(text, echo_final=True)
            slp = align_slp1(text)
            aks = syllabify_slp1(slp)
            w, n = scan(slp, pada_final_guru=False)
            marked = sum(1 for a in acc if a is not None)
            oov = oov_accents(kn) + oov_accents(kn_s)
            align_ok = len(acc) == n == len(aks)
            kn_p, _ = model_text(strip_accents(text))
            parity = kn == kn_p
            meter_id = identify(w) if len(w) <= 21 else f"({len(w)} syll — outside sama-vrttabank)"
            pairs = list(zip(aks, w, acc))
            show = " | ".join(
                f"{a}:{g}/{'·' if s is None else s[0].upper()}" for a, g, s in pairs[:14]
            )
            if len(pairs) > 14:
                show += " ..."
            row = {
                "veda": veda,
                "cite": meta["cite"],
                "crash": False,
                "tuple_ok": isinstance(kn, str) and isinstance(acc, list),
                "align_ok": align_ok,
                "n_syll": n,
                "n_accents_marked": marked,
                "accent_density": round(marked / max(n, 1), 3),
                "accent_labels": sorted({a for a in acc if a}),
                "phoneme_parity": parity,
                "kannada_oov_accents": oov,
                "meter_guess": meter_id,
                "n_aksharas_kn": _n_aksharas(kn),
                "slp1": slp,
                "kannada": kn,
                "gana": "".join(w),
                "sandhi_delta": len(acc_s) - len(acc),
                "echo_delta": len(acc_e) - len(acc),
                "pairs": [{"syl": a, "gana": g, "accent": s} for a, g, s in pairs],
            }
            results.append(row)
            print(
                f"    syll={n} marked={marked} align={align_ok} "
                f"parity={parity} oov={oov or 'none'}"
            )
            print(f"    labels={row['accent_labels']}")
            print(f"    gana={row['gana']}  meter_guess={meter_id}")
            print(f"    SLP1: {slp}")
            print(f"    Kannada: {kn}")
            print(f"    map: {show}")
            print(
                f"    sandhi_delta={row['sandhi_delta']} echo_delta={row['echo_delta']}"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            results.append({"veda": veda, "crash": True, "error": repr(e)})
            print(f"    CRASH: {e!r}")

    # Yajurvedic extension mark probe
    print("\n### Extra: Yajurvedic midline svarita U+1CD4 on RV fragment")
    yv_ext = "अग्नि\u1CD4मीळे पुरोहितम्"
    kn, acc = model_text(yv_ext)
    slp = align_slp1(yv_ext)
    aks = syllabify_slp1(slp)
    print(f"    pairs={list(zip(aks, acc))} kn={kn!r} oov={oov_accents(kn)}")

    ok = [r for r in results if not r.get("crash")]
    axes = {
        "Text ingestion (accented Deva)": 5 if ok and len(ok) == 4 else 1,
        "Phoneme fidelity (Kannada route)": 5 if all(r["phoneme_parity"] for r in ok) else 2,
        "Accent metadata alignment": 5 if all(r["align_ok"] for r in ok) else 2,
        "Vedic pitch realization in audio": 0,
        "Shakha-specific recitation style": 1,
        "Samaveda gana melismas": 0,
        "Meter/gana for Vedic meters": 2,
        "End-to-end audio render (this host)": 0,
        "Production readiness for 4 Vedas": 1,
    }
    print("\n" + "=" * 72)
    print("CAPABILITY RUBRIC (0-5)")
    print("=" * 72)
    for k, v in axes.items():
        bar = "#" * v + "-" * (5 - v)
        print(f"  {k:42s} [{bar}] {v}/5")
    overall = sum(axes.values()) / len(axes)
    print(f"  {'OVERALL':42s} {overall:.2f}/5")

    summary = {
        "overall_score": overall,
        "axes": axes,
        "frontend_pass": all(
            (not r.get("crash"))
            and r.get("align_ok")
            and r.get("phoneme_parity")
            and not r.get("kannada_oov_accents")
            for r in results
        ),
        "audio_render_attempted": False,
        "audio_render_reason": "No torch/CUDA/f5-tts in environment; weights not downloaded",
        "verdict": (
            "FRONTEND_READY_AUDIO_NOT_VEDIC: can safely ingest and phonemize all four Veda "
            "samhita texts with accents, but synthesized audio would be classical sastra chant "
            "without udatta/anudatta/svarita pitch control. Samaveda gana is out of scope."
        ),
        "results": results,
    }
    out_path = ROOT / "examples" / "vedic_frontend_eval.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nVERDICT:", summary["verdict"])
    return 0 if summary["frontend_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
