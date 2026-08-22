# -*- coding: utf-8 -*-
"""Frontend tests for Vedic svara routing (no GPU / no torch)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chandas_labeler import scan, syllabify_slp1
from prep_text import align_slp1, model_text, model_text_sandhi, model_text_norm
from render_plan import plan_render, n_aksharas, has_vedic_marks
import limits

ACCENT_CP = set(range(0x0951, 0x0955)) | set(range(0x1CD0, 0x1D00))


def _oov_accents(s):
    return [c for c in s if ord(c) in ACCENT_CP]


def test_model_text_returns_tuple():
    kn, acc = model_text("रामः")
    assert isinstance(kn, str) and isinstance(acc, list)
    kn2, acc2 = model_text_sandhi("रामः", echo_final=False)
    assert isinstance(kn2, str) and isinstance(acc2, list)
    kn3, acc3 = model_text_norm("संस्कार")
    assert isinstance(kn3, str) and isinstance(acc3, list)


def test_accented_phoneme_parity_and_no_oov():
    vedic = "अ\u0952ग्निमी\u0951ळे पु\u0952रोहि\u0951तम्"
    plain = "अग्निमीळे पुरोहितम्"
    kn_v, acc_v = model_text(vedic)
    kn_p, acc_p = model_text(plain)
    assert kn_v == kn_p
    assert _oov_accents(kn_v) == []
    assert any(a is not None for a in acc_v)
    assert all(a is None for a in acc_p)


def test_accent_array_aligns_with_gana_syllables():
    vedic = "अ\u0952ग्निमी\u0951ळे पु\u0952रोहि\u0951तम्"
    kn, acc = model_text(vedic)
    slp = align_slp1(vedic)
    aks = syllabify_slp1(slp)
    _w, n = scan(slp, pada_final_guru=False)
    assert len(acc) == n == len(aks)
    assert list(zip(aks, acc)) == [
        ("a", "anudatta"),
        ("gni", None),
        ("mI", "svarita"),
        ("le", None),
        ("pu", "anudatta"),
        ("ro", None),
        ("hi", "svarita"),
        ("tam", None),
    ]
    assert kn.startswith("ಅ")


def test_om_two_nucleus_alignment():
    src = "ॐ\u0951 नम\u0952\u0903"
    kn, acc = model_text(src)
    aks = syllabify_slp1(align_slp1(src))
    assert list(zip(aks, acc)) == [
        ("A", "svarita"),
        ("UM", None),
        ("na", None),
        ("maH", "anudatta"),
    ]
    assert _oov_accents(kn) == []


def test_sandhi_utva_keeps_accent_on_merged_nucleus():
    """रा॑मः अ॒स्ति → rAmo 'sti: anudatta on a must land on mo, not sti."""
    from prep_text import remap_accents_slp1, visarga_sandhi
    from indic_transliteration import sanscript

    src = "रा\u0951मः अ\u0952स्ति"
    _kn, acc = model_text_sandhi(src, echo_final=False)
    clean, raw = __import__("prep_text", fromlist=["prep_deva"]).prep_deva(src)
    pre = sanscript.transliterate(clean, sanscript.DEVANAGARI, sanscript.SLP1)
    post = visarga_sandhi(pre)
    aks = syllabify_slp1(post)
    assert list(zip(aks, acc)) == [
        ("rA", "svarita"),
        ("mo", "anudatta"),
        ("sti", None),
    ]
    # direct remap unit check
    mapped = remap_accents_slp1(pre, post, raw)
    assert mapped == ["svarita", "anudatta", None]


def test_sandhi_echo_pads_new_nucleus_with_none():
    src = "गुरु\u0952ः"
    _kn, acc = model_text_sandhi(src, echo_final=True)
    assert acc == [None, "anudatta", None]


def test_yajurvedic_midline_svarita():
    src = "अग्नि\u1CD4मीळे"
    _kn, acc = model_text(src)
    aks = syllabify_slp1(align_slp1(src))
    assert ("gni", "svarita") in list(zip(aks, acc))


def test_classical_shloka_all_none_accents():
    src = "वसुदेवसुतं देवं कंसचाणूरमर्दनम्"
    kn, acc = model_text(src)
    assert all(a is None for a in acc)
    assert len(acc) == scan(align_slp1(src), pada_final_guru=False)[1]
    assert kn


def test_limits_skip_marks_count_om():
    assert limits._n_aksharas("अ\u0952ग्नि") == limits._n_aksharas("अग्नि")
    assert limits._n_aksharas("ॐ") == 1
    assert limits.has_vedic_marks("अ\u0952")
    assert limits.validate_one_shloka("अ\u0952ग्निमी\u0951ळे") is None
    assert limits.vedic_noop_warning("अ\u0952")


def test_n_aksharas_empty_safe():
    assert n_aksharas("") == 0
    assert n_aksharas("ಕ") == 1


def test_plan_render_noop_vedic():
    plan = plan_render("अ\u0952ग्निमी\u0951ळे पुरोहितम्")
    assert plan["noop"] is True
    assert plan["ok"] is True
    assert plan["vedic_input"] is True
    assert plan["accents_drive_audio"] is False
    assert all(p["align_ok"] for p in plan["pieces"])
    assert has_vedic_marks(plan["source_text"])


def test_plan_render_classical_meter():
    text = (
        "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\n"
        "देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥"
    )
    plan = plan_render(text)
    assert plan["ok"]
    assert not plan["vedic_input"]
    assert plan["n_padas"] == 2
