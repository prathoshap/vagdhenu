"""Golden tests for the anusvāra sandhi rule in the text frontend (`_anusvara_m`).

Run:  python -m pytest tests/  (or: python tests/test_frontend_sandhi.py)

`_anusvara_m(s, policy)` is policy-gated:
  - "acoustic" (DEFAULT): upstream behavior — y/r/l/v/ś/ṣ/s/h all → ಮ್ (an explicit consonant the
    acoustic model voices reliably; guards against bare-anusvāra conjunct drops).
  - "shastra": grammatically correct per Pāṇini 8.4.58 (anusvārasya yayi parasavarṇaḥ) — parasavarṇa
    only before "yay" (excludes श ष स ह); ಯ→candrabindu ಁ (nasalized-ya), labials→ಮ್, र ल व श ष स ह→ಂ.
    Mirrors tts_normalize.rewrite_anusvara.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from render_core import _anusvara_m

# stops assimilate homorganically under BOTH policies
COMMON = [
    ("ಸಂಕ", "ಸಙ್ಕ", "ka-varga → ṅ"),
    ("ಸಂಚ", "ಸಞ್ಚ", "ca-varga → ñ"),
    ("ಸಂಟ", "ಸಣ್ಟ", "ṭa-varga → ṇ"),
    ("ಸಂತ", "ಸನ್ತ", "ta-varga → n"),
    ("ಸಂಪ", "ಸಮ್ಪ", "pa-varga (labial) → m"),
    ("ರಾಮಂ", "ರಾಮಂ", "segment-final ṃ untouched"),
]

# the policy-divergent cases (y/r/l/v/ś/ṣ/s/h)
ACOUSTIC = [
    ("ಶಮಂ ಯಾ", "ಶಮಮ್ ಯಾ", "before य → ಮ್ (upstream default)"),
    ("ಸಿಂಹ",   "ಸಿಮ್ಹ",   "before ह → ಮ್ (upstream)"),
    ("ಸಂಸಾರ",  "ಸಮ್ಸಾರ",  "before स → ಮ್ (upstream hardens sibilant too)"),
]
SHASTRA = [
    ("ಶಮಂ ಯಾ", "ಶಮಁ ಯಾ", "before य → candrabindu ಁ (nasalized-ya)"),
    ("ಸಿಂಹ",   "ಸಿಂಹ",   "before ह → retain ṃ (siṃha)"),
    ("ಸಂವಾದ",  "ಸಂವಾದ",  "before व → retain ṃ (saṃvāda)"),
    ("ಸಂಶಯ",   "ಸಂಶಯ",   "before श → retain ṃ (saṃśaya)"),
]


def _check(cases, policy):
    fails = [f"[{policy}] {note}: {inp!r}->{_anusvara_m(inp, policy)!r} (want {exp!r})"
             for inp, exp, note in cases if _anusvara_m(inp, policy) != exp]
    return fails


def test_anusvara_policies():
    fails = []
    for policy in ("acoustic", "shastra"):
        fails += _check(COMMON, policy)          # stops identical under both
    fails += _check(ACOUSTIC, "acoustic")
    fails += _check(SHASTRA, "shastra")
    # default policy must equal acoustic (upstream reproduction)
    assert _anusvara_m("ಸಿಂಹ") == _anusvara_m("ಸಿಂಹ", "acoustic"), "default policy must be 'acoustic'"
    assert not fails, "anusvāra regressions:\n  " + "\n  ".join(fails)


if __name__ == "__main__":
    test_anusvara_policies()
    print("OK — both anusvāra policies pass (acoustic=upstream default, shastra=Pāṇini 8.4.58)")
