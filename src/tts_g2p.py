#!/usr/bin/env python3
"""SLP1 G2P for normalized Sanskrit Devanagari.

Input: normalized Devanagari (output of tts_normalize.normalize).
Output: SLP1 phoneme string.

Extensions over standard SLP1:
  Z   jihvāmūlīya (visarga → unvoiced k-varga)
  V   upadhmānīya (visarga → unvoiced p-varga)
  y~  nasalized ya (Vedic-style anunāsika before y)
  |   single danda (phrase pause)
  ||  double danda (verse-end pause)
"""

from indic_transliteration import sanscript


def to_slp1(text: str) -> str:
    s = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.SLP1)
    # Vedic extensions (pass through from input)
    s = s.replace('ᳵ', 'Z')
    s = s.replace('ᳶ', 'V')
    # Nasalized ya: sanscript renders ँय as ~ya; move tilde after the y
    s = s.replace('~y', 'y~')
    # Pause tokens: . → |  (also handles .. → ||)
    s = s.replace('.', '|')
    return s


if __name__ == '__main__':
    import argparse, csv, sys
    sys.path.insert(0, '<REPO>/Final_Files/Scripts')
    from tts_normalize import normalize

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='metadata CSV path; if omitted, run built-in samples')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    if not args.input:
        samples = [
            'ॐ॥ नारायणं निखिलपूर्णगुणैकदेहं निर्दोषमाप्यतममप्यखिलैः सुवाक्यैः। अस्योद्भवादिदमशेषविशेषतोऽपि वन्द्यं सदा प्रियतमं मम सन्नमामि॥ १॥',
            '(॥ ॐ अथातो ब्रह्मजिज्ञासा ॐ॥) ओतत्ववाची ह्योङ्कारः शास्त्रादौ॥ ९॥',
            'संयोगः संस्कारः अंक पञ्च रङ्ग दुःख दुःप्रसह वाक्क',
        ]
        for s in samples:
            n = normalize(s)
            print('IN  :', s)
            print('NORM:', n)
            print('SLP1:', to_slp1(n))
            print()
        sys.exit(0)

    with open(args.input) as f:
        r = csv.DictReader(f, delimiter='|')
        for i, row in enumerate(r):
            if i >= args.limit: break
            n = normalize(row['text'])
            print(f'#{row["shloka_no"]}')
            print('  NORM:', n)
            print('  SLP1:', to_slp1(n))
            print()
