#!/usr/bin/env python3
"""Normalize Devanagari text for Sanskrit TTS.

Operates on saṃhitā Devanagari from the Anuvyakhyana TTS metadata CSVs.
Output is Devanagari (+ Vedic extension chars ᳵ ᳶ ँ) ready for SLP1 G2P.

Rules:
1. Strip editorial bracketing: ( ) " " ' ' - and ZWNJ. Inner text is kept (all recited).
2. Strip verse-number blocks (e.g., trailing " १॥") and stray Devanagari/ASCII numerals.
3. Expand ॐ → ओम्.
4. Visarga before unvoiced k-varga (क ख) → jihvāmūlīya ᳵ.
   Visarga before unvoiced p-varga (प फ) → upadhmānīya ᳶ.
5. Anusvāra before a varga consonant → homorganic nasal + halant.
   Anusvāra before य → chandrabindu ँ (G2P will render as nasalized-y).
   Anusvāra before {र ल व श ष स ह} → keep as anusvāra (nasalized continuant).
   Anusvāra at word-end / before । ॥ → म् (proper /m/).
"""

import re

KVARGA_UNVOICED = 'कख'
PVARGA_UNVOICED = 'पफ'
KVARGA = 'कखगघङ'
CVARGA = 'चछजझञ'
TVARGA_RETRO = 'टठडढण'
TVARGA_DENT = 'तथदधन'
PVARGA = 'पफबभम'

VARGA_TO_NASAL = {}
for c in KVARGA: VARGA_TO_NASAL[c] = 'ङ्'
for c in CVARGA: VARGA_TO_NASAL[c] = 'ञ्'
for c in TVARGA_RETRO: VARGA_TO_NASAL[c] = 'ण्'
for c in TVARGA_DENT: VARGA_TO_NASAL[c] = 'न्'
for c in PVARGA: VARGA_TO_NASAL[c] = 'म्'
VARGA_ALL = set(KVARGA + CVARGA + TVARGA_RETRO + TVARGA_DENT + PVARGA)

NON_VARGA_KEEP = set('रलवशषसह')

JIHVAMULIYA = 'ᳵ'
UPADHMANIYA = 'ᳶ'
CHANDRABINDU = 'ँ'
VISARGA = 'ः'
ANUSVARA = 'ं'

STRIP_CHARS = set('()""\'\'“”‘’-‌')

WORD_END = set(' \t\n।॥')


def strip_editorial(text: str) -> str:
    return ''.join(c for c in text if c not in STRIP_CHARS)


def strip_numerals(text: str) -> str:
    # Verse-number block: optional ws + digits + optional ws + closing danda(s)
    text = re.sub(r'\s*[०-९0-9]+\s*[।॥]+', '॥', text)
    # Any stray digits
    text = re.sub(r'[०-९0-9]+', '', text)
    return text


def expand_om(text: str) -> str:
    return text.replace('ॐ', 'ओम्')


def rewrite_visarga(text: str) -> str:
    text = re.sub(f'{VISARGA}([{KVARGA_UNVOICED}])', JIHVAMULIYA + r'\1', text)
    text = re.sub(f'{VISARGA}([{PVARGA_UNVOICED}])', UPADHMANIYA + r'\1', text)
    return text


def rewrite_anusvara(text: str) -> str:
    out = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == ANUSVARA:
            nxt = text[i+1] if i+1 < n else ''
            if nxt in VARGA_ALL:
                out.append(VARGA_TO_NASAL[nxt])
            elif nxt == 'य':
                out.append(CHANDRABINDU)
            elif nxt in NON_VARGA_KEEP:
                out.append(ANUSVARA)
            elif nxt == '' or nxt in WORD_END:
                out.append('म्')
            else:
                out.append(ANUSVARA)
        else:
            out.append(ch)
        i += 1
    return ''.join(out)


def collapse_ws(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    # Collapse repeated danda groups left by verse-number stripping
    text = re.sub(r'(॥)(\s*॥)+', '॥', text)
    text = re.sub(r'(।)(\s*।)+', '।', text)
    return text.strip()


def normalize(text: str) -> str:
    text = strip_editorial(text)
    text = strip_numerals(text)
    text = expand_om(text)
    text = rewrite_visarga(text)
    text = rewrite_anusvara(text)
    text = collapse_ws(text)
    return text


if __name__ == '__main__':
    import argparse, csv, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='metadata CSV path; if omitted, run built-in samples')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    if not args.input:
        samples = [
            'ॐ॥ नारायणं निखिलपूर्णगुणैकदेहं निर्दोषमाप्यतममप्यखिलैः सुवाक्यैः। अस्योद्भवादिदमशेषविशेषतोऽपि वन्द्यं सदा प्रियतमं मम सन्नमामि॥ १॥',
            '(॥ ॐ अथातो ब्रह्मजिज्ञासा ॐ॥) ओतत्ववाची ह्योङ्कारः शास्त्रादौ॥ ९॥',
            '"द्रव्यं कर्म च कालश्च स्वभावो जीव एव च। यदनुग्रहतः सन्ति न सन्ति यदुपेक्षया॥" १३॥',
            'तमेव शास्त्रप्रभवं प्रणम्य जगद्गुरूणां गुरुमञ्जसैव। विशेषतो मे परमाख्यविद्याव्याख्यां करोम्यन्वपि चाहमेव॥ २॥',
            'संयोगः संस्कारः अंक पञ्च रङ्ग दुःख दुःप्रसह वाक्क',
        ]
        for s in samples:
            print('IN: ', s)
            print('OUT:', normalize(s))
            print()
        sys.exit(0)

    with open(args.input) as f:
        r = csv.DictReader(f, delimiter='|')
        for i, row in enumerate(r):
            if i >= args.limit: break
            t = row['text']
            print(f'#{row["shloka_no"]}')
            print('  IN: ', t)
            print('  OUT:', normalize(t))
            print()
