#!/usr/bin/env python3
"""Syllable segmenter for SLP1 Sanskrit.

Input: SLP1 string (output of tts_g2p.to_slp1).
Output: list of syllable dicts.

Each syllable:
  text   — SLP1 substring (onset + vowel + coda)
  onset  — consonant cluster before the vowel
  vowel  — the vowel character
  coda   — consonants attached after the vowel (only if word-final or last of a ≥2-consonant
           cluster, per the standard "maximize onset" rule)
  is_word_final  — true if next non-space/pause character is space/pause/eof
  is_pada_final  — true if this is the last syllable before a `|` or end-of-string

Prosodic gap (for guru/laghu, computed in task 4):
  consonants_to_next_vowel — count of consonant phonemes between this vowel and the
    next vowel, counting M H Z V as consonants, ignoring spaces, broken by `|` pauses.
    `y~` counts as 1 consonant (the ~ is a diacritic).

Quirks:
  '  (avagraha) — silent elision marker; skipped entirely.
  ~  (nasalization) — attaches to the preceding y, treated as part of that consonant.
"""

VOWELS = set('aAiIuUfFxXeEoO')
SEPS = set(' |')


def _is_consonant(c: str) -> bool:
    return c not in VOWELS and c not in SEPS and c not in "'~"


def syllabify(slp1: str) -> list[dict]:
    syllables = []
    n = len(slp1)
    i = 0
    onset_buf = []  # consonants accumulated for the next syllable's onset

    def attach_trailing_to_last(cons: list[str]):
        """All trailing consonants become coda of the last syllable."""
        if not syllables:
            return
        text = ''.join(cons)
        syllables[-1]['coda'] += text
        syllables[-1]['text'] += text

    while i < n:
        c = slp1[i]
        if c == "'":
            # Avagraha — silent, skip
            i += 1
            continue
        if c in SEPS:
            # Flush any pending consonants as coda of previous syllable
            attach_trailing_to_last(onset_buf)
            onset_buf = []
            if c == '|' and syllables:
                syllables[-1]['is_pada_final'] = True
                syllables[-1]['is_word_final'] = True
            elif c == ' ' and syllables:
                syllables[-1]['is_word_final'] = True
            i += 1
            continue
        if c in VOWELS:
            # Emit a syllable: onset_buf + vowel
            syl = {
                'text': ''.join(onset_buf) + c,
                'onset': ''.join(onset_buf),
                'vowel': c,
                'coda': '',
                'is_word_final': False,
                'is_pada_final': False,
            }
            syllables.append(syl)
            onset_buf = []
            i += 1
            # Look ahead for the next vowel; consonants between split per maximize-onset rule
            cons_between = []
            while i < n and slp1[i] not in VOWELS and slp1[i] not in SEPS and slp1[i] != "'":
                cc = slp1[i]
                if cc == '~' and cons_between:
                    # Attach nasalization to preceding consonant (y~ as one unit)
                    cons_between[-1] += '~'
                else:
                    cons_between.append(cc)
                i += 1
            if i < n and slp1[i] in VOWELS:
                # Intervocalic cluster — maximize onset: last consonant is onset of next syl
                if len(cons_between) >= 2:
                    coda_part = cons_between[:-1]
                    onset_part = cons_between[-1:]
                    syllables[-1]['coda'] = ''.join(coda_part)
                    syllables[-1]['text'] += syllables[-1]['coda']
                    onset_buf = onset_part
                elif len(cons_between) == 1:
                    onset_buf = cons_between
                # else cons_between empty — adjacent vowels, no onset for next
            else:
                # Hit a separator or eof — all trailing consonants are coda of current syl
                syllables[-1]['coda'] = ''.join(cons_between)
                syllables[-1]['text'] += syllables[-1]['coda']
                onset_buf = []
            continue
        # Consonant — buffer it for next syllable's onset
        if c == '~' and onset_buf:
            onset_buf[-1] += '~'
        else:
            onset_buf.append(c)
        i += 1

    # End of string: flush any trailing consonants
    attach_trailing_to_last(onset_buf)
    if syllables:
        syllables[-1]['is_word_final'] = True
        syllables[-1]['is_pada_final'] = True
    return syllables


if __name__ == '__main__':
    import argparse, csv, sys
    sys.path.insert(0, '<REPO>/Final_Files/Scripts')
    from tts_normalize import normalize
    from tts_g2p import to_slp1

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='metadata CSV path; if omitted, run built-in samples')
    ap.add_argument('--limit', type=int, default=5)
    args = ap.parse_args()

    def show(slp1):
        syls = syllabify(slp1)
        marks = []
        for s in syls:
            tag = ''
            if s['is_pada_final']: tag = '‖'
            elif s['is_word_final']: tag = '·'
            marks.append(s['text'] + tag)
        print('  Syl:', ' '.join(marks))
        print(f'  ({len(syls)} syllables)')

    if not args.input:
        samples = [
            'nArAyaRam',
            'brahmajijYAsA',
            'om|| nArAyaRam niKilapUrRaguREkadeham',
            'tat ca sandiSyate',
            'say~ogaH saMskAraH duZKa duVprasaha vAkka',
        ]
        for s in samples:
            print('SLP1:', s)
            show(s)
            print()
        sys.exit(0)

    with open(args.input) as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='|')):
            if i >= args.limit: break
            slp1 = to_slp1(normalize(row['text']))
            print(f'#{row["shloka_no"]}')
            print('  SLP1:', slp1)
            show(slp1)
            print()
