#!/usr/bin/env python3
"""Guru/laghu prosodic weight tagger.

Input: list of syllable dicts (output of tts_syllabify.syllabify).
Mutates each dict in place adding:
  weight       — 'G' (guru / heavy) or 'L' (laghu / light)
  weight_cause — short string explaining why ('long_vowel', 'visarga', 'anusvara',
                 'cluster', 'pada_final_anceps', 'light')

Standard rule (Pāṇinian):
  GURU if:
    - vowel is long (A I U F X e E o O — both pure-long and the diphthongs e ai o au), OR
    - short vowel followed immediately by anusvāra (M) or visarga (H), OR
    - short vowel followed by a conjunct of ≥2 consonants before the next vowel
      (consonants counted across word boundaries; broken by `|` pauses), OR
    - syllable is pāda-final (anceps convention).
  LAGHU otherwise (short vowel + ≤1 consonant before next vowel).

Counting notes:
  M H Z V all count as consonants. `~` (nasalization) is a diacritic on the
  preceding y and does NOT add a slot. Spaces are transparent (don't break clusters).
  `|` (pause) breaks the cluster — anything after the pause is not counted.
"""

LONG_VOWELS = set('AIUFXeEoO')
SHORT_VOWELS = set('aiufx')
HEAVY_AFTER_SET = set('MH')  # immediate visarga / anusvāra → heavy


def _cluster_chars(s: str) -> list[str]:
    """Drop nasalization diacritics from a cluster string."""
    return [c for c in s if c != '~']


def tag_weights(syllables: list[dict]) -> None:
    for k, s in enumerate(syllables):
        v = s['vowel']
        if v in LONG_VOWELS:
            s['weight'] = 'G'
            s['weight_cause'] = 'long_vowel'
            continue
        # Short vowel — look at the cluster between this vowel and the next.
        gap = _cluster_chars(s['coda'])
        if not s['is_pada_final'] and k + 1 < len(syllables):
            gap += _cluster_chars(syllables[k+1]['onset'])
        if gap and gap[0] == 'H':
            s['weight'] = 'G'
            s['weight_cause'] = 'visarga'
            continue
        if gap and gap[0] == 'M':
            s['weight'] = 'G'
            s['weight_cause'] = 'anusvara'
            continue
        if len(gap) >= 2:
            s['weight'] = 'G'
            s['weight_cause'] = 'cluster'
            continue
        if s['is_pada_final']:
            s['weight'] = 'G'
            s['weight_cause'] = 'pada_final_anceps'
            continue
        s['weight'] = 'L'
        s['weight_cause'] = 'light'


if __name__ == '__main__':
    import argparse, csv, sys
    sys.path.insert(0, '<REPO>/Final_Files/Scripts')
    from tts_normalize import normalize
    from tts_g2p import to_slp1
    from tts_syllabify import syllabify

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='metadata CSV path; if omitted, run built-in samples')
    ap.add_argument('--limit', type=int, default=5)
    args = ap.parse_args()

    def show(slp1):
        syls = syllabify(slp1)
        tag_weights(syls)
        line1 = []
        line2 = []
        for s in syls:
            label = s['text']
            sep = '‖' if s['is_pada_final'] else ('·' if s['is_word_final'] else '')
            line1.append(label + sep)
            line2.append(s['weight'].rjust(len(label)) + (' ' if sep else ''))
        print('  Syl:    ', ' '.join(line1))
        print('  Weight: ', ' '.join(line2))
        pattern = ''.join(s['weight'] for s in syls)
        print(f'  Pattern: {pattern}  ({len(syls)} syllables)')

    if not args.input:
        samples = [
            # Anuṣṭubh — expected pattern roughly G/L variable, 8 syllables per pāda
            'nArAyaRam niKilapUrRaguREkadeham nirdozamApyatamamapyaKilEH suvAkyEH| asyodBavAdidamaSezaviSezato\'pi vandyam sadA priyatamam mama sannamAmi||',
            'tameva SAstrapraBavam praRamya jagadgurURAm gurumaYjasEva| viSezato me paramAKyavidyAvyAKyAm karomyanvapi cAhameva||',
            # Edge cases
            'taM ca| brahmaSezI||',
            'om|| ekam||',
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
