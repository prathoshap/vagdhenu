#!/usr/bin/env python3
"""Vṛtta (meter) detection and pāda-position assignment for Sanskrit shlokas.

Detection strategy:
  1. Total syllable count narrows the candidate vṛttas (each has a fixed total).
  2. For rigid-pattern vṛttas (vasantatilakā, mandākrāntā, etc.) the four pādas
     must match the canonical pattern (last syllable per pāda is anceps).
  3. Upajāti / indravajra / upendravajra share total=44; each pāda must
     independently match either the indravajra or upendravajra template.
  4. 32-syl rows that don't match a rigid pattern default to anuṣṭubh (the
     pattern is highly flexible inside the 8-syl pāda).
  5. Anything else → 'unknown' (treated as gadya / flat F0 template downstream).

Once meter is known, each syllable gets:
  pada_index           — 0-based pāda number (0..3 typically)
  pos_in_pada          — 0-based syllable position within its pāda
  pos_in_pada_norm     — pos / (pada_length - 1) in [0, 1] (for embedding)
"""

# Each entry: (name, syllables_per_pada, list_of_valid_pada_patterns)
# In each pattern, '*' or last position is treated as anceps.
METERS = [
    # 11-syl
    ('indravajra',         11, ['GGLGGLLGLGG']),
    ('upendravajra',       11, ['LGLGGLLGLGG']),
    ('upajati',            11, ['GGLGGLLGLGG', 'LGLGGLLGLGG']),
    # 12-syl
    ('vamshastha',         12, ['LGLGGLLGLGLG']),
    ('indravamsha',        12, ['GGLGGLLGLGLG']),
    # 14-syl
    ('vasantatilaka',      14, ['GGLGLLLGLLGLGG']),
    # 15-syl
    ('malini',             15, ['LLLLLLGGLGGLGGG']),
    # 17-syl
    ('shikharini',         17, ['LGGGGGLLLLLGGGGLG']),
    ('mandakranta',        17, ['GGGGLLLLLGGLGGLGG']),
    ('harini',             17, ['LLLLLGGGGGLGLLGLG']),
    ('prithvi',            17, ['LGLLLGLGLLLGGLGGL']),
    # 19-syl
    ('shardulavikridita',  19, ['GGGLLGLGLLLGGGLGGLG']),
    # 21-syl
    ('sragdhara',          21, ['GGGGLGGGLLLLLLGGLGGLG']),
]

ANUSHTUBH_PADA = 8


def _match_pada(observed: str, template: str) -> bool:
    """Match observed pāda weight string against template. Last position is anceps."""
    if len(observed) != len(template):
        return False
    for i, (o, t) in enumerate(zip(observed, template)):
        if i == len(template) - 1:
            continue  # anceps
        if o != t:
            return False
    return True


def _match_meter(pattern: str, plen: int, templates: list[str]) -> bool:
    """Pattern is the full GL-string. Split into 4 pādas of plen, each must match any template."""
    if len(pattern) != 4 * plen:
        return False
    for i in range(4):
        pada = pattern[i*plen:(i+1)*plen]
        if not any(_match_pada(pada, t) for t in templates):
            return False
    return True


def mark_sutras(syllables: list[dict]) -> None:
    """Identify and mark sutra spans within the syllable stream.

    A sutra block in Anuvyakhyana is a Brahma Sutra recitation embedded in the verse,
    delimited by ॐ on both ends and wrapped by double-daṇḍas. After SLP1+syllabify:
      Type A — a single 'om' that is pāda-final (the ritual leading ॐ before a shloka).
      Type B — a span starting with 'om' at a phrase boundary and ending with 'om'
               that is pāda-final, where the span has ≥ 2 syllables.

    Mutates each syllable dict in place, adding key `is_sutra` (bool).
    """
    n = len(syllables)
    for s in syllables:
        s['is_sutra'] = False
    if n == 0:
        return
    # Type A: lone ritual 'om' that is pāda-final by itself.
    if syllables[0]['text'] == 'om' and syllables[0]['is_pada_final']:
        syllables[0]['is_sutra'] = True
    # Type B: scan for om-bracketed spans.
    i = 0
    while i < n:
        is_phrase_start = (i == 0 or syllables[i-1]['is_pada_final'])
        if (syllables[i]['text'] == 'om'
                and is_phrase_start
                and not syllables[i]['is_pada_final']):
            closed = False
            for j in range(i + 1, n):
                if syllables[j]['text'] == 'om' and syllables[j]['is_pada_final']:
                    for k in range(i, j + 1):
                        syllables[k]['is_sutra'] = True
                    i = j + 1
                    closed = True
                    break
            if closed:
                continue
        i += 1


def detect_meter(syllables: list[dict]) -> dict:
    """Return meter info: {name, pada_length, num_padas, padas}.

    Operates on the FULL syllable list — caller can mark_sutras first and pass
    only non-sutra syllables for cleanest detection.
    """
    pattern = ''.join(s['weight'] for s in syllables)
    n = len(pattern)

    for name, plen, templates in METERS:
        if _match_meter(pattern, plen, templates):
            return {
                'name': name,
                'pada_length': plen,
                'num_padas': 4,
                'padas': [pattern[i*plen:(i+1)*plen] for i in range(4)],
            }
    # Anuṣṭubh fallback (32 syl)
    if n == 4 * ANUSHTUBH_PADA:
        return {
            'name': 'anushtubh',
            'pada_length': ANUSHTUBH_PADA,
            'num_padas': 4,
            'padas': [pattern[i*8:(i+1)*8] for i in range(4)],
        }
    # Half-shloka anuṣṭubh (16 syl, 2 pādas)
    if n == 2 * ANUSHTUBH_PADA:
        return {
            'name': 'anushtubh_half',
            'pada_length': ANUSHTUBH_PADA,
            'num_padas': 2,
            'padas': [pattern[i*8:(i+1)*8] for i in range(2)],
        }
    return {
        'name': 'unknown',
        'pada_length': None,
        'num_padas': None,
        'padas': [pattern],
    }


def assign_pada_positions(syllables: list[dict], meter: dict) -> None:
    """Assign pada_index / pos_in_pada / pos_in_pada_norm to each syllable.

    Syllables marked `is_sutra=True` get pada_index = -1 and skip the counter.
    Non-sutra syllables get sequential pāda positions based on meter['pada_length'].
    """
    plen = meter['pada_length']
    metric_idx = 0  # counter over non-sutra syllables only
    for s in syllables:
        if s.get('is_sutra'):
            s['pada_index'] = -1
            s['pos_in_pada'] = -1
            s['pos_in_pada_norm'] = -1.0
            continue
        if plen is None:
            # Unknown meter — pretend single pāda of full length
            s['pada_index'] = 0
            s['pos_in_pada'] = metric_idx
            s['pos_in_pada_norm'] = 0.0
        else:
            s['pada_index'] = metric_idx // plen
            s['pos_in_pada'] = metric_idx % plen
            s['pos_in_pada_norm'] = (metric_idx % plen) / max(1, plen - 1)
        metric_idx += 1


def analyze(syllables: list[dict]) -> dict:
    """Full meter analysis: mark sutras, detect meter on shloka-only, assign positions.

    Mutates `syllables` in place. Returns the meter dict.
    """
    mark_sutras(syllables)
    shloka_syls = [s for s in syllables if not s['is_sutra']]
    meter = detect_meter(shloka_syls)
    assign_pada_positions(syllables, meter)
    return meter


if __name__ == '__main__':
    import argparse, csv, collections, sys
    sys.path.insert(0, '<REPO>/Final_Files/Scripts')
    from tts_normalize import normalize
    from tts_g2p import to_slp1
    from tts_syllabify import syllabify
    from tts_weight import tag_weights

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='metadata CSV path; if omitted, run a stats sweep')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--show', action='store_true', help='show per-row detection')
    args = ap.parse_args()

    def run(slp1):
        syls = syllabify(slp1)
        tag_weights(syls)
        m = analyze(syls)
        return syls, m

    if args.input and args.show:
        with open(args.input) as f:
            for i, row in enumerate(csv.DictReader(f, delimiter='|')):
                if i >= args.limit: break
                slp1 = to_slp1(normalize(row['text']))
                syls, m = run(slp1)
                nsutra = sum(1 for s in syls if s['is_sutra'])
                print(f'#{row["shloka_no"]}  {len(syls)} syl ({nsutra} sutra)  -> {m["name"]}')
                for j, pada in enumerate(m['padas']):
                    print(f'  pāda {j}: {pada}')
                print()
        sys.exit(0)

    # Sweep stats across all CSVs
    counts = collections.Counter()
    by_count = collections.defaultdict(collections.Counter)
    sutra_rows = 0
    for fn in ['metadata.csv','metadata_A2.csv','metadata_A3.csv','metadata_A4.csv']:
        with open(fn) as f:
            for row in csv.DictReader(f, delimiter='|'):
                slp1 = to_slp1(normalize(row['text']))
                syls = syllabify(slp1)
                tag_weights(syls)
                m = analyze(syls)
                counts[m['name']] += 1
                shloka_n = sum(1 for s in syls if not s['is_sutra'])
                by_count[shloka_n][m['name']] += 1
                if any(s['is_sutra'] for s in syls):
                    sutra_rows += 1
    print(f'Rows with sutra/ritual-om: {sutra_rows}')
    print('Meter distribution:')
    for name, c in counts.most_common():
        print(f'  {name:24s} {c}')
    print('\nUnknown-meter syllable-count breakdown (top 12):')
    unknowns = [(n, by_count[n].get('unknown', 0), sum(by_count[n].values())) for n in by_count]
    unknowns.sort(key=lambda x: -x[1])
    for n, u, total in unknowns[:12]:
        if u == 0: continue
        print(f'  {n} syl: {u} unknown out of {total} rows')
