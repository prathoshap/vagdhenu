#!/usr/bin/env python3
"""Build the combined StyleTTS2 training dataset from the four Anuvyakhyana metadata CSVs.

Outputs (in output/TTS/):
  training_master.csv   — flat format: path|phonemes|speaker|... (one row per clip)
  training_master.jsonl — one JSON per row with full per-syllable feature stream

Speaker is `pilot_recitereshacharya` for all rows (single chanter for Anuvyakhyana).
tts_exclude defaults to False for A1/A2 (no column) and is preserved for A3/A4.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, '<REPO>/Final_Files/Scripts')
from tts_normalize import normalize
from tts_g2p import to_slp1
from tts_syllabify import syllabify
from tts_weight import tag_weights
from tts_meter import analyze


SOURCE_CSVS = [
    ('metadata.csv',    1),
    ('metadata_A2.csv', 2),
    ('metadata_A3.csv', 3),
    ('metadata_A4.csv', 4),
]

SPEAKER = 'pilot_recitereshacharya'

FLAT_COLUMNS = [
    'path', 'phonemes', 'speaker',
    'duration_s', 'adhyaya', 'shloka_no', 'block_id',
    'vrtta', 'pada_length', 'num_syllables', 'num_sutra_syllables',
    'tts_exclude',
]


def build_row(src_row: dict, adhyaya: int) -> tuple[dict, dict]:
    """Process one source CSV row. Returns (flat_dict, jsonl_dict)."""
    text_orig = src_row['text']
    text_norm = normalize(text_orig)
    slp1 = to_slp1(text_norm)
    syls = syllabify(slp1)
    tag_weights(syls)
    meter = analyze(syls)

    nsutra = sum(1 for s in syls if s['is_sutra'])
    excl = src_row.get('tts_exclude', '').strip().lower() in ('true', '1', 'yes')

    flat = {
        'path': src_row['filename'],
        'phonemes': slp1,
        'speaker': SPEAKER,
        'duration_s': float(src_row['duration_s']),
        'adhyaya': adhyaya,
        'shloka_no': src_row['shloka_no'],
        'block_id': src_row['block_id'],
        'vrtta': meter['name'],
        'pada_length': meter['pada_length'] if meter['pada_length'] is not None else '',
        'num_syllables': len(syls),
        'num_sutra_syllables': nsutra,
        'tts_exclude': 'true' if excl else 'false',
    }
    full = {
        **flat,
        'text_original': text_orig,
        'text_normalized': text_norm,
        'meter': {
            'name': meter['name'],
            'pada_length': meter['pada_length'],
            'num_padas': meter['num_padas'],
            'padas': meter['padas'],
        },
        'syllables': [
            {
                'text': s['text'],
                'vowel': s['vowel'],
                'weight': s['weight'],
                'pada_index': s['pada_index'],
                'pos_in_pada': s['pos_in_pada'],
                'pos_in_pada_norm': s['pos_in_pada_norm'],
                'is_sutra': s['is_sutra'],
                'is_pada_final': s['is_pada_final'],
                'is_word_final': s['is_word_final'],
            }
            for s in syls
        ],
    }
    return flat, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_dir', default='<REPO>/Final_Files/Anuvyakhyana/output/TTS')
    ap.add_argument('--out_csv', default=None, help='default: <source_dir>/training_master.csv')
    ap.add_argument('--out_jsonl', default=None, help='default: <source_dir>/training_master.jsonl')
    args = ap.parse_args()

    out_csv = args.out_csv or os.path.join(args.source_dir, 'training_master.csv')
    out_jsonl = args.out_jsonl or os.path.join(args.source_dir, 'training_master.jsonl')

    total = 0
    excluded = 0
    meter_counts = {}
    with open(out_csv, 'w', newline='') as fcsv, open(out_jsonl, 'w') as fj:
        w = csv.DictWriter(fcsv, fieldnames=FLAT_COLUMNS, delimiter='|')
        w.writeheader()
        for fname, adhyaya in SOURCE_CSVS:
            path = os.path.join(args.source_dir, fname)
            with open(path) as f:
                for row in csv.DictReader(f, delimiter='|'):
                    flat, full = build_row(row, adhyaya)
                    w.writerow(flat)
                    fj.write(json.dumps(full, ensure_ascii=False) + '\n')
                    total += 1
                    if flat['tts_exclude'] == 'true':
                        excluded += 1
                    meter_counts[flat['vrtta']] = meter_counts.get(flat['vrtta'], 0) + 1
    print(f'Wrote {total} rows to {out_csv}')
    print(f'Wrote {total} rows to {out_jsonl}')
    print(f'tts_exclude=true: {excluded}')
    print('\nMeter counts:')
    for m, c in sorted(meter_counts.items(), key=lambda x: -x[1]):
        print(f'  {m:24s} {c}')


if __name__ == '__main__':
    main()
