#!/usr/bin/env python3
"""Convert Conformer alignment JSON files → per-clip .npy attention-frame
duration arrays matching the format of MFA's .npy outputs.

Shape: (len(slp1) + 2,) — leading PAD + per-SLP1-phoneme durations + trailing PAD.
Frame rate: 40 fps attention grid.

Algorithm:
1. Load Conformer JSON: subwords with dur_frames_40fps and start/end frames.
2. Distribute each subword's duration evenly across its Devanagari chars
   (after stripping ▁ word-start marker).
3. Walk aksharas of the Devanagari text (parse_aksharas from compare_aligners).
4. For each akshara: sum its constituent char durations to get akshara duration.
5. G2P the akshara → SLP1 phonemes; distribute akshara duration evenly across
   the SLP1 phonemes.
6. Add leading-pad-attention frames (8) + trailing-pad frames (computed from
   total audio duration vs SLP1 frame sum) to match MFA parser convention.

Output: <PROD>/sanskrit-tts/data/styletts2_data/conformer_durations/*.npy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Reuse text pipeline + akshara parser
ST2 = Path('<PROD>/sanskrit-tts/model/StyleTTS2')
SCRIPTS = Path('<PROD>/sanskrit-tts/scripts')
sys.path.insert(0, str(ST2))
sys.path.insert(0, str(SCRIPTS))
from tts_normalize import normalize as deva_normalize  # noqa: E402
from tts_g2p import to_slp1  # noqa: E402
from compare_aligners import parse_aksharas  # noqa: E402


ROOT = Path('<PROD>/sanskrit-tts')
DIR_CONFORMER_JSON = ROOT / 'data' / 'styletts2_data' / 'conformer_alignments'
DIR_OUT_NPY = ROOT / 'data' / 'styletts2_data' / 'conformer_durations'
DIR_OUT_NPY.mkdir(parents=True, exist_ok=True)

# Reference MFA durations — we use these to anchor TOTAL length per clip,
# preserving mel-duration consistency. Conformer contributes only the
# RELATIVE per-phoneme distribution (tighter consonants/vowels).
DIR_MFA = ROOT / 'data' / 'styletts2_data' / 'mfa_durations'


def parse_one(json_path: Path):
    data = json.loads(json_path.read_text())
    deva = data['devanagari']
    subwords = data['subwords']

    # 1. Build per-Devanagari-char duration list by distributing subword durations
    char_durs = []  # (char, dur_frames_40fps)
    for sw in subwords:
        text = sw['subword'].replace('▁', '')
        n_chars = len(text) if text else 1
        per_char = sw['dur_frames_40fps'] / n_chars
        for ch in text:
            char_durs.append((ch, per_char))

    # 2. Walk aksharas; for each, sum char durs and G2P to SLP1
    aksharas = parse_aksharas(deva)
    slp1_full = to_slp1(deva_normalize(deva))

    # Char-by-char position tracking: how many DEVA chars consumed by Conformer
    # vs how many we're walking through aksharas
    cc_idx = 0
    per_slp1_dur = []  # one entry per SLP1 char (excluding PADs)

    for ak in aksharas:
        # Count Conformer chars matching this akshara (skip pure-space aksharas;
        # Conformer often doesn't emit chars for whitespace).
        ak_text_no_space = ak.replace(' ', '')
        ak_dur_total = 0.0
        if ak_text_no_space:
            consumed = 0
            while consumed < len(ak_text_no_space) and cc_idx < len(char_durs):
                _ch, dur = char_durs[cc_idx]
                ak_dur_total += dur
                cc_idx += 1
                consumed += 1

        # G2P this akshara to learn its SLP1 expansion
        ak_slp1 = to_slp1(deva_normalize(ak))
        n_slp1 = len(ak_slp1)
        if n_slp1 == 0:
            continue  # punctuation that G2P drops

        # Distribute akshara's duration evenly across its SLP1 phonemes
        if ak_dur_total < n_slp1:
            ak_dur_total = float(n_slp1)  # at least 1 frame per phoneme
        per_phoneme = ak_dur_total / n_slp1
        for _ in range(n_slp1):
            per_slp1_dur.append(per_phoneme)

    # 3. Pad/truncate per_slp1_dur to match SLP1 length
    n_slp1_total = len(slp1_full)
    if len(per_slp1_dur) != n_slp1_total:
        if len(per_slp1_dur) < n_slp1_total:
            per_slp1_dur += [2.0] * (n_slp1_total - len(per_slp1_dur))
        else:
            per_slp1_dur = per_slp1_dur[:n_slp1_total]

    # 4. Use Conformer's per-phoneme durations EXACTLY (no rescaling).
    # The CTC-blank gap between conformer_phoneme_sum and audio total is
    # absorbed into the trailing PAD (silence) — NOT multiplied across
    # per-phoneme durations. This preserves Conformer's acoustic fidelity
    # while keeping total length aligned with mel/audio.
    mfa_path = DIR_MFA / json_path.with_suffix('.npy').name
    if not mfa_path.exists():
        raise FileNotFoundError(f'MFA reference missing: {mfa_path}')
    mfa = np.load(mfa_path)
    if len(mfa) != n_slp1_total + 2:
        raise ValueError(f'MFA len {len(mfa)} != slp1+2 = {n_slp1_total + 2}')

    leading_pad = int(mfa[0])
    audio_total_attn = int(mfa.sum())  # use MFA total as audio-length anchor

    # 5. Cumulative-endpoint rounding (preserves Conformer per-phoneme exactly)
    cum = np.cumsum(per_slp1_dur)
    int_durs = np.zeros(n_slp1_total, dtype=np.int64)
    prev = 0
    for i, c in enumerate(cum):
        end = int(round(c))
        int_durs[i] = max(1, end - prev)
        prev = end

    # 6. Trailing PAD absorbs the silence gap: gap = audio_total - leading_pad - phoneme_sum
    phoneme_sum = int(int_durs.sum())
    trailing_pad = audio_total_attn - leading_pad - phoneme_sum
    if trailing_pad < 1:
        # Conformer overshot (rare); just clip to minimum + adjust last phoneme
        trailing_pad = 4
        overshoot = phoneme_sum - (audio_total_attn - leading_pad - trailing_pad)
        if overshoot > 0 and n_slp1_total > 0:
            int_durs[-1] = max(1, int_durs[-1] - overshoot)

    # 7. Assemble: leading PAD (from MFA) + Conformer per-phoneme + silence trailing PAD
    out = np.concatenate([
        np.array([leading_pad], dtype=np.int64),
        int_durs,
        np.array([trailing_pad], dtype=np.int64),
    ])
    return out, n_slp1_total, deva, slp1_full


def main():
    json_files = sorted(DIR_CONFORMER_JSON.glob('*.json'))
    if not json_files:
        print(f'No JSON files in {DIR_CONFORMER_JSON}')
        return
    print(f'Parsing {len(json_files)} clips → {DIR_OUT_NPY}')

    n_ok, n_fail = 0, 0
    failures = []
    for fp in json_files:
        base = fp.stem
        out_path = DIR_OUT_NPY / f'{base}.npy'
        try:
            durs, n_slp1, deva, slp1 = parse_one(fp)
            np.save(out_path, durs)
            n_ok += 1
            total_frames = int(durs.sum())
            total_s = total_frames / 40.0
            if n_ok <= 5 or n_ok % 200 == 0:
                print(f'  [ok] {base}  SLP1_len={n_slp1}  frames={total_frames}  ({total_s:.2f}s)')
        except Exception as e:
            n_fail += 1
            failures.append((base, str(e)))
            print(f'  [FAIL] {base}: {type(e).__name__}: {e}')

    print(f'\nDone: {n_ok} ok, {n_fail} failed')
    if failures:
        print('Failures:')
        for b, e in failures[:10]:
            print(f'  {b}: {e}')


if __name__ == '__main__':
    main()
