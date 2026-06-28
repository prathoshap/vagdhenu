#!/usr/bin/env python3
"""Phonemize a directory of Sanskrit text into SLP1 sequences for MLM pretraining.

Handles multiple input formats:
  - Devanagari (e.g., our local corpus)
  - IAST (e.g., much of GRETIL)
  - ITRANS, HK, SLP1 (less common — best-effort)

Pipeline per document:
  detect_script → transliterate-to-Devanagari → tts_normalize → tts_g2p → SLP1

Output: single corpus file, one document per line. Documents over max_chars are split.
Quality filter rejects:
  - documents with < min_chars
  - documents with high non-Devanagari proportion after normalization
  - documents with low character entropy (likely OCR garbage)
"""

import argparse
import math
import os
import re
import sys
from collections import Counter
from glob import glob

sys.path.insert(0, '<PROD>/sanskrit-tts/scripts')
from tts_normalize import normalize as deva_normalize
from tts_g2p import to_slp1

from indic_transliteration import sanscript, detect


DEVA_RE = re.compile(r'[ऀ-ॿ]')
IAST_DIACRITIC_RE = re.compile(r'[āīūṛṝḷḹṅñṭḍṇśṣṃḥ]')


def detect_script(text: str) -> str:
    """Return 'devanagari', 'iast', or 'unknown'."""
    deva_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    if deva_count > len(text) * 0.3:
        return 'devanagari'
    if IAST_DIACRITIC_RE.search(text):
        return 'iast'
    try:
        d = detect.detect(text[:500])
        return d.lower() if d else 'unknown'
    except Exception:
        return 'unknown'


def to_devanagari(text: str) -> str:
    """Transliterate to Devanagari if needed."""
    s = detect_script(text)
    if s == 'devanagari':
        return text
    if s == 'iast':
        return sanscript.transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)
    # Best-effort fallback (often gets ITRANS/HK right)
    try:
        return sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)


def quality_ok(deva_text: str, min_chars=200, min_entropy=3.0, max_non_deva_frac=0.30) -> bool:
    """Reject low-quality docs (mostly for OCR'd Sangraha)."""
    if len(deva_text) < min_chars:
        return False
    deva_chars = sum(1 for c in deva_text if 0x0900 <= ord(c) <= 0x097F or c.isspace())
    if deva_chars / len(deva_text) < (1.0 - max_non_deva_frac):
        return False
    if shannon_entropy(deva_text) < min_entropy:
        return False
    return True


def chunk_text(text: str, max_chars: int = 8000) -> list[str]:
    """Split very long documents into sentence-like chunks for MLM training."""
    if len(text) <= max_chars:
        return [text]
    # Split on daṇḍa, double-daṇḍa, or paragraph breaks
    parts = re.split(r'(?<=[।॥])\s+|\n{2,}', text)
    chunks = []
    cur = []
    cur_len = 0
    for p in parts:
        if cur_len + len(p) > max_chars and cur:
            chunks.append(' '.join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p)
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def collect_files(root: str) -> list[str]:
    files = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith('.txt'):
                files.append(os.path.join(r, fn))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus_root', default='<PROD>/sanskrit-tts/corpora',
                    help='root dir containing subdirs per source')
    ap.add_argument('--out_path', default='<PROD>/sanskrit-tts/corpora/sanskrit.slp1.txt')
    ap.add_argument('--max_chars_per_chunk', type=int, default=8000,
                    help='Devanagari char count; ~1800 SLP1 tokens (well below 512 model max)')
    ap.add_argument('--min_chars', type=int, default=200)
    ap.add_argument('--no_filter', action='store_true', help='disable quality filter (use for clean corpora)')
    args = ap.parse_args()

    files = collect_files(args.corpus_root)
    print(f'Found {len(files)} text files under {args.corpus_root}')

    stats = Counter()
    total_in_chars = 0
    total_out_phonemes = 0
    with open(args.out_path, 'w', encoding='utf-8') as fout:
        for i, p in enumerate(files):
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except Exception as e:
                stats['read_error'] += 1
                continue
            if not text or len(text) < args.min_chars:
                stats['too_short'] += 1
                continue

            deva = to_devanagari(text)
            if not args.no_filter and not quality_ok(deva, min_chars=args.min_chars):
                stats['quality_filter'] += 1
                continue

            chunks = chunk_text(deva, max_chars=args.max_chars_per_chunk)
            for ch in chunks:
                normalized = deva_normalize(ch)
                slp1 = to_slp1(normalized)
                slp1 = slp1.strip()
                if len(slp1) < 50:
                    stats['empty_after_g2p'] += 1
                    continue
                fout.write(slp1 + '\n')
                total_in_chars += len(ch)
                total_out_phonemes += len(slp1)
                stats['chunks_emitted'] += 1
            stats['files_processed'] += 1
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(files)} files; {stats["chunks_emitted"]} chunks; '
                      f'{total_out_phonemes/1e6:.1f}M phoneme tokens')

    print('\n=== Done ===')
    print(f'Output: {args.out_path}')
    print(f'Total input chars: {total_in_chars/1e6:.1f}M')
    print(f'Total SLP1 phonemes: {total_out_phonemes/1e6:.1f}M')
    print('Stats:')
    for k, v in stats.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
