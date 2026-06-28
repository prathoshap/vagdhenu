#!/usr/bin/env python3
"""Forced alignment using the pre-installed AI4Bharat IndicConformer (Sanskrit).

Uses the existing NeMo model already cached on <gpu-host>:
  ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large

Procedure:
1. Load model, switch to CTC decoding strategy to expose per-frame log-probs.
2. For each clip: forward audio → CTC log-probs (vocab 257, frame rate 25 fps).
3. Tokenize the reference Devanagari text via the model's Sanskrit BPE.
4. Run torchaudio.functional.forced_align(emissions, targets, blank=256) →
   per-frame target index. Collapse to per-subword frame ranges.
5. Convert Conformer 25 fps → our attention 40 fps (×1.6) via cumulative-
   endpoint rounding (matches MFA parser convention).
6. Emit JSON per clip: subword + Devanagari char timings + 40-fps frame
   durations.

Run on server:
  CUDA_VISIBLE_DEVICES=0 \\
  <PROD>/miniconda3/envs/sarvamoola/bin/python \\
    <PROD>/sanskrit-tts/scripts/align_indic_conformer.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as TAF

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf


ROOT = Path('<PROD>/sanskrit-tts')
WAVS_DIR = ROOT / 'data' / 'wavs'
MASTER_JSONL = ROOT / 'data' / 'training_master.jsonl'
DUR_DIR_MFA = ROOT / 'data' / 'styletts2_data' / 'mfa_durations'
OUT_DIR = ROOT / 'data' / 'styletts2_data' / 'conformer_alignments'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ('<HOST>/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/'
              'ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large/'
              'c82246cc7136e7f1be8df3090e3a07d9/'
              'indicconformer_stt_sa_hybrid_rnnt_large.nemo')

CONFORMER_FPS = 25.0  # 40 ms per frame
ATTN_FPS = 40.0       # our attention grid (24 kHz / 600 hop after stride 2)
SR_TARGET = 16000     # IndicConformer expects 16 kHz


def load_model(device='cuda'):
    print(f'Loading model: {MODEL_PATH}')
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        MODEL_PATH, map_location=device)
    model.eval()
    # Switch to CTC decoding to expose per-frame alignments
    decoding_cfg = OmegaConf.to_container(model.cfg.aux_ctc.decoding, resolve=True)
    decoding_cfg['preserve_alignments'] = True
    decoding_cfg['compute_timestamps'] = True
    model.change_decoding_strategy(OmegaConf.create(decoding_cfg), decoder_type='ctc')
    return model


def get_ctc_log_probs(model, audio_path: str, device='cuda'):
    """Forward audio → CTC log-probs (T, V). Resamples to 16 kHz first."""
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR_TARGET:
        wav = torchaudio.functional.resample(wav, sr, SR_TARGET)
    wav = wav.to(device)
    length = torch.tensor([wav.size(1)], device=device)

    with torch.no_grad():
        # NeMo's forward: log_probs are at the CTC head's frame rate
        encoded, encoded_len = model.forward(
            input_signal=wav, input_signal_length=length)
        # encoded: (B, D, T_enc); ctc head projects to vocab
        log_probs = model.ctc_decoder(encoder_output=encoded)
        # log_probs: (B, T, V) — already log-probs
    return log_probs.squeeze(0).cpu(), int(encoded_len.item())


def tokenize_reference(model, text_deva: str):
    """Devanagari → Sanskrit BPE token ids using the model's tokenizer."""
    sa_tok = model.tokenizer.tokenizers_dict['sa']
    ids = sa_tok.text_to_ids(text_deva)
    return ids, sa_tok


def conformer_frame_to_attn_frame(conformer_idx):
    """25 fps → 40 fps via cumulative-endpoint scaling."""
    return int(round(conformer_idx * (ATTN_FPS / CONFORMER_FPS)))


def align_one(model, audio_path: str, text_deva: str, device='cuda'):
    log_probs, T = get_ctc_log_probs(model, audio_path, device=device)
    target_ids, sa_tok = tokenize_reference(model, text_deva)

    targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)
    emissions = log_probs.unsqueeze(0)  # (1, T, V)
    blank_id = log_probs.size(-1) - 1  # CTC blank is the last index (256 for vocab 257)

    # torchaudio.functional.forced_align returns (alignments, scores)
    # alignments: (B, T) — target index aligned at each frame (or -1 for blank)
    alignments, scores = TAF.forced_align(emissions, targets, blank=blank_id)
    alignments = alignments.squeeze(0).tolist()  # length T

    # Collapse aligned frames into per-target-token spans (token_idx, start_frame, end_frame)
    spans = []
    cur_tgt = None
    start = None
    for t, a in enumerate(alignments):
        if a == blank_id or a == -1:
            continue
        # In torchaudio.forced_align, the alignment IS the actual token id at each
        # non-blank frame (not target index). Token ids in the targets sequence;
        # but multiple consecutive frames with the same token_id belong to the
        # same target instance only if they're in run. We collapse by run.
        if cur_tgt is None or a != cur_tgt:
            if cur_tgt is not None:
                spans.append((cur_tgt, start, t - 1))
            cur_tgt = a
            start = t
    if cur_tgt is not None:
        spans.append((cur_tgt, start, len(alignments) - 1))

    # Map spans to subwords + Devanagari char timings
    subwords = []
    for tid, sf, ef in spans:
        try:
            sw = sa_tok.ids_to_tokens([tid])[0]
        except Exception:
            sw = f'[{tid}]'
        sf_attn = conformer_frame_to_attn_frame(sf)
        ef_attn = conformer_frame_to_attn_frame(ef + 1) - 1  # inclusive
        subwords.append({
            'subword': sw,
            'start_frame_25fps': sf,
            'end_frame_25fps': ef,
            'start_frame_40fps': sf_attn,
            'end_frame_40fps': ef_attn,
            'start_s': sf / CONFORMER_FPS,
            'end_s': (ef + 1) / CONFORMER_FPS,
        })

    # Per-subword duration in 40 fps (cumulative-endpoint to avoid drift)
    cum_end_40 = []
    for sw in subwords:
        cum_end_40.append(sw['end_frame_40fps'] + 1)
    prev = 0
    for sw, end in zip(subwords, cum_end_40):
        sw['dur_frames_40fps'] = max(1, end - prev)
        prev = end

    return subwords, log_probs.shape[0]


def select_clips(mode='pilot', n=30):
    """mode='pilot' returns 30 stratified clips; mode='full' returns all eligible."""
    records = [json.loads(l) for l in MASTER_JSONL.read_text().splitlines() if l.strip()]
    eligible = [r for r in records
                if (WAVS_DIR / r['path']).exists()
                and (DUR_DIR_MFA / f'{r["path"].rsplit(".wav",1)[0]}.npy').exists()]
    if mode == 'full':
        return sorted(eligible, key=lambda r: r['path'])

    # pilot mode: stratified
    anu = [r for r in eligible if r.get('vrtta') == 'anushtubh']
    rare = [r for r in eligible if r.get('vrtta') in ('vasantatilaka', 'indravajra', 'upajati', 'vamshastha')]
    SLP1_CONS = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh')
    conjunct_heavy = []
    for r in eligible:
        ph = r.get('phonemes', '')
        clusters = 0; i = 0
        while i < len(ph):
            if ph[i] in SLP1_CONS:
                j = i
                while j < len(ph) and ph[j] in SLP1_CONS:
                    j += 1
                if j - i >= 2:
                    clusters += 1
                i = j
            else:
                i += 1
        if clusters >= 5:
            conjunct_heavy.append(r)

    def spread(lst, k):
        if not lst:
            return []
        lst = sorted(lst, key=lambda r: r['path'])
        if len(lst) <= k:
            return lst
        idx = [int(i * (len(lst) - 1) / (k - 1)) for i in range(k)]
        return [lst[i] for i in idx]

    picks = spread(anu, 10) + spread(rare, 5) + spread(conjunct_heavy, 15)
    seen = set(); out = []
    for r in picks:
        if r['path'] not in seen:
            seen.add(r['path']); out.append(r)
        if len(out) >= n:
            break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'full'], default='pilot',
                    help='pilot = 30 stratified clips; full = all eligible clips')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  Mode: {args.mode}')
    model = load_model(device=device)

    clips = select_clips(mode=args.mode, n=30)
    print(f'\n{args.mode} clips: {len(clips)}')
    print('  meters:', Counter(c.get('vrtta', 'unknown') for c in clips))

    n_ok, n_fail = 0, 0
    for r in clips:
        base = r['path'].rsplit('.wav', 1)[0]
        out_path = OUT_DIR / f'{base}.json'
        if out_path.exists():
            print(f'  [skip] {base}')
            n_ok += 1; continue
        try:
            subwords, T = align_one(model, str(WAVS_DIR / r['path']),
                                     r['text_original'], device=device)
            total_s = subwords[-1]['end_s'] if subwords else 0.0
            out_path.write_text(json.dumps({
                'base': base,
                'devanagari': r['text_original'],
                'vrtta': r.get('vrtta'),
                'duration_s_master': r.get('duration_s'),
                'subwords': subwords,
                'n_conformer_frames': T,
                'total_dur_s': float(total_s),
                'model': 'indicconformer_stt_sa_hybrid_ctc_rnnt_large',
                'language': 'sa',
            }, indent=2, ensure_ascii=False))
            n_ok += 1
            print(f'  [ok]   {base}  ({len(subwords)} subwords, {total_s:.2f}s)')
        except Exception as e:
            n_fail += 1
            print(f'  [FAIL] {base}: {type(e).__name__}: {e}')

    print(f'\nDone: {n_ok} ok, {n_fail} failed')
    print(f'Outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
