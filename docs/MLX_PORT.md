# MLX backend — measurements

All numbers: Mac Studio, Apple M5 Max (18-core CPU, 128 GB), MLX 0.32.4, macOS 27, nfe 64 · cfg 3.0 · speed 0.90
unless stated. Scripts: `tools/parity_mlx.py` (correctness), and in the dev experiment `profile_mlx.py` /
`render_mlx.py` (timings). "RTF" = render seconds ÷ audio seconds (lower is better).

## 1. End-to-end: verse → wav (`Renderer.render_one`, seed 60)

| verse (pādas) | audio | A6000 CUDA (tech report RTF 1.24) | torch 2.5 MPS on this Mac* | MLX, first working port | **MLX, optimised** |
|---|---|---|---|---|---|
| Vasudevasutaṃ (2) | 9.67 s | ≈ 12.0 s | 32.2 s (RTF 3.37) | 3.78 s (0.39) | **2.80 s (0.29)** |
| Śuklāmbaradharaṃ (2) | 9.49 s | ≈ 11.8 s | 31.7 s (3.38) | 3.73 s (0.39) | **2.78 s (0.29)** |
| Mālinī tongue-twister (4) | 16.89 s | ≈ 20.9 s | 68.7 s (4.08) | 8.33 s (0.49) | **6.05 s (0.36)** |
| Karāgre vasate (2) | 9.13 s | ≈ 11.3 s | 31.9 s (3.48) | 3.66 s (0.40) | **2.75 s (0.30)** |
| peak memory | | 2.5 GB (report) | — | 3.3 GB | 3.8 GB |

\* upstream code unchanged, `device="mps"`; BigVGAN's grouped `conv_transpose1d` is unsupported on MPS, so the vocoder ran on CPU.
A6000 seconds are derived from the report's RTF 1.24. Model load (safetensors, mmap): 0.1 s; warm-up render: 1.5 s.

## 2. Where the time went, and what each optimisation bought

DiT: dim 1024 / depth 22 / heads 16, N = 932 frames (4.7 s reference + 5.2 s generated), CFG as a batch of 2.

| stage | before | after | change |
|---|---|---|---|
| DiT step (eager MLX) | 24.0 ms | — | baseline; ~35 TFLOPS effective |
| DiT step under `mx.compile` | 24.0 ms | **20.1 ms** | −16 %; a new (batch, length) costs one 35 ms trace, then cached |
| 64-step sample, 1 pāda | 1.54 s | **1.29 s** | constants (text embed, cond projection, AdaLN mods) hoisted out of the loop + compile |
| 64-step sample, 2 pādas batched | 1.47 s/pāda | 1.24 s/pāda | batching pādas keeps one GPU job per verse; the GPU is already saturated at B=1 |
| BigVGAN, 5.2 s of audio (fp32) | 351 ms | **174 ms** | fused Metal kernel for the anti-aliased SnakeBeta |
| BigVGAN, 4.7 s of audio (fp32) | 309 ms | **122 ms** | same |
| one AA-SnakeBeta, C=768 T=1956 | 0.32 ms | **0.06 ms** | 108 of these per vocoder call |
| one AA-SnakeBeta, C=24 T=125 184 | 0.25 ms | **0.09 ms** | |

BigVGAN breakdown before the kernel (fp32, 489 frames): transposed-conv upsamplers 10 ms, resblocks 336 ms, of which the
anti-aliased activations ≈ 281 ms (84 %): each is a 2× polyphase upsample → sin² → 12-tap downsample, i.e. ~10 passes over
memory as separate ops. The kernel does it in one pass per output sample (72 MACs + 12 `precise::sin`) and matches the
op-by-op version to 2e-7.

Rejected: half-precision vocoder. bf16 BigVGAN: 271 ms but max error 0.69 of full scale (mean 8e-3); fp16: 273 ms, max 0.19
(mean 1.6e-3). The DiT is fine in bf16 (see §3); BigVGAN stays fp32.

## 3. Correctness vs the torch reference (IndicF5 13f7c4d + NVIDIA BigVGAN, torch 2.5.1 CPU)

`tools/parity_mlx.py`. Relative error = max |Δ| / max |reference| (mean in brackets).

| check | MLX CPU stream (exact fp32) | MLX GPU fp32 | MLX GPU bf16 DiT |
|---|---|---|---|
| tokenizer vs `convert_char_to_pinyin` (54 bank/verse strings) | identical | identical | identical |
| `preprocess_ref_audio_text` waveform | 0 (bit-exact) | 0 | 0 |
| vocos log-mel (3 reference clips) | 1.0e-5 – 5.5e-5 | same (numpy) | same |
| DiT step, conditional / null | 1.6e-6 / 3.3e-6 | 2.3e-3 / 5.7e-3 | — |
| batched (padded) item == solo item | 1.1e-6 / 1.8e-6 | 5e-7 / 4.6e-4 | — |
| BigVGAN on a real mel (15 360 samples) | 9.6e-6 | 3.8e-2 (1.9e-3) | n/a |
| full `CFM.sample`, 6 steps, same noise | 2.8e-5 | — | — |
| full `CFM.sample`, 16 steps, same noise | — | 2.6e-2 (9.1e-4) | 9.4e-2 (3.2e-3) |
| 17 unit tests (`tests/test_vagdhenu_mlx.py`) | pass | pass | — |

The GPU column is the hardware, not the port: a bare `mx.matmul` of fp32 random matrices (K = 512…2048) has relative error
7.9e-4 on this GPU versus 9.5e-7 on the CPU stream and in numpy (bf16: 3.6e-3, fp16: 4.5e-4) — Apple's fp32 GPU matmul
runs at reduced (TF32-like) precision, and the same shows for convolutions. Padded-vs-solo differs on the GPU only
because the matmul tiles differ (CPU: 1.8e-6).

## 4. A/B against upstream renders (different noise draws, so not sample-identical)

| verse | duration torch / MLX | loudness torch / MLX | mean \|Δ time-averaged log-mel\| |
|---|---|---|---|
| Vasudevasutaṃ | 9.55 s / 9.67 s | −18.1 / −17.8 dBFS | 0.082 nats |
| Śuklāmbaradharaṃ | 9.39 s / 9.49 s | −17.9 / −17.7 | 0.096 |
| Mālinī | 16.84 s / 16.89 s | −18.9 / −18.8 | 0.088 |
| Karāgre | 9.17 s / 9.13 s | −17.4 / −17.3 | 0.084 |

(0.09 nats ≈ 0.8 dB of spectral-envelope difference — the spread you get between two seeds of the same pipeline.)
