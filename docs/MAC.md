# Running Vāgdhenu on a Mac

The upstream project targets a CUDA 12.1 GPU. This branch makes the backend selectable so the
same code runs on Apple Silicon via Metal (MPS), or on CPU, with no change to CUDA behavior.

**Apple Silicon only.** PyTorch has published no macOS x86_64 wheel since 2.2.2 and there is no
Metal backend for Intel Macs. If `uname -m` says `x86_64`, stop here and use a CUDA box.

## What changed

`src/device.py` picks the backend in order `cuda` → `mps` → `cpu`, and `VAGDHENU_DEVICE` overrides
it. Every entry point imports it before `torch`, because PyTorch reads `PYTORCH_ENABLE_MPS_FALLBACK`
once, when the Metal dispatch table registers at `import torch`. Setting it later does nothing.

That fallback matters here: the mel front-end and the vocos ISTFT head both call FFT kernels whose
Metal coverage varies by torch build. With the fallback armed, a missing kernel runs on CPU instead
of raising. Stage 2 of the self-test tells you exactly which kernels that applies to.

On the GPU boxes the detector returns the string `cuda`, which is exactly what was hardcoded
before, so production renders are unaffected.

## Setup

```bash
git clone git@github.com:shyamsfo/vagdhenu.git && cd vagdhenu
git checkout mac-port

python3.10 -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh              # torch arm64, deps, BigVGAN clone, weights -> models/
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"    # put this in your shell rc
brew install ffmpeg                # pydub decodes the reference clips
```

`scripts/setup.sh` branches on `uname`, so the same command works on Linux and macOS. It refuses
Intel Macs with an explanation rather than failing halfway through a pip install.

## Run order

Work through these in order. Do not skip to a full render, because a failure there is hard to
attribute, and stages 1 to 3 need no model weights at all.

```bash
# 1. cheap stages: backend, ffmpeg, BigVGAN, weights present, text frontend
python scripts/selftest.py --stages 1 2 3

# 2. full run at the serving config (roughly twice as fast as the locked nfe 64)
python scripts/selftest.py --nfe 32

# 3. the real thing: render the sample verse through the production path
python src/render.py --shard examples/sample_shard.json \
  --results /tmp/res.json --outdir out --nfe 32

# 4. compare against a render from the GPU box
scp gpubox:out/selftest_cuda_nfe32.wav .
python scripts/selftest.py --nfe 32 --compare selftest_cuda_nfe32.wav

# 5. interactive, if you want a UI
python demo/server.py
```

Stage 2 is the interesting one. It re-runs the five hot kernels in a child process with the
fallback disabled, so anything Metal lacks raises instead of quietly running on CPU. You get a
per-kernel verdict of `native` or `CPU-FALLBACK`.

`demo/app.py` stays GPU-only; it imports Hugging Face ZeroGPU's `spaces` package. Use
`demo/server.py` locally.

## What to expect

| Configuration | Real-time factor | Notes |
|---|---|---|
| A6000, nfe 64 (locked) | 1.24 | production, per tech report |
| A6000, nfe 32 (serving) | 0.63 | production, per tech report |
| 16-core x86 CPU, nfe 64 | 22.7 | measured — one anuṣṭubh, 3m33s wall for 9.39 s audio |
| Apple Metal, nfe 32 | unmeasured | anuṣṭubh renders correctly, 5/5 hot kernels native, seed 60 no retries |

The tech report's CPU claim of "~22 min per hemistich" (RTF ~140) was presumably measured
single-threaded or on much older hardware; treat 22.7 as the number to beat, not 140. Real-time
factor is compute seconds per audio second, so lower is better and anything above 1 is slower than
listening to it. Peak inference memory is 2.5 GB, so any Apple Silicon Mac has room. Start with
`--nfe 32` and a single verse before attempting a batch.

## Known differences from the GPU output

Renders will **not** be bit-identical to CUDA output. Different backend, different kernel ordering.
The md5-identical guarantee in the tech report holds on CUDA only.

This has a practical consequence. The renderer retries a clip up to four times when its
root-mean-square energy falls below threshold, advancing the seed each attempt. Seeds vetted by ear
on the A6000 may land differently on Metal. Listen before trusting a batch.

Stage 6 of the self-test compares two wavs by mel distance rather than by waveform equality, which
is the only comparison that means anything across backends.

## Verifying the port did not break CUDA

Run this on a GPU box. If the checksums match, the port is provably safe for production.

```bash
git stash
python src/render.py --shard examples/sample_shard.json --results /tmp/a.json --outdir out_before
git stash pop
python src/render.py --shard examples/sample_shard.json --results /tmp/b.json --outdir out_after
md5sum out_before/*.wav out_after/*.wav
```

## Status

The port has been executed end-to-end on Apple Silicon (M3 Max, Metal at nfe 32) and on a 16-core
x86 Linux CPU box (nfe 64). In both cases anuṣṭubh rendered correctly, seed 60 first-try, no
retries. Stage 2 of the self-test reports all five hot kernels running native on Metal on the
current torch build; the `PYTORCH_ENABLE_MPS_FALLBACK` armor remains, but nothing has needed to
fall back in practice.

The CUDA no-regression check on the T4 (g4dn.4xlarge) confirmed that `main` and `mac-port` both
render `examples/sample_shard.json` cleanly at nfe 64 with the same duration and no retries.
Byte-level md5s differ between branches on the T4 — but byte-level md5s also differ between two
consecutive same-code runs, so that's cuDNN algorithm nondeterminism on Turing (sm_75), not a
regression. **The A6000 md5-identity check from the "Verifying the port did not break CUDA"
section above has not been run** — that is the definitive test and remains open pending access to
the production hardware.

Still open: Metal wall-time RTF (unmeasured), Metal coverage for meters other than anuṣṭubh
(longer meters with more solver-step accumulation may drift below the RMS energy floor and force
seed advances).
