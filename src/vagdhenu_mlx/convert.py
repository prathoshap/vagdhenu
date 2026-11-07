"""Convert the PyTorch Vāgdhenu checkpoints to plain safetensors for the MLX backend.

    python -m vagdhenu_mlx.convert --voice models/voice_steer_ema_2026-06-17.pt \
        --vocoder models/voc_bigvgan_EMA_2026-06-11.pth --out models/mlx

This is the only step that needs torch. Checkpoints are opened with ``weights_only=True`` (no
arbitrary pickle code runs). The output holds:
  dit.safetensors      EMA DiT weights (``ema_model.`` prefix stripped), float32, torch layout
  bigvgan.safetensors  BigVGAN-v2 generator with weight-norm folded in (w = g * v / ||v||), float32
Layout conversion to MLX (channels-last convs) happens at load time, so these files stay
framework-neutral and can be inspected with any safetensors reader.
"""
import argparse, os, shutil

import numpy as np


def _np(t):
    return t.detach().float().cpu().numpy()


def convert_dit(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("ema_model_state_dict", ck)
    out = {}
    for k, v in sd.items():
        if k in ("initted", "step") or not hasattr(v, "shape"):
            continue
        k = k.replace("ema_model.", "")
        if k.startswith("mel_spec.") or k.endswith("rotary_embed.inv_freq"):
            continue  # recomputed analytically
        out[k] = _np(v)
    return out


def convert_bigvgan(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("model", ck.get("generator", ck))
    out = {}
    for k, v in sd.items():
        if k.endswith("weight_v"):
            base = k[: -len("weight_v")]
            g = sd[base + "weight_g"].float()
            v = v.float()
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))  # weight_norm dim=0
            out[base + "weight"] = _np(g * v / norm)
        elif k.endswith("weight_g"):
            continue
        else:
            out[k] = _np(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="DiT EMA checkpoint (.pt)")
    ap.add_argument("--vocoder", required=True, help="BigVGAN generator checkpoint (.pth)")
    ap.add_argument("--vocab", default=None, help="vocab.txt to copy beside the weights")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.numpy import save_file
    os.makedirs(a.out, exist_ok=True)
    dit = convert_dit(a.voice)
    save_file(dit, os.path.join(a.out, "dit.safetensors"),
              metadata={"source": os.path.basename(a.voice), "format": "vagdhenu-dit-v1"})
    voc = convert_bigvgan(a.vocoder)
    save_file(voc, os.path.join(a.out, "bigvgan.safetensors"),
              metadata={"source": os.path.basename(a.vocoder), "format": "bigvgan-v2-24k-100band-256x"})
    if a.vocab:
        shutil.copy(a.vocab, os.path.join(a.out, "vocab.txt"))
    n = lambda d: sum(int(np.prod(x.shape)) for x in d.values())
    print(f"dit: {len(dit)} tensors, {n(dit)/1e6:.1f}M params | bigvgan: {len(voc)} tensors, {n(voc)/1e6:.1f}M params -> {a.out}")


if __name__ == "__main__":
    main()
