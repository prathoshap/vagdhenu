"""Standalone BigVGAN vocoder (mel -> audio) for Matcha-TTS Sanskrit.
Reuses VITS2 Generator/MultiPeriodDiscriminator/losses; mel = Matcha's exact fn.
Warm-starts the generator body from a VITS2-BigVGAN decoder (fresh conv_pre)."""
import os, sys, glob, random, argparse, math
import torch, torch.nn.functional as F, soundfile as sf, numpy as np
os.chdir("<PROD>/sanskrit-tts/model/VITS2")
sys.path.insert(0, ".")
from models import Generator, MultiPeriodDiscriminator
from losses import feature_loss, generator_loss, discriminator_loss

# ---- Matcha's exact mel function (copied from matcha/utils/audio.py) ----
from librosa.filters import mel as librosa_mel_fn
_mel_basis, _hann = {}, {}
def mel_spectrogram(y, n_fft, num_mels, sr, hop, win, fmin, fmax, center=False):
    global _mel_basis, _hann
    key = f"{fmax}_{y.device}"
    if key not in _mel_basis:
        m = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(m).float().to(y.device)
        _hann[str(y.device)] = torch.hann_window(win).to(y.device)
    pad = int((n_fft - hop) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_hann[str(y.device)],
                      center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(_mel_basis[key], spec)
    return torch.log(torch.clamp(spec, min=1e-5))  # C=1

MP = dict(n_fft=1024, num_mels=80, sr=24000, hop=256, win=1024, fmin=0, fmax=None)
SEG = 8192

class VocDS(torch.utils.data.Dataset):
    def __init__(self, fl):
        self.paths = [ln.split("\t")[0].strip() for ln in open(fl) if ln.strip()]
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        a, sr = sf.read(self.paths[i])
        a = a.astype(np.float32)
        if a.ndim > 1: a = a.mean(1)
        if len(a) >= SEG:
            s = random.randint(0, len(a) - SEG); a = a[s:s + SEG]
        else:
            a = np.pad(a, (0, SEG - len(a)))
        a = torch.from_numpy(a)
        mel = mel_spectrogram(a.unsqueeze(0), **MP)[0]
        return mel, a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--warm", default="/tmp/bigvgan_decoder_25k.pt")
    ap.add_argument("--logdir", default="<PROD>/sanskrit-tts/model/VITS2/logs/matcha_bigvgan_vocoder")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    dev = "cuda"

    net_g = Generator(80, "1", [3,7,11], [[1,3,5],[1,3,5],[1,3,5]], [8,8,2,2], 512, [16,16,4,4],
                      gin_channels=0, use_bigvgan=True, snake_logscale=True, activation="snakebeta").to(dev)
    net_d = MultiPeriodDiscriminator(False).to(dev)

    # warm-start body (skip conv_pre: 192->512 in source vs 80->512 here)
    if a.warm and os.path.exists(a.warm):
        dec = torch.load(a.warm, map_location="cpu")
        dec = {k[4:]: v for k, v in dec.items() if k.startswith("dec.")}
        gsd = net_g.state_dict()
        load = {k: v for k, v in dec.items() if k in gsd and gsd[k].shape == v.shape and not k.startswith("conv_pre")}
        net_g.load_state_dict(load, strict=False)
        print(f"[warm] grafted {len(load)}/{len(gsd)} body tensors from BigVGAN decoder (conv_pre fresh)")

    opt_g = torch.optim.AdamW(net_g.parameters(), 2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(net_d.parameters(), 2e-4, betas=(0.8, 0.99))
    dl = torch.utils.data.DataLoader(VocDS("filelists/anu_train.txt.cleaned"), batch_size=a.bs,
                                     shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    use_wandb = not a.smoke
    if use_wandb:
        import wandb; wandb.init(project="sanskrit-tts", name="matcha-bigvgan-vocoder")
    step = 0
    net_g.train(); net_d.train()
    while step < a.steps:
        for mel, audio in dl:
            mel, audio = mel.to(dev), audio.to(dev).unsqueeze(1)  # (B,80,32),(B,1,8192)
            y_hat = net_g(mel)                                    # (B,1,8192)
            T = min(y_hat.size(-1), audio.size(-1)); y_hat = y_hat[..., :T]; y = audio[..., :T]
            # discriminator
            ydr, ydg, _, _ = net_d(y, y_hat.detach())
            loss_d, _, _ = discriminator_loss(ydr, ydg)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            # generator
            mel_hat = mel_spectrogram(y_hat.squeeze(1), **MP)
            mt = min(mel_hat.size(-1), mel.size(-1))
            loss_mel = F.l1_loss(mel_hat[..., :mt], mel[..., :mt]) * 45
            ydr, ydg, fr, fg = net_d(y, y_hat)
            loss_fm = feature_loss(fr, fg)
            loss_gadv, _ = generator_loss(ydg)
            loss_g = loss_gadv + loss_fm + loss_mel
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            if step % 20 == 0:
                nan = torch.isnan(loss_g).item() or torch.isnan(loss_d).item()
                print(f"step {step}: d={loss_d.item():.3f} g_adv={loss_gadv.item():.3f} fm={loss_fm.item():.2f} mel={loss_mel.item():.3f} NaN={nan}", flush=True)
                if use_wandb: wandb.log({"loss/d": loss_d.item(), "loss/g_adv": loss_gadv.item(), "loss/fm": loss_fm.item(), "loss/mel": loss_mel.item()}, step=step)
            if step and step % 5000 == 0 and not a.smoke:
                torch.save({"model": net_g.state_dict(), "step": step}, f"{a.logdir}/voc_G_{step}.pth")
                torch.save({"model": net_d.state_dict(), "step": step}, f"{a.logdir}/voc_D_{step}.pth")
                print(f"saved checkpoint at {step}")
            step += 1
            if step >= a.steps or (a.smoke and step >= a.smoke):
                print("SMOKE OK" if a.smoke else "done"); return
if __name__ == "__main__":
    main()
