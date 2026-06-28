"""Rung 1 — Phase-2 vocoder fine-tune: map Matcha's PREDICTED (teacher-forced, GT-aligned)
mels -> real audio. Warm-starts from the Phase-1 BigVGAN vocoder. Closes the 2-stage gap."""
import os, sys, glob, random, argparse, torch, torch.nn.functional as F, soundfile as sf, numpy as np
VD = "<PROD>/sanskrit-tts/model/VITS2"; os.chdir(VD); sys.path.insert(0, VD)
from models import Generator, MultiPeriodDiscriminator
from losses import feature_loss, generator_loss, discriminator_loss
from librosa.filters import mel as librosa_mel_fn
_mb, _hw = {}, {}
def mel_spectrogram(y, n_fft, num_mels, sr, hop, win, fmin, fmax, center=False):
    k = f"{fmax}_{y.device}"
    if k not in _mb:
        mm = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mb[k] = torch.from_numpy(mm).float().to(y.device); _hw[str(y.device)] = torch.hann_window(win).to(y.device)
    pad = int((n_fft - hop) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_hw[str(y.device)],
                      center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(_mb[k], spec), min=1e-5))
MP = dict(n_fft=1024, num_mels=80, sr=24000, hop=256, win=1024, fmin=0, fmax=None)
SEG_F = 32; SEG = SEG_F * 256  # 32 mel frames = 8192 samples

class P2DS(torch.utils.data.Dataset):
    def __init__(self, manifest):
        self.rows = [ln.strip().split("\t") for ln in open(manifest) if "\t" in ln]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        wav, melp = self.rows[i]
        mel = torch.load(melp, map_location="cpu").float()        # (80, T) predicted log-mel
        a, sr = sf.read(wav); a = (a if a.ndim == 1 else a.mean(1)).astype(np.float32)
        T = mel.shape[1]
        if T <= SEG_F:
            mel = F.pad(mel, (0, SEG_F + 1 - T)); T = mel.shape[1]
        s = random.randint(0, T - SEG_F)
        mel_seg = mel[:, s:s + SEG_F]
        a_seg = a[s * 256: s * 256 + SEG]
        if len(a_seg) < SEG: a_seg = np.pad(a_seg, (0, SEG - len(a_seg)))
        return mel_seg, torch.from_numpy(a_seg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="<PROD>/sanskrit-tts/data/predicted_mels/manifest.txt")
    ap.add_argument("--warm_dir", default="<PROD>/sanskrit-tts/model/VITS2/logs/matcha_bigvgan_vocoder")
    ap.add_argument("--logdir", default="<PROD>/sanskrit-tts/model/VITS2/logs/matcha_bigvgan_phase2")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.logdir, exist_ok=True); dev = "cuda"
    net_g = Generator(80, "1", [3,7,11], [[1,3,5],[1,3,5],[1,3,5]], [8,8,2,2], 512, [16,16,4,4],
                      gin_channels=0, use_bigvgan=True, snake_logscale=True, activation="snakebeta").to(dev)
    net_d = MultiPeriodDiscriminator(False).to(dev)
    # warm-start from latest Phase-1 vocoder
    gck = sorted([f for f in glob.glob(a.warm_dir + "/voc_G_*.pth") if "ema" not in f], key=lambda f: int(f.split("_G_")[1].split(".")[0]))[-1]
    net_g.load_state_dict(torch.load(gck, map_location="cpu")["model"]); print(f"[warm] G <- {os.path.basename(gck)}")
    dck = sorted(glob.glob(a.warm_dir + "/voc_D_*.pth"), key=lambda f: int(f.split("_D_")[1].split(".")[0]))
    if dck: net_d.load_state_dict(torch.load(dck[-1], map_location="cpu")["model"]); print(f"[warm] D <- {os.path.basename(dck[-1])}")
    opt_g = torch.optim.AdamW(net_g.parameters(), 1e-4, betas=(0.8, 0.99))   # lower LR = fine-tune
    opt_d = torch.optim.AdamW(net_d.parameters(), 1e-4, betas=(0.8, 0.99))
    dl = torch.utils.data.DataLoader(P2DS(a.manifest), batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    use_wandb = not a.smoke
    if use_wandb:
        import wandb; wandb.init(project="sanskrit-tts", name="matcha-bigvgan-phase2")
    step = 0; net_g.train(); net_d.train()
    while step < a.steps:
        for mel, audio in dl:
            mel, audio = mel.to(dev), audio.to(dev).unsqueeze(1)
            y_hat = net_g(mel); T = min(y_hat.size(-1), audio.size(-1)); y_hat = y_hat[..., :T]; y = audio[..., :T]
            ydr, ydg, _, _ = net_d(y, y_hat.detach()); loss_d, _, _ = discriminator_loss(ydr, ydg)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            mh, mg = mel_spectrogram(y_hat.squeeze(1), **MP), mel_spectrogram(y.squeeze(1), **MP)
            mt = min(mh.size(-1), mg.size(-1)); loss_mel = F.l1_loss(mh[..., :mt], mg[..., :mt]) * 45  # target = GT-audio mel
            ydr, ydg, fr, fg = net_d(y, y_hat); loss_fm = feature_loss(fr, fg); loss_ga, _ = generator_loss(ydg)
            loss_g = loss_ga + loss_fm + loss_mel
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            if step % 20 == 0:
                print(f"step {step}: d={loss_d.item():.3f} g_adv={loss_ga.item():.3f} fm={loss_fm.item():.2f} mel={loss_mel.item():.3f} NaN={torch.isnan(loss_g).item()}", flush=True)
                if use_wandb: wandb.log({"loss/d": loss_d.item(), "loss/g_adv": loss_ga.item(), "loss/fm": loss_fm.item(), "loss/mel": loss_mel.item()}, step=step)
            if step and step % 5000 == 0 and not a.smoke:
                torch.save({"model": net_g.state_dict(), "step": step}, f"{a.logdir}/p2_G_{step}.pth")
                torch.save({"model": net_d.state_dict(), "step": step}, f"{a.logdir}/p2_D_{step}.pth"); print(f"saved {step}")
            step += 1
            if step >= a.steps or (a.smoke and step >= a.smoke): print("SMOKE OK" if a.smoke else "done"); return
if __name__ == "__main__": main()
