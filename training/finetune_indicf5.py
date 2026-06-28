"""Fine-tune IndicF5 (F5/flow-matching) on pilot_reciter 5h. Warm-start from the GRN-corrected
checkpoint loaded directly into the CFM. DDP via `accelerate launch --multi_gpu`."""
import argparse, torch
from f5_tts.infer.utils_infer import load_model
from f5_tts.model import DiT, Trainer
from f5_tts.model.dataset import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", required=True)
ap.add_argument("--warm", required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--save_dir", required=True)
ap.add_argument("--wandb_name", required=True)
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=19200)
ap.add_argument("--bstype", default="frame")
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--save_per", type=int, default=2000)
a = ap.parse_args()

CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=a.vocab, device="cpu")
ck = torch.load(a.warm, map_location="cpu", weights_only=True)
sd = ck.get("model_state_dict", ck)
miss, unexp = cfm.load_state_dict(sd, strict=False)
print("[warm-start] missing(non-melspec):", len([m for m in miss if "mel_spec" not in m]), "| unexpected:", len(unexp), flush=True)

trainer = Trainer(
    cfm, epochs=a.epochs, learning_rate=a.lr,
    num_warmup_updates=a.warmup, save_per_updates=a.save_per, last_per_steps=a.save_per,
    checkpoint_path=a.save_dir, batch_size=a.bs, batch_size_type=a.bstype, max_samples=64,
    grad_accumulation_steps=1, max_grad_norm=1.0,
    logger="wandb", wandb_project="indicf5-sanskrit", wandb_run_name=a.wandb_name,
    mel_spec_type="vocos", log_samples=False,
)
MELKW = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
             target_sample_rate=24000, mel_spec_type="vocos")
train_dataset = load_dataset("indicf5", "custom", dataset_type="CustomDatasetPath",
                             mel_spec_kwargs=MELKW, data_dir=a.data_dir)
trainer.train(train_dataset)
