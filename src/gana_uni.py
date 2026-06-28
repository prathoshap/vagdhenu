"""Unified continuous prosody conditioner for IndicF5: gaṇa(L/G) + continuous F0 (pitch) +
continuous duration (timing), all added to the text path, zero-init (non-destructive), gain-scaled.
Continuous channels = Fourier-features → MLP. Training threads each syllable's ACTUAL f0/dur
(per-char lookups keyed by model_text); inference feeds metrical targets. Reuses gaṇa from gana_f5."""
import math, json, types, torch, torch.nn as nn, torch.nn.functional as F
from gana_f5 import GANA_GAIN, _gana_ids

F0_GAIN = 10.0
DUR_GAIN = 10.0
F0_LOOKUP_C = "<PROD>/f0_lookup_paired.json"   # {text: [[n_frames, per_char_vals], ...]}
DUR_LOOKUP_C = "<PROD>/dur_lookup_paired.json"
_F0C = None; _DURC = None
def _pick(cands, nf, n):                       # disambiguate paired (same-text) clips by frame count
    if not cands: return [0.0]*n
    if nf is None or len(cands) == 1: return cands[0][1]
    return min(cands, key=lambda c: abs(c[0]-nf))[1]
def _f0c(t, nf=None):
    global _F0C
    if _F0C is None: _F0C = json.load(open(F0_LOOKUP_C))
    return _pick(_F0C.get(t), nf, len(t))
def _durc(t, nf=None):
    global _DURC
    if _DURC is None: _DURC = json.load(open(DUR_LOOKUP_C))
    return _pick(_DURC.get(t), nf, len(t))


class ContEncode(nn.Module):
    """Continuous scalar (per char) → text-dim vector. Fourier features + MLP, zero-init output."""
    def __init__(s, td, K=16, scale=6.0):
        super().__init__()
        s.register_buffer("freqs", (2.0 ** torch.arange(K)) * math.pi / scale)
        s.mlp = nn.Sequential(nn.Linear(2*K+1, td), nn.SiLU(), nn.Linear(td, td))
        nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)
    def forward(s, v):                       # (B,T) -> (B,T,td)
        x = v.unsqueeze(-1)
        ff = torch.cat([x, torch.sin(x * s.freqs), torch.cos(x * s.freqs)], dim=-1)
        return s.mlp(ff)


def _padfit(v, seq_len):
    v = v[:, :seq_len]
    if v.shape[1] < seq_len: v = F.pad(v, (0, seq_len - v.shape[1]))
    return v


def add_cond_uni(dit, n_gana=3):
    """Attach gaṇa embed + continuous F0 encoder + continuous duration encoder (all zero-init)."""
    td = dit.text_embed.text_embed.embedding_dim; dev = next(dit.parameters()).device
    dit.gana_embed = nn.Embedding(n_gana, td).to(dev); nn.init.zeros_(dit.gana_embed.weight)
    dit.f0_encode = ContEncode(td).to(dev)
    dit.dur_encode = ContEncode(td).to(dev)
    dit.forward = types.MethodType(cond_forward_uni, dit)
    return dit


def cond_forward_uni(self, x, cond, text, time, drop_audio_cond, drop_text, mask=None, gana=None, f0=None, dur=None):
    batch, seq_len = x.shape[0], x.shape[1]
    if time.ndim == 0: time = time.repeat(batch)
    t = self.time_embed(time); text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
    if not drop_text:
        if gana is not None:
            g = _padfit(gana, seq_len); text_embed = text_embed + GANA_GAIN * self.gana_embed(g)
        if f0 is not None:
            text_embed = text_embed + F0_GAIN * self.f0_encode(_padfit(f0, seq_len))
        if dur is not None:
            text_embed = text_embed + DUR_GAIN * self.dur_encode(_padfit(dur, seq_len))
    x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
    rope = self.rotary_embed.forward_from_seq_len(seq_len)
    residual = x if self.long_skip_connection is not None else None
    for block in self.transformer_blocks: x = block(x, t, mask=mask, rope=rope)
    if self.long_skip_connection is not None: x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
    return self.proj_out(self.norm_out(x, t))


def cond_cfm_forward_uni(self, inp, text, *, lens=None, noise_scheduler=None):
    import random as _r
    from torch.nn.utils.rnn import pad_sequence
    from f5_tts.model.utils import list_str_to_idx, lens_to_mask, mask_from_frac_lengths
    if inp.ndim == 2: inp = self.mel_spec(inp); inp = inp.permute(0, 2, 1)
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
    gana = f0 = dur = None
    if isinstance(text, list):
        lns = lens.tolist() if lens is not None else [None]*len(text)   # mel frames per clip -> disambiguate paired
        gana = pad_sequence([torch.tensor(_gana_ids(t), dtype=torch.long) for t in text], padding_value=0, batch_first=True).to(device)
        f0 = pad_sequence([torch.tensor(_f0c(t, lns[i] if lns[i] is None else int(lns[i])), dtype=torch.float) for i,t in enumerate(text)], padding_value=0.0, batch_first=True).to(device)
        dur = pad_sequence([torch.tensor(_durc(t, lns[i] if lns[i] is None else int(lns[i])), dtype=torch.float) for i,t in enumerate(text)], padding_value=0.0, batch_first=True).to(device)
        text = list_str_to_idx(text, self.vocab_char_map).to(device)
    if lens is None: lens = torch.full((batch,), seq_len, device=device)
    mask = lens_to_mask(lens, length=seq_len)
    frac = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
    rsm = mask_from_frac_lengths(lens, frac)
    if mask is not None: rsm &= mask
    x1 = inp; x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    phi = (1 - t) * x0 + t * x1; flow = x1 - x0
    cond = torch.where(rsm[..., None], torch.zeros_like(x1), x1)
    dac = _r.random() < self.audio_drop_prob
    if _r.random() < self.cond_drop_prob: dac = True; dt = True
    else: dt = False
    pred = self.transformer(x=phi, cond=cond, text=text, time=time, drop_audio_cond=dac, drop_text=dt, gana=gana, f0=f0, dur=dur)
    loss = F.mse_loss(pred, flow, reduction="none")[rsm]
    return loss.mean(), cond, pred
