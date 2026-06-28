"""GaṇaDiT integration for IndicF5 — adds a zero-init per-char gaṇa(L/G) embedding to the
text path. Non-destructive: at init (gana_embed=0) the model == the champion exactly.
Patches an existing DiT instance (preserves champion weights) instead of forking the package.
"""
import types, json, torch, torch.nn as nn, torch.nn.functional as F

# --- gain accelerators (gaṇa-run lesson: zero-init embeds grow too slow at lr 2e-5) ---
GANA_GAIN = 15.0
SWARA_GAIN = 20.0
SWARA_LOOKUP_PATH = "<PROD>/swara_lookup_paired.json"   # {text: [[n_frames, ids], ...]}
_SWARA_LOOKUP = None
def _swara_ids(t, nf=None):
    global _SWARA_LOOKUP
    if _SWARA_LOOKUP is None: _SWARA_LOOKUP = json.load(open(SWARA_LOOKUP_PATH))
    cands = _SWARA_LOOKUP.get(t)
    if not cands: return [0]*len(t)
    if nf is None or len(cands) == 1: return cands[0][1]
    return min(cands, key=lambda c: abs(c[0]-nf))[1]   # disambiguate paired same-text clips by frames


def add_cond(dit, n_gana=3, n_swara=5):
    """Attach zero-init gaṇa(L/G) AND swara(note) embeddings; rebind forward to add both
    (gain-scaled). Zero-init ⇒ model == warm-start exactly at step 0."""
    td = dit.text_embed.text_embed.embedding_dim
    dev = next(dit.parameters()).device
    dit.gana_embed = nn.Embedding(n_gana, td).to(dev);  nn.init.zeros_(dit.gana_embed.weight)
    dit.swara_embed = nn.Embedding(n_swara, td).to(dev); nn.init.zeros_(dit.swara_embed.weight)
    dit.forward = types.MethodType(cond_forward, dit)
    return dit


def cond_forward(self, x, cond, text, time, drop_audio_cond, drop_text, mask=None, gana=None, swara=None):
    batch, seq_len = x.shape[0], x.shape[1]
    if time.ndim == 0: time = time.repeat(batch)
    t = self.time_embed(time)
    text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
    if not drop_text:
        if gana is not None:
            g = gana[:, :seq_len]
            if g.shape[1] < seq_len: g = F.pad(g, (0, seq_len-g.shape[1]), value=0)
            text_embed = text_embed + GANA_GAIN * self.gana_embed(g)
        if swara is not None:
            s = swara[:, :seq_len]
            if s.shape[1] < seq_len: s = F.pad(s, (0, seq_len-s.shape[1]), value=0)
            text_embed = text_embed + SWARA_GAIN * self.swara_embed(s)
    x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
    rope = self.rotary_embed.forward_from_seq_len(seq_len)
    residual = x if self.long_skip_connection is not None else None
    for block in self.transformer_blocks: x = block(x, t, mask=mask, rope=rope)
    if self.long_skip_connection is not None: x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
    x = self.norm_out(x, t)
    return self.proj_out(x)


def cond_cfm_forward(self, inp, text, *, lens=None, noise_scheduler=None):
    """CFM training forward threading BOTH gaṇa (char_gana_ids, deterministic) and swara
    (_swara_ids lookup keyed by the model_text). 1-char→1-id ⇒ both align per-char."""
    import random as _r
    from torch.nn.utils.rnn import pad_sequence
    from f5_tts.model.utils import list_str_to_idx, lens_to_mask, mask_from_frac_lengths
    if inp.ndim == 2: inp = self.mel_spec(inp); inp = inp.permute(0, 2, 1)
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
    gana = swara = None
    if isinstance(text, list):
        lns = lens.tolist() if lens is not None else [None]*len(text)   # mel frames -> disambiguate paired same-text clips
        gana = pad_sequence([torch.tensor(_gana_ids(t), dtype=torch.long) for t in text], padding_value=0, batch_first=True).to(device)
        swara = pad_sequence([torch.tensor(_swara_ids(t, lns[i] if lns[i] is None else int(lns[i])), dtype=torch.long) for i,t in enumerate(text)], padding_value=0, batch_first=True).to(device)
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
    pred = self.transformer(x=phi, cond=cond, text=text, time=time,
                            drop_audio_cond=dac, drop_text=dt, gana=gana, swara=swara)
    loss = F.mse_loss(pred, flow, reduction="none")[rsm]
    return loss.mean(), cond, pred


def gana_forward(self, x, cond, text, time, drop_audio_cond, drop_text, mask=None, gana=None):
    batch, seq_len = x.shape[0], x.shape[1]
    if time.ndim == 0:
        time = time.repeat(batch)
    t = self.time_embed(time)
    text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
    if gana is not None and not drop_text:                      # add per-char gaṇa (zero-init -> no-op at start)
        g = gana[:, :seq_len]
        if g.shape[1] < seq_len:
            g = F.pad(g, (0, seq_len - g.shape[1]), value=0)
        text_embed = text_embed + self.gana_embed(g)
    x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
    rope = self.rotary_embed.forward_from_seq_len(seq_len)
    residual = x if self.long_skip_connection is not None else None
    for block in self.transformer_blocks:
        x = block(x, t, mask=mask, rope=rope)
    if self.long_skip_connection is not None:
        x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
    x = self.norm_out(x, t)
    return self.proj_out(x)


def add_gana(dit, n_gana=3):
    """Attach a zero-init gaṇa embedding and rebind forward. n_gana: 0=filler,1=laghu,2=guru."""
    td = dit.text_embed.text_embed.embedding_dim
    dev = next(dit.parameters()).device
    dit.gana_embed = nn.Embedding(n_gana, td).to(dev)
    nn.init.zeros_(dit.gana_embed.weight)
    dit.forward = types.MethodType(gana_forward, dit)
    return dit


_GANA_CACHE = {}
def _gana_ids(t):
    """Per-char L/G ids for text t, cached (764 unique texts → scanned once each)."""
    v = _GANA_CACHE.get(t)
    if v is None:
        from chandas_labeler import char_gana_ids
        v = char_gana_ids(t)[0]
        _GANA_CACHE[t] = v
    return v


def gana_cfm_forward(self, inp, text, *, lens=None, noise_scheduler=None):
    """CFM training forward with gaṇa threaded — computes gaṇa from the same text strings
    CFM converts (list_str_to_idx is 1-id/char → guaranteed alignment). Bind via MethodType."""
    import random as _r
    from torch.nn.utils.rnn import pad_sequence
    from f5_tts.model.utils import list_str_to_idx, lens_to_mask, mask_from_frac_lengths
    if inp.ndim == 2:
        inp = self.mel_spec(inp); inp = inp.permute(0, 2, 1)
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
    gana = None
    if isinstance(text, list):
        gana = pad_sequence([torch.tensor(_gana_ids(t), dtype=torch.long) for t in text],
                            padding_value=0, batch_first=True).to(device)
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
    pred = self.transformer(x=phi, cond=cond, text=text, time=time,
                            drop_audio_cond=dac, drop_text=dt, gana=gana)
    loss = F.mse_loss(pred, flow, reduction="none")[rsm]
    return loss.mean(), cond, pred
