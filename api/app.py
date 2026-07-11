"""Vāgdhenu — FastAPI REST API + embedded web UI.

Endpoints
---------
GET  /                  → embedded HTML UI (form + audio player)
GET  /api/health        → liveness + model status
GET  /api/meters        → list of available meters (chandas)
POST /api/chant         → JSON in, MP3 bytes out (audio/mpeg)
POST /api/chant/json    → JSON in, JSON envelope out (base64 mp3 + meta)

Run locally
-----------
    uv run uvicorn api.app:app --host 0.0.0.0 --port 8000

The server binds 0.0.0.0:8000 — point a browser or curl at http://<host>:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from urllib.parse import quote


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from . import lifespan as L
from .schemas import ChantRequest

# ── audio helpers ──────────────────────────────────────────────────────────────
SR = 24000


def _wav_to_mp3(audio: "list[float] | bytes", sr: int = SR) -> bytes:
    """Encode a float32 numpy array (or raw PCM bytes) to MP3 via pydub.

    Falls back to WAV if ffmpeg is not installed (pydub needs ffmpeg for mp3)."""
    import numpy as np

    if isinstance(audio, (bytes, bytearray)):
        # Assume already-encoded audio bytes; return as-is.
        return bytes(audio)
    arr = np.asarray(audio, dtype=np.float32)
    # float32 → int16 PCM
    pcm = (arr * 32767).clip(-32768, 32767).astype("int16")
    from pydub import AudioSegment

    seg = AudioSegment(
        pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1
    )
    buf = io.BytesIO()
    try:
        seg.export(buf, format="mp3", bitrate="128k")
        return buf.getvalue()
    except Exception:
        # ffmpeg missing → fall back to WAV-in-bytes (still playable by <audio>).
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()


# ── lifespan: warm the model once ──────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Warm the model in a thread so we don't block the event loop.
    if os.environ.get("VAGDHENU_LAZY", "0") != "1":
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, L.get_renderer)
    yield


app = FastAPI(
    title="Vāgdhenu — Sanskrit Chant TTS",
    version="1.0.0",
    description="REST API that renders a Sanskrit verse into traditional metered chant. Returns MP3.",
    lifespan=_lifespan,
)


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    ready = L._renderer is not None
    return {"status": "ok", "model_loaded": ready, "meters": len(L.METERS)}


@app.get("/api/meters")
async def meters():
    return {"meters": L.METERS, "fallback": L._FALLBACK, "auto": L._AUTO}


@app.post("/api/chant")
async def chant(req: ChantRequest, request: Request):
    """Synthesize one shloka → return raw MP3 bytes (audio/mpeg)."""
    meta, audio = await _synthesize(req, request)
    mp3 = _wav_to_mp3(audio, meta.sample_rate)
    mime = "audio/mpeg" if mp3[:2] != b"RI" else "audio/wav"
    return Response(content=mp3, media_type=mime, headers={"X-Meter": quote(meta.meter), "X-Duration": str(meta.duration_s)})


@app.post("/api/chant/json")
async def chant_json(req: ChantRequest, request: Request):
    """Synthesize one shloka → return JSON envelope (base64 mp3 + metadata)."""
    meta, audio = await _synthesize(req, request)
    mp3 = _wav_to_mp3(audio, meta.sample_rate)
    return {
        "meta": meta.model_dump(),
        "audio_base64": base64.b64encode(mp3).decode("ascii"),
        "audio_mime": "audio/mpeg" if mp3[:2] != b"RI" else "audio/wav",
    }


async def _synthesize(req: ChantRequest, request: Request):
    from .schemas import ChantMeta

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    # Meter resolution
    if not req.meter or req.meter == L._AUTO:
        detected = L.detect_meter(text)
        used, recognized = L.resolve_meter(detected)
    else:
        used, recognized = L.resolve_meter(req.meter)

    renderer = L.get_renderer()
    loop = asyncio.get_running_loop()
    try:
        sr, audio = await loop.run_in_executor(
            None, lambda: renderer.render_one(text, used, seed=int(req.seed))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rendering failed: {e}")

    meta = ChantMeta(
        meter=used,
        recognized=recognized,
        seed=int(req.seed),
        duration_s=round(len(audio) / sr, 3),
        sample_rate=sr,
    )
    return meta, audio


# ── embedded web UI ────────────────────────────────────────────────────────────
_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vāgdhenu — Sanskrit Chant</title>
<style>
  :root { --bg:#0f1117; --card:#1a1d27; --accent:#e8c46b; --text:#e8e8e8; --muted:#9aa0aa; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); }
  .wrap { max-width:820px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:1.6rem; margin:0 0 4px; }
  h1 span { color:var(--accent); }
  .sub { color:var(--muted); font-size:.9rem; margin-bottom:20px; }
  .card { background:var(--card); border-radius:14px; padding:20px; margin-bottom:16px;
          border:1px solid #2a2e3a; }
  label { display:block; font-size:.85rem; color:var(--muted); margin-bottom:6px; }
  textarea, select { width:100%; background:#12141c; color:var(--text); border:1px solid #2a2e3a;
                     border-radius:8px; padding:12px; font-size:1rem; font-family:inherit; }
  textarea { min-height:120px; resize:vertical; line-height:1.6; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .row > div { flex:1; min-width:160px; }
  button { background:var(--accent); color:#1a1d27; border:none; border-radius:8px;
           padding:12px 24px; font-size:1rem; font-weight:600; cursor:pointer; margin-top:16px; }
  button:disabled { opacity:.5; cursor:wait; }
  audio { width:100%; margin-top:14px; }
  .status { margin-top:12px; font-size:.9rem; color:var(--accent); min-height:1.2em; }
  .err { color:#ff6b6b; }
  details { margin-top:8px; }
  summary { cursor:pointer; color:var(--muted); font-size:.85rem; }
  .hint { font-size:.8rem; color:var(--muted); margin-top:6px; }
  a { color:var(--accent); }
  code { background:#12141c; padding:2px 6px; border-radius:4px; font-size:.85rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Vāgdhenu — <span>Sanskrit Chant</span></h1>
  <div class="sub">Paste a Sanskrit verse (any Indic script) → get a chanted MP3. Meter is auto-detected.</div>

  <div class="card">
    <label for="text">Your verse — one shloka at a time</label>
    <textarea id="text" placeholder="Paste a single Sanskrit verse in any Indian script…"
      >वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।
देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥</textarea>
    <div class="hint">One shloka per chant · separate pādas with newlines or <code>।</code> / <code>॥</code></div>

    <details>
      <summary>Advanced (optional)</summary>
      <div class="row" style="margin-top:12px">
        <div>
          <label for="meter">Meter (chandas)</label>
          <select id="meter"><option value="">✨ Auto-detect</option></select>
        </div>
        <div>
          <label for="seed">Seed (0–1000)</label>
          <input id="seed" type="number" value="60" min="0" max="1000"
                 style="width:100%;background:#12141c;color:var(--text);border:1px solid #2a2e3a;border-radius:8px;padding:12px;font-size:1rem">
        </div>
      </div>
    </details>

    <button id="btn" onclick="chant()">🎧 Chant it</button>
    <div class="status" id="status"></div>
    <audio id="player" controls style="display:none"></audio>
  </div>

  <div class="card">
    <h3 style="margin:0 0 8px">REST API</h3>
    <div class="hint">
      <code>POST /api/chant</code> — JSON body <code>{"text":"…","seed":60}</code> → returns MP3 bytes.<br>
      <code>POST /api/chant/json</code> — same body → JSON envelope with base64 MP3 + metadata.<br>
      <code>GET /api/meters</code> — list of available meters.
    </div>
  </div>

  <div class="sub" style="text-align:center;margin-top:24px">
    Voice & method: <a href="https://huggingface.co/prathoshap/vagdhenu" target="_blank">prathoshap/vagdhenu</a> ·
    <a href="https://github.com/prathoshap/vagdhenu" target="_blank">GitHub</a> · Apache-2.0
  </div>
</div>

<script>
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const player = document.getElementById('player');

// Load meter list
fetch('/api/meters').then(r=>r.json()).then(d=>{
  const sel = document.getElementById('meter');
  (d.meters||[]).forEach(m=>{
    const o=document.createElement('option'); o.value=m; o.textContent=m; sel.appendChild(o);
  });
});

async function chant(){
  const text = document.getElementById('text').value.trim();
  if(!text){ status.innerHTML='<span class="err">Please paste a verse first.</span>'; return; }
  const meter = document.getElementById('meter').value || null;
  const seed = parseInt(document.getElementById('seed').value||'60',10);
  btn.disabled = true; status.textContent = 'Chanting… (first run loads the model, ~30–60s)';
  player.style.display = 'none';
  try{
    const res = await fetch('/api/chant',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, meter, seed})
    });
    if(!res.ok){
      const err = await res.json().catch(()=>({detail:res.statusText}));
      throw new Error(err.detail || 'rendering failed');
    }
    const meterName = decodeURIComponent(res.headers.get('X-Meter') || '?');
    const dur = res.headers.get('X-Duration') || '?';
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    player.src = url; player.style.display = 'block'; player.play();
    status.innerHTML = `🪔 Meter: <b>${meterName}</b> · ${dur}s`;
  }catch(e){
    status.innerHTML = '<span class="err">'+e.message+'</span>';
  }finally{
    btn.disabled = false;
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
