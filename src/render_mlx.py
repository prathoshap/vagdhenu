"""Render a verse to a chanted wav on Apple silicon (MLX backend).

    python src/render_mlx.py --text "वसुदेवसुतं देवं कंसचाणूरमर्दनम् । देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥" --out out.wav
    python src/render_mlx.py --shard examples/sample_shard.json --outdir out      # same shard format as render.py

Any Indic script or IAST is accepted; the meter is detected when not given. Weights: models/mlx
(scripts/setup_mlx.sh), override with --model or $CHAMP_ROOT/mlx.
"""
import argparse, json, os, sys, time

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vagdhenu_mlx import Renderer, detect_meter_key  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--text", help="verse (pādas separated by daṇḍa or newline)")
ap.add_argument("--shard", help="JSON list of {id, meter, padas[], seed, out} like render.py")
ap.add_argument("--meter", default="auto")
ap.add_argument("--seed", type=int, default=60)
ap.add_argument("--out", default="out.wav")
ap.add_argument("--outdir", default="out")
ap.add_argument("--model", default=os.path.join(os.environ.get("CHAMP_ROOT", os.path.join(HERE, "..", "models")), "mlx"))
ap.add_argument("--nfe", type=int, default=64)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--speed", type=float, default=0.90)
ap.add_argument("--fp32", action="store_true", help="run the DiT in fp32 instead of bfloat16")
a = ap.parse_args()
if not (a.text or a.shard):
    ap.error("--text or --shard required")

import mlx.core as mx  # noqa: E402
t0 = time.time()
r = Renderer(a.model, nfe=a.nfe, cfg=a.cfg, speed=a.speed, dtype=mx.float32 if a.fp32 else mx.bfloat16)
print(f"[load] {time.time()-t0:.1f}s", flush=True)

jobs = []
if a.text:
    jobs.append(dict(id=os.path.splitext(os.path.basename(a.out))[0], meter=a.meter, padas=a.text, seed=a.seed, out=a.out))
if a.shard:
    for j in json.load(open(a.shard, encoding="utf-8")):
        jobs.append(dict(j, out=os.path.join(a.outdir, os.path.basename(j.get("out", j["id"] + ".wav")))))
for j in jobs:
    text = j["padas"]
    meter = j.get("meter") or "auto"
    if meter in ("auto", ""):
        meter = detect_meter_key(text if isinstance(text, str) else " । ".join(text)) or "vasantatilaka"
    t0 = time.time()
    sr, au = r.render_one(text, meter, seed=int(j.get("seed", 60)), no_sandhi=bool(j.get("no_sandhi", True)))
    os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
    sf.write(j["out"], au, sr)
    print(f"[{j['id']}] meter={meter} {len(au)/sr:.1f}s audio in {time.time()-t0:.1f}s -> {j['out']}", flush=True)
