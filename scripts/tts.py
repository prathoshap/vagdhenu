#!/usr/bin/env python3
"""tts.py — wrapper: Sanskrit .txt in, .mp3 out.

Reads shloka(s) from a text file, splits into verses (॥) and hemistichs (।), drives
src/render.py, and stitches the per-hemistich wavs into MP3(s).

    python scripts/tts.py verse.txt                      # -> ./verse.mp3
    python scripts/tts.py verse.txt -o /tmp/vande.mp3    # -> that exact path
    python scripts/tts.py verse.txt --meter vasantatilakā --nfe 64 --seed 42

For large corpora — chunk into fixed-size MP3s with per-chunk resume:
    python scripts/tts.py bhagavatam.txt --chunk-size 20 -o bhagavatam.mp3 --resume
    # -> bhagavatam_001.mp3, bhagavatam_002.mp3, ...
    python scripts/tts.py mixed.txt --auto-meter --chunk-size 20 -o mixed.mp3
    # detect meter per verse; verses that don't classify fall back to --meter

Requires the venv from scripts/setup.sh and the BigVGAN clone. This script sets
PYTHONPATH for the render subprocess itself, so you don't need to export it.
"""
import argparse, json, os, re, subprocess, sys, tempfile, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RENDER = REPO / "src" / "render.py"
BIGVGAN = REPO / "BigVGAN"
# Verse-number lines like "1", "१", "೧" (Deva U+0966-096F, Kannada U+0CE6-0CEF)
_DIGITS_RE = re.compile(r"^[\d०-९೦-೯\s]+$")


def split_verses(text: str) -> list[list[str]]:
    """Split text into verses, each a list of hemistichs.

    Verse boundaries prefer ॥; falls back to blank-line separated blocks when no ॥
    is present. Within a verse, hemistichs are separated by । and newlines. Lines
    that are only digits (verse numbers like '१' or '1') are dropped."""
    if "॥" in text:
        raw = text.split("॥")
    else:
        raw = re.split(r"\n\s*\n", text)
    verses = []
    for v in raw:
        hs = []
        for chunk in v.split("।"):
            for line in chunk.splitlines():
                line = line.strip()
                if line and not _DIGITS_RE.match(line):
                    hs.append(line)
        if hs:
            verses.append(hs)
    return verses


def detect_meter(verse_text: str) -> str:
    """Best-effort chandas detection. Returns bank meter key (wav-stem, no diacritics
    e.g. 'anushtubh', 'vasantatilaka') or '' if unrecognized. Mirrors the logic in
    render_core.detect_meter_key without pulling torch."""
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import prep_text as PT
        from indic_transliteration import sanscript
        from tts_syllabify import syllabify
        from tts_weight import tag_weights
        from tts_meter import detect_meter as _detect
    except Exception:
        return ""
    try:
        d = PT.to_deva(verse_text).replace("॥", "|").replace("।", "|").replace("\n", " | ")
        d = "".join(c for c in d if not (c.isdigit() or ("०" <= c <= "९")) and c not in "\"'“”‘’()")
        slp = re.sub(r"\s+", " ", sanscript.transliterate(d, sanscript.DEVANAGARI, sanscript.SLP1)).strip()
        syls = syllabify(slp)
        tag_weights(syls)
        name = _detect(syls).get("name", "unknown")
    except Exception:
        return ""
    if name in ("anushtubh_half", "anushtubh"):
        return "anushtubh"
    if name in ("unknown", None, ""):
        return ""
    return name


def build_shard(verses: list[list[str]], meters: list[str], seed: int,
                outdir: Path) -> list[dict]:
    """Flatten (verses × hemistichs) into shard entries, propagating per-verse meter."""
    shard = []
    idx = 0
    for v, m in zip(verses, meters):
        for h in v:
            shard.append({
                "id": f"clip_{idx:04d}", "meter": m, "padas": [h],
                "seed": seed, "no_sandhi": True,
                "out": str(outdir / f"clip_{idx:04d}.wav"),
            })
            idx += 1
    return shard


def run_render(shard_path: Path, outdir: Path, results_path: Path, nfe: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BIGVGAN}{os.pathsep}{env.get('PYTHONPATH','')}"
    cmd = [sys.executable, str(RENDER),
           "--shard", str(shard_path),
           "--results", str(results_path),
           "--outdir", str(outdir),
           "--nfe", str(nfe)]
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO))


def concat_to_mp3(wavs: list[Path], out_mp3: Path, gap_ms: int = 350) -> None:
    """ffmpeg concat with a short silent gap between hemistichs."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH — 'brew install ffmpeg'")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rate = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(wavs[0])]
        ).decode().strip()
        gap_wav = td / "gap.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono",
             "-t", f"{gap_ms/1000:.3f}", str(gap_wav)],
            check=True)
        listing = td / "concat.txt"
        with listing.open("w") as f:
            for i, w in enumerate(wavs):
                if i: f.write(f"file '{gap_wav.as_posix()}'\n")
                f.write(f"file '{w.resolve().as_posix()}'\n")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-codec:a", "libmp3lame", "-qscale:a", "2",
             str(out_mp3)],
            check=True)


def render_chunk(verses: list[list[str]], meters: list[str], seed: int,
                 nfe: int, out_mp3: Path, keep_wavs: bool) -> None:
    """Render one chunk of verses to a single MP3."""
    work = Path(tempfile.mkdtemp(prefix="tts_"))
    try:
        wav_dir = work / "wavs"; wav_dir.mkdir()
        shard = build_shard(verses, meters, seed, wav_dir)
        shard_path = work / "shard.json"
        shard_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
        results_path = work / "results.json"
        run_render(shard_path, wav_dir, results_path, nfe)

        wavs = [Path(c["out"]) for c in shard]
        missing = [w for w in wavs if not w.exists()]
        if missing:
            sys.exit(f"render produced no wav for: {[str(m) for m in missing]} — see {results_path}")

        concat_to_mp3(wavs, out_mp3)

        if keep_wavs:
            keep_dir = out_mp3.parent / f"{out_mp3.stem}_wavs"
            keep_dir.mkdir(exist_ok=True)
            for w in wavs:
                shutil.copy2(w, keep_dir / w.name)
            print(f"[tts] kept wavs in {keep_dir}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanskrit .txt -> chanted .mp3")
    ap.add_argument("input", type=Path, help="UTF-8 text file with one or more shlokas")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output path (default: <input-stem>.mp3 in current dir). "
                         "With --chunk-size, treated as a base pattern: <stem>_NNN.mp3.")
    ap.add_argument("--format", default="mp3", choices=["mp3"],
                    help="output format (only mp3 for now)")
    ap.add_argument("--meter", default="anushtubh",
                    help="meter key in src/reference_bank/bank.json (default: anushtubh); "
                         "also the fallback when --auto-meter can't identify a verse")
    ap.add_argument("--auto-meter", action="store_true",
                    help="detect meter per verse (needs a full 4-pāda verse); "
                         "unrecognized verses fall back to --meter")
    ap.add_argument("--chunk-size", type=int, default=None, metavar="N",
                    help="emit one MP3 per N verses (default: one MP3 for the whole file); "
                         "output filenames are suffixed _001, _002, ...")
    ap.add_argument("--resume", action="store_true",
                    help="with --chunk-size, skip chunks whose output MP3 already exists")
    ap.add_argument("--nfe", type=int, default=32,
                    help="flow-matching steps (32 serving, 64 locked)")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--keep-wavs", action="store_true",
                    help="keep the per-hemistich wavs alongside the mp3(s)")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input not found: {args.input}")
    if args.resume and not args.chunk_size:
        sys.exit("--resume requires --chunk-size (chunk-level resume only)")
    if args.chunk_size is not None and args.chunk_size < 1:
        sys.exit("--chunk-size must be >= 1")

    text = args.input.read_text(encoding="utf-8")
    verses = split_verses(text)
    if not verses:
        sys.exit("no verses found in input")
    total_hemistichs = sum(len(v) for v in verses)
    print(f"[tts] {len(verses)} verse(s), {total_hemistichs} hemistich(s)")
    # short single-verse path: print hemistichs (backward-compat)
    if len(verses) == 1 and total_hemistichs <= 8 and not args.chunk_size:
        for i, h in enumerate(verses[0]):
            print(f"  [{i}] {h}")

    # per-verse meter
    if args.auto_meter:
        meters = []
        unknown = 0
        for v in verses:
            m = detect_meter(" ".join(v))
            if not m:
                m = args.meter
                unknown += 1
            meters.append(m)
        if unknown:
            print(f"[tts] auto-meter: {unknown}/{len(verses)} verse(s) unrecognized, "
                  f"using --meter='{args.meter}'")
    else:
        meters = [args.meter] * len(verses)

    # base output path
    base_output = args.output or (Path.cwd() / f"{args.input.stem}.mp3")
    if base_output.is_dir():
        base_output = base_output / f"{args.input.stem}.mp3"
    base_output = base_output.resolve()
    base_output.parent.mkdir(parents=True, exist_ok=True)
    suffix = base_output.suffix or ".mp3"

    # chunk plan
    if args.chunk_size:
        chunks = [(i, verses[i:i+args.chunk_size], meters[i:i+args.chunk_size])
                  for i in range(0, len(verses), args.chunk_size)]
        w = max(3, len(str(len(chunks))))
        outputs = [base_output.parent / f"{base_output.stem}_{k+1:0{w}d}{suffix}"
                   for k in range(len(chunks))]
    else:
        chunks = [(0, verses, meters)]
        outputs = [base_output]

    # render loop
    for k, ((v_start, v_list, m_list), out_mp3) in enumerate(zip(chunks, outputs)):
        v_end = v_start + len(v_list)
        n_hemi = sum(len(v) for v in v_list)
        prefix = (f"[tts] chunk {k+1}/{len(chunks)} "
                  f"(verses {v_start+1}..{v_end}, {n_hemi} hemistichs)"
                  if len(chunks) > 1 else "[tts] rendering")
        if args.resume and out_mp3.exists():
            print(f"{prefix}: {out_mp3.name} exists, skipping")
            continue
        print(f"{prefix} -> {out_mp3}")
        render_chunk(v_list, m_list, args.seed, args.nfe, out_mp3, args.keep_wavs)
        print(f"[tts] wrote {out_mp3}")


if __name__ == "__main__":
    main()
