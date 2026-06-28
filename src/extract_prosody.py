"""Per-akṣara prosody bank — for each QC-aligned clip, extract F0 (librosa.pyin), energy and
duration per akṣara, keyed to gaṇa features. F0 stored both absolute (Hz) and speaker-normalized
(semitones vs the clip's voiced median) so contours transfer across reference recordings. Feeds
the Tier-2 F0 conditioner (training target) AND inference (meter-matched reference → impose contour).
Mirrors extract_dgt.py's phone→akṣara mapping. LOCAL/CPU."""
import json, re, os, sys, numpy as np, librosa
sys.path.insert(0, "<REPO>/Sarvamoola/scripts")
from chandas_labeler import syllabify_slp1, scan, _clean, VOWELS, LONG

ROOT = "<REPO>/Final_Files/Prathosh_data"
TG = f"{ROOT}/aligned_qc"
AUD = f"{ROOT}/mfa16"                       # 16 kHz is plenty for F0<=400 Hz, faster pyin
SR = 16000; HOP = 256                       # 16 ms frames
man = {json.loads(l)["clip_id"]: json.loads(l) for l in open(f"{ROOT}/prathosh_manifest_v2.jsonl")}

def phones_of(tgfile):
    t = open(tgfile, encoding="utf-8").read()
    m = re.search(r'name = "phones".*?(?=item \[\d+\]:|\Z)', t, re.S)
    ivs = re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', m.group(0), re.S)
    return [(float(a), float(b), x.strip()) for a, b, x in ivs if x.strip()]

def f0_energy(wav):
    y, _ = librosa.load(wav, sr=SR)
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, hop_length=HOP, frame_length=2048)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0), len(rms)); f0, rms = f0[:n], rms[:n]
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    med = np.nanmedian(f0)                                 # clip voiced median (speaker ref pitch)
    return f0, rms, times, (med if np.isfinite(med) else 150.0)

def span_prosody(f0, rms, times, t0, t1, med):
    sel = (times >= t0) & (times < t1)
    fseg = f0[sel]; rseg = rms[sel]
    vv = fseg[np.isfinite(fseg)]
    if len(vv) == 0:                                        # unvoiced akṣara (e.g. pure stop)
        return dict(f0_hz=0.0, f0_st=0.0, f0_slope=0.0, f0_rng=0.0, voiced=0.0,
                    energy=float(np.nanmean(rseg)) if len(rseg) else 0.0)
    f0_hz = float(np.median(vv))
    st = 12*np.log2(vv/med)                                 # semitones relative to clip median
    slope = float(np.polyfit(np.arange(len(vv)), st, 1)[0]) if len(vv) > 1 else 0.0
    return dict(f0_hz=round(f0_hz, 1), f0_st=round(float(np.median(st)), 3),
                f0_slope=round(slope, 4), f0_rng=round(float(st.max()-st.min()), 3),
                voiced=round(len(vv)/max(1, len(fseg)), 3), energy=round(float(np.mean(rseg)), 5))

rows = []; skip = 0; done = 0
for cid, r in man.items():
    f = f"{TG}/{cid}.TextGrid"; w = f"{AUD}/{cid}.wav"
    if not (os.path.exists(f) and os.path.exists(w)): skip += 1; continue
    ph = phones_of(f); slp = r["align_text"]
    aks = syllabify_slp1(slp); ww, _ = scan(slp, pada_final_guru=False)
    if len(aks) != len(ww): skip += 1; continue
    if sum(len(_clean(a)) for a in aks) != len(ph): skip += 1; continue
    f0, rms, times, med = f0_energy(w)
    cur = 0; n = len(aks)
    pl = n//4 if n % 4 == 0 and n >= 8 else 0
    for i, ak in enumerate(aks):
        L = len(_clean(ak)); t0, t1 = ph[cur][0], ph[cur+L-1][1]; cur += L
        akc = _clean(ak); vi = next((j for j, c in enumerate(akc) if c in VOWELS), 0)
        vowel = next((c for c in akc if c in VOWELS), "a")
        pr = span_prosody(f0, rms, times, t0, t1, med)
        pip = round((i % pl)/(pl-1), 3) if pl > 1 else round(i/max(1, n-1), 3)
        pfin = 1 if pl and (i+1) % pl == 0 else (1 if i == n-1 else 0)
        rows.append(dict(clip=cid, idx=i, n=n, pos=round(i/max(1, n-1), 3),
                         lg=ww[i], matra=(2 if ww[i] == "G" else 1), n_onset=vi,
                         long_v=1 if vowel in LONG else 0, pip=pip, pfin=pfin,
                         dur=round(t1-t0, 4), clip_med_hz=round(med, 1), **pr))
    done += 1
    if done % 50 == 0: print(f"  {done} clips...", flush=True)

json.dump(rows, open(f"{ROOT}/prosody_bank.json", "w"))
# sanity: prosodic patterns the conditioner should capture
G = [r for r in rows if r["lg"] == "G" and r["f0_hz"] > 0]
Lr = [r for r in rows if r["lg"] == "L" and r["f0_hz"] > 0]
pf = [r for r in rows if r["pfin"] == 1 and r["f0_hz"] > 0]
npf = [r for r in rows if r["pfin"] == 0 and r["f0_hz"] > 0]
print(f"\nprosody_bank: {len(rows)} akṣaras from {done} clips (skipped {skip})")
print(f"median F0: GURU {np.median([r['f0_st'] for r in G]):+.2f} st  LAGHU {np.median([r['f0_st'] for r in Lr]):+.2f} st")
print(f"pāda-final F0 {np.median([r['f0_st'] for r in pf]):+.2f} st vs non-final {np.median([r['f0_st'] for r in npf]):+.2f} st (expect drop)")
print(f"pāda-final slope {np.median([r['f0_slope'] for r in pf]):+.3f} vs non-final {np.median([r['f0_slope'] for r in npf]):+.3f} st/frame")
print("saved prosody_bank.json")
