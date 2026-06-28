# Vāgdhenu — Sanskrit Chant TTS · Technical Report

_Scrubbed/anonymized public copy of the internal master report. Lab hostnames, IPs, usernames, and absolute paths are genericized; the pilot reciter's identity and data are withheld. Source of truth for the team remains the internal report._

---


_Single authoritative consolidation of the entire project: design, datasets, experiments (E0–E80h), production pipeline, lab paths, learnings. Consolidated 2026-06-24 from ~22 scattered docs across two machines (see Appendix D for the source list this replaces)._

> **Reading guide.** §1 is the current truth in one box. §2 is history. §3–§14 are the operational reference (infra, models, data, frontend, pipelines, QC, video, prosody). §15 = MBTN (shipped). §16 = zero-shot. §17 = Bhāgavatam (next). §18 = distilled learnings. Appendices: A path index · B experiment index · C glossary · D consolidated sources.
>
> **Where docs disagreed, this report states the AS-BUILT / current truth and flags the discrepancy inline (⚠).**

---

## 1. EXECUTIVE SUMMARY — current status

**What this is.** A production-grade **single-speaker Sanskrit *chant* TTS** (pārāyaṇa recitation of classical ślokas), plus the data/render/video pipelines around it. Two reciter voices exist: **Prathosh** (the builder; production voice) and **the pilot reciter** (pilot voice; Anuvyakhyana track).

**Current production model (as-built, the one that rendered MBTN):**
- **Architecture:** IndicF5 / F5-TTS — a flow-matching **DiT** (OT-CFM mel-infilling), `dim 1024 / depth 22 / heads 16 / ff_mult 2 / text_dim 512 / conv 4`, ~337M params, **no native duration or pitch head**. Routes Sanskrit through **Kannada script** (IndicF5 was trained on Indic scripts; Devanagari triggers Hindi schwa-deletion).
- **Voice checkpoint:** `CHAMPION_2026-06-11/voice_steer_ema_2026-06-17.pt` (voice-steered, step 2760). Fallback: `voice_armA_ema_2026-06-11.pt` (reference-driven champion).
- **Vocoder:** nvidia **BigVGAN-v2** (`bigvgan_v2_24khz_100band_256x`) fine-tuned on F5 vocos-mel → EMA soup → `voc_bigvgan_EMA_2026-06-11.pth`. **MANDATORY** — vocos produces a long-vowel phase "shiver" (E76).
- **Text frontend:** `prep_text.py` — `model_text` (plain Deva→Kannada) / `model_text_sandhi` (+ visarga sandhi).
- **MOS:** ~4.6 (expert listener, E32). Conjuncts incl. retroflex-aspirates 100% correct — the class the earlier Matcha line could not crack.

**Shipped:** **MBTN** (Mahābhārata Tātparya Nirṇaya — 32 adhyāyas, 5,183 verses) as 32 YouTube videos (Devanagari + Kannada cards, hemistich highlight, tanpura), **17 h 34 m total**, finals in `MBTN_videos/Final_production/`.

**Next:** **Śrīmad Bhāgavatam** (~18k verses), **audio-only** (app), **drop MFA QC**, source-validation-first (§17).

**Publication (decided 2026-06-24):** the whole project will be released publicly as an academic-style tech report structured as **two case studies — (1) MBTN (video, shipped) and (2) Bhāgavatam app (audio, to build)** — written after the app exists. Release scope = report + Sanskrit text frontend (open source) + model weights + curated samples; venues = HF model/space card + arXiv + blog. Data and the production voice are Prathosh's own (no third-party consent issue); remaining gates = base-model licenses (NVIDIA BigVGAN-v2, AI4Bharat IndicF5) for weight redistribution, source-text edition provenance, intended-use note, and scrubbing of paths/IPs/hostnames/names. See §20.

**The two big architectural truths:**
1. **F5 trade-off:** bulletproof *content* fidelity, but *prosody is text-driven and not designable* — a text-side prosody conditioner is architecturally inert (E59/E65/E68/E78). The reference and a voice-steering retrain are the only working prosody levers (E79).
2. **Reference IS the lever:** voice + swara contour + pace come from the reference clip, governed by the **half-reference rule** (ref_text must exactly match the ref audio's spoken span on a word/daṇḍa boundary).

---

## 2. PROJECT GOAL & SCOPE

- **Goal:** chant (not read) classical Sanskrit with correct prosody, metrically accurate durations, tradition-faithful F0 contours; "best Sanskrit TTS in existence" (no comparable system).
- **Quality bar:** target **MOS 4.0–4.3** (design projection for 30h); achieved ~4.6 via IndicF5 clone. Benchmarks: Human 4.6–4.7; ElevenLabs 4.5–4.6; OpenAI tts-1-hd 4.3–4.4; Azure 4.0–4.2.
- **Input:** Saṃhitā (already-sandhified) Devanagari (or Kannada — script-independent, `model_text` round-trips losslessly).
- **Output:** single speaker; **classical śāstra chant, no Vedic svaras** (udātta/anudātta).
- **Scope:** locked to **padya (metrical verse)**; gadya (prose) was dropped for the production model but is being revisited for Bhāgavatam Skandha-5 (§17).
- **Use cases:** pārāyaṇa/chanting content (sweet spot), study recordings, audio commentary, accessibility.

---

## 3. MODEL LINEAGE & HISTORY (four eras + two probes)

The project moved through four architecture eras. **Only Era 4 is production; Eras 1–3 are retired but documented so dead-ends aren't re-walked.**

### Era 1 — StyleTTS2 (E0–E8, ~2026-05-15 → 06-01)
- **E0 pilot:** StyleTTS2 fork, 5h Anuvyakhyana (the pilot reciter, 1929 clips), LJSpeech warm-start, 80+45 epochs → `epoch_2nd_00045.pth`, **MOS ~3.0**. Bugs: English PL-BERT loaded silently (config pointed at `Utils/PLBERT/`); English ASRCNN → MAS guessed alignments.
- **E1 = v0_mfa (breakthrough, MOS 4.2+):** three coupled changes — **Sanskrit PL-BERT** (`Utils/PLBERT_sanskrit/step_100000.t7`), **MFA-direct duration** (skip MAS), **fine-tune from pilot**. Dur Loss 1.47→0.57. `epoch_2nd_00025.pth`. Visarga/jihvāmūlīya/samyukta near-perfect, OOD prosody appeared.
- **E2–E8:** PAD-symmetry (no change), cluster-duration scaling (worse → conjunct elongation is a *decoder* limit, not alignment), IndicConformer alignment (identical to MFA → alignment not the cause), Conformer-duration fine-tunes (junk; CTC blank-eating). StyleTTS2 work locked to MFA-direct, then retired when VITS2 won.

### Era 2 — VITS2 (E9–E18, 2026-06-01 → 06-02)
- VITS2-main: 39.9M, from scratch, 5h, SLP1 (57 symbols), no PL-BERT, internal MAS, HiFi-GAN+iSTFT decoder. "Better than StyleTTS2, crisper conjuncts" → became track #1, retired v0_mfa. **Conjuncts still muffled** (E17 → data sparsity: ठ=28, ढ=35, ड=60 occurrences).
- **E9 VITS2+PL-BERT: KILLED** (baseline marginally better every metric). **E10/B1 VITS2+BigVGAN e2e: KILLED** (muffled). **E15** Sanskrit PL-BERT v2 (`step_225000.t7`) stopped — no consumer. **E16** the "60s explosion" is seed-intrinsic (seed 0 → 62s) → median-of-N seeds. Retired when Matcha won.

### Era 3 — Matcha-TTS + BigVGAN vocoder (E19–E28, 2026-06-02 → 06-03, ~MOS 4.2–4.3)
- Matcha acoustic model 18.2M, `checkpoint_epoch=399.ckpt` (mel_mean −4.644, std 2.196, fmax null). Vocoder bake-off (E20): BigVGAN ✅ / Vocos robotic ❌ / HiFi-GAN.
- **E23 MR-STFT BigVGAN WINS:** "hum gone, no noise, much better conjuncts and repha." Production vocoder = MR-STFT-EMA. **Repha smoothing is ACOUSTIC (vocoder), not data** (repha = 8000+ occurrences).
- **E24 Turing-test reframe** → plan 25h chant-first Prathosh corpus. **E25** corpus audit (MBTN 5,096 verses, SMV 1,004). **E26** chandas-duration prior (rule-based) → bake as training conditioning. **E27** recording manifest (12,487 segments, ~40h). **E28** ठ→ट swap confirms data sparsity.
- **Frozen pilot (2026-06-03):** Matcha ep399 + MR-STFT-v2-annealed BigVGAN-EMA, n_timesteps 32, temp 0.667, no HPF. **Superseded entirely by Era 4**; survives only in `Sarvamoola/production_assets/mrstft_v2_annealed/`.

### Era 4 — IndicF5 / F5-TTS (E29–E80h, 2026-06-03 → 06-18) — **THE PRODUCTION LINEAGE**
- **E29** IndicF5 (AI4Bharat, F5-TTS DiT, MIT, 1417h/11 langs) zero-shot Sanskrit cloning works. ★ **GRN bug**: ckpt stores `weight/bias`, installed f5_tts uses `gamma/beta` zero-init → text-encoder GRN ran as identity → garbled; monkeypatch fixes (|sum| 0→126.7). Devanagari→Hindi schwa-deletion → **route via Kannada script**.
- **E30–E32** fine-tune the pilot reciter 5h, **Kannada vs SLP1 ablation → Kannada wins** (rich pretrained embeddings). DiT lr 1e-5, frame-batch 19200, 600 epochs, bf16, vocos 100-mel. **E32 = ★ ~4.6 MOS chant clone** (`idf5_kannada_FINAL.pt`, 5.1GB) — all conjuncts incl. retroflex-aspirates 100%.
- **E33–E36** pivot to vṛtta conditioning (F5 has no duration head). Levers L0 ref-bank / L1 fix_duration / L2 text-elongation / L3 train-time chandas. Reference-prosody sensitivity confirmed (1.6× duration swing from reference alone, E35).
- **E37–E45** gaṇa labeler (98.5% match), GaṇaFeaturizer/DiT integration, **E41 implicit gaṇa is INERT** (ablation +0.00%) → build explicit Arch-B (DurationPredictor + LengthRegulator); MFA d_gt (guru:laghu 1.33–1.75×); DurationPredictor v2.
- **E47–E57** Prathosh corpus (E47: 3.02h/764 verses), `prep_text.py` (E48), fine-tunes + the **data-fix saga** (E50: Audacity lead-in offset; hyphen→join; snap-to-gap), reference-prosody 2-path, swara conditioner (E54 "exactly what I wanted" at gain), any-style paired data (E57: swara overrides conflicting reference). **Champion `prathosh_champion_styleA.pt` (model_8000).** Wobble = generator, not vocoder (definitive, E57 close-out).
- **E58–E65** Arm A (plain) vs Arm B (Kannada-norm) → **Arm A wins**; **E59 conditioner DROPPED** (inert, reference fully determines contour) + vocoder fix; **E62 ★ GOLD `CHAMPION_2026-06-11/` snapshot**; **E64 serving config locked**; **E65 reference = exact per-pāda L/G match**; conditioner buried with hard evidence.
- **E66–E70** tempo transfers from reference; **E67 cfg=3.0 = the tremor/jitter lever**; **E68 swara embedding = DEFINITIVE NEGATIVE** (F5 self-infilling reveals pitch from context mel → text-side token redundant → no gradient — *never re-attempt*); E69 visarga normalizer; **E70 ×3 half-reference (~20s)**.
- **E71 CosyVoice3** probe (overfit, 2.8h too small). **E72–E75 Indic-Parler** pivot (steerable prosody but content drift) — **dropped** (E76: F5 content is 100% perfect).
- **E76 ★ BigVGAN-v2 phase fix → PRODUCTION LOCKED** (`$PROD/production/`). **E77** prosody wall (reference is a *weak* prosody carrier; contour is text-driven). **E78** B-route conditioner trains but cannot control → DEAD END. **E79 ★ THE WORKING LEVER = voice-steering retrain + half-reference rule → SHIPPED** (`voice_steer_ema_2026-06-17.pt`, step 2760, steering corr 0.31→0.143). **E80–E80h** repeated-syllable repeat-prime fix, chorus self-double, tanpura, gap/visarga rules, internal sandhi (satva) ON.

---

## 4. CURRENT PRODUCTION STACK (as-built) — and the two render paths

There are **two render entry points** — the single source of much past confusion. Both use the same models/frontend; they differ in defaults:

| | **`render_batch.py`** (the MBTN batch renderer) | **`render.sh` / `render_production.py`** (champion single-verse) |
|---|---|---|
| Used for | MBTN's 32-adhyāya production; all batch/marathon work | one-off single-verse renders, A/B tests |
| Default voice | **`voice_steer_ema_2026-06-17.pt`** (steered → Prathosh) | **`voice_armA_ema_2026-06-11.pt`** (reference-driven) |
| Default **cfg** | **3.0** ⚠ | **1.2** ⚠ (user README; older champion guidance) |
| Default seed | 60 (MBTN marathon base; +att retry) | 50 |
| nfe / speed | 64 / 0.90 | 64 / 0.90 |
| gap / gap_halant | 0.55 / 0.20 | 0.30 / 0.20 |
| vocoder | BigVGAN-v2 (same) | BigVGAN-v2 (same) |

⚠ **cfg reconciliation:** the *as-built* MBTN renderer (`render_batch.py`) defaults to **cfg 3.0** (matches E67/E77's "cfg is the tremor lever, raise to 3.0"). The champion single-verse README still says **cfg 1.2** (E76 lock, not updated). For Bhāgavatam, the batch renderer (cfg 3.0) is the relevant baseline; tune per-ear.

**Locked inference params** (full table in §19). Headline: euler · nfe 64 · cfg 3.0 (batch) · sway −0.7 · speed 0.90 · BigVGAN-v2 · per-clip `fix_duration = ref_len + n_syll × sec_per_syll`.

---

## 5. LAB INFRASTRUCTURE

| Host | Alias | IP | User | GPUs | Role |
|---|---|---|---|---|---|
| ECE box | `GPU host` | GPU-HOST-A | user | 2× RTX A6000 48GB | **All production + training.** |
| ECE GPU host B | `GPU host B` / `GPU host` | GPU-HOST-B | prathosh / user | 2× RTX A6000 48GB | Spare; 4-GPU fan-out shards 2,3. |

- **SSH:** use the **alias** (key `~/.ssh/id_ed25519_ece`); raw IP skips IdentityFile → permission denied. sshd is intermittently slow → `-o ConnectTimeout=10..25`, retry up to 3×, use `tmux` for durable jobs. Filter banner noise: `grep -vE 'WARNING|post-quantum|upgraded|vulnerable'`.
- **GPU host B gotchas:** `/home` is 100% full → everything on `$HOME_HOST/hdd` (5.5T). `bigvgan` pip pkg copied to `…/pylibs/bigvgan`; BigVGAN HF cache (3.6G) on `$HOME_HOST/hdd/.cache/huggingface`; `HF_HOME` redirected. GPU host B user for user→GPU host B SSH is **`user@`** (default user wrong). **No shared filesystem** — rsync wavs user↔GPU host B over LAN (~110 MB/s).
- **Durable Python env (NEVER /tmp for production):** `$PROD/miniconda3/envs/indicf5/bin/python` (f5_tts, bigvgan, torch 2.4.1+cu121, librosa 0.11, transformers 4.46.3, accelerate==0.34.2). Other envs: `mfa` (MFA 3.3.9), `nemo1` (NeMo 1.19, IndicConformer), `matcha` (torch 2.3.1). Analysis venv `/tmp/prosenv` (numpy<1.25 + librosa + pyworld) — base anaconda has a numba/numpy clash.
- **/tmp fills with checkpoints** (5.4 GB each) → save champions to `/home`, prune.

---

## 6. MODELS & CHECKPOINTS

### Gold production (`$PROD/CHAMPION_2026-06-11/`, backup `Final_Files/CHAMPION_2026-06-11/`)
| File | What |
|---|---|
| `voice_steer_ema_2026-06-17.pt` | ★ **Production voice** (E79). Voice-steered on 179 paired clips, step 2760. Reference is a strong prosody lever (steering 0.31→0.143). `ema_model_state_dict` + optimizer. |
| `voice_armA_ema_2026-06-11.pt` | **Champion (fallback)**, Arm A EMA, step 14800. F5/IndicF5 DiT. Reference-DRIVEN (voice+swara+pace from reference) — clones a reference's *style* but is voice-locked to its training voices (garbles a truly new voice). |
| `voc_bigvgan_EMA_2026-06-11.pth` | **Vocoder.** nvidia BigVGAN-v2 24k/100-band/256x fine-tuned on F5 vocos-mel, EMA soup of last-6. **md5 be29d98f43e5a33e3700a49d98ec75cb.** |
| `prep_text_2026-06-11.py`, `MANIFEST.md`, `CHECKSUMS.md5` | frontend snapshot + provenance. |

- **Base IndicF5 (unsteered, generalizes / true zero-shot):** `~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/model.safetensors` (+ `checkpoints/vocab.txt`). Load: strip `ema_model._orig_mod.` prefix → DiT (364 tensors, exact). Caveats: schwa-deletion on raw Devanagari (fix: Kannada routing), flat (non-chant) prosody.
- **Loading caveat (load-bearing):** the BigVGAN ckpt MUST load via `bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")` then `g.load_state_dict(sd.get("model",sd)); g.remove_weight_norm()`. The VITS2 `Generator` class silently half-loads → monotone garbage (E62).
- **Other durable ckpts:** `idf5_kannada_FINAL.pt` (the pilot reciter champion, 5.1G, E32), `armA_voice.pt` (5.4G full-w-optimizer resume), `prathosh_champion_styleA.pt` (E57 model_8000), `gana_ckpt/model_last.pt` (shelved conditioner), `armB_mirror/model_last.pt`. Legacy Matcha: `model/Matcha-TTS/.../checkpoint_epoch=399.ckpt` + MR-STFT in `production_assets/mrstft_v2_annealed/`.
- **Sizing/serving (E60):** F5 voice 337M + BigVGAN 112M = ~449M (~900MB fp16). nfe64 fp32 RTF 1.24; **nfe32 RTF 0.63** (serving); CPU ~22 min/half (GPU mandatory). Peak inference 2.5GB.

### What each model does (the zero-shot map — see §16)
- **steered** → imposes Prathosh's voice (not zero-shot).
- **base IndicF5** → clones any voice (true zero-shot), but flat prosody + schwa needs Kannada routing.
- **armA champion** → chant swara from reference, but voice-locked (garbles OOD voices).

---

## 7. DATASETS & RECORDING CORPUS

### Two corpora
- **the pilot reciter / Anuvyakhyana pilot (5h):** 1929 clips, anuṣṭubh-heavy (87%), audible word-gaps (taught the model "space→pause"). Source of v0_mfa / IndicF5 champion. Also a separate finished deliverable (§13).
- **Prathosh chant corpus (production):** new speaker, chant-first, no warm-start. As of E47: **~3.02h kept (181 min, 764 verses, 33,704 syllables)** across 10 sessions in `Final_Files/Prathosh_data/`. Plan was **25h clean (50–65h raw)** ⚠ (original design said 30h; redesign settled 25h). ~9 meter families.

### Recording strategy / protocol (the hard-won rules)
- **Equipment:** one room, one mic (large-diaphragm condenser), fixed ~15–20cm + pop filter, noise floor ≤ −60 dB, **24-bit/48kHz** (downsample to 24kHz for training), peaks ~ −6dB, save lossless WAV. **Tanpura/śruti drone in the monitor** (anchors Sa across sessions; NOT in the recording).
- **Delivery:** natural pārāyaṇa (not theatrical); steady pace/register session-to-session; slight natural tempo variation is *good* (feeds prosody manifold).
- **★ Word-boundary rule (the key pilot fix):** **no audible gaps between words within a pāda** — each pāda is one continuous breath-group. **Pause ONLY at daṇḍa** (`।` small, `॥` larger). Long meters: one *silent* breath at the **yati** (caesura), same fixed position every clip (vasantatilakā after 8; mandākrāntā 4,10; śikhariṇī 6; śārdūla 12; sragdharā 7,14).
- **Phonetic priorities** (pilot weaknesses): hold long vowels ~2×; sustain terminal visarga; articulate retroflex ṇ/ṣ/ṭ/ḍ; audible aspiration; hold geminates; don't smear sph/kṣ/jñ/hr/dr.
- **Session hygiene:** 1–2h sessions, warm up + discard first minutes, consistent time-of-day, re-check a session-1 reference each session, interleave 3–4 meters/session (never all-one-meter for hours).

### The vṛtta recording sheet (`Texts/recording_sheet_vruttas.md`)
"Sumadhva-vijaya, meter-diverse (10/meter target)" — **62 verses, 10 meters**, one file per verse named by `Sarga.Śloka` (e.g. `4.53.wav`). Anuṣṭubh 4 (the pilot reciter covers), 10-akṣara 10, indravajrā/upajāti 10, vaṃśastha 10, **rucirā 4 (HELD-OUT)**, vasantatilakā 10, **mālinī 4 (HELD-OUT)**, mandākrāntā/śikhariṇī 5, śārdūla 4, sragdharā 1 (use ~10 from harivayustuti.json). Held-out meters = unseen-meter generalization test.

### Text sources (the data spine)
- **Primary: MBTN** — `Texts/mbtn.json` (canonical, per-verse) + `Sarvamoola/Texts/mbtn_split.json` (per-line, the build source). 32 adhyāyas, **5,096 verses** (build counts 5,183 incl. maṅgalas/structure), ~213k akṣaras (~27–42h chant). Meter: anuṣṭubh 48% / triṣṭubh-jagatī 39% / vasantatilakā 13% / long <1%. ठ=154, ढ=48 (below 300 booster threshold). repha 8000+ (acoustic, not data).
- **Sumadhvavijaya:** `Texts/sumadhvavijayah_mula.json` (16 sargas, 1,004 verses; built from a 9.7MB scrape, commentaries stripped). Adds mālinī 5.6% + atijagatī.
- **Vāyu Stuti:** `Texts/vayustuti.json` (43 verses, all sragdharā) — ⚠ **repha-corrupted** (तीर्थ→तीथर्), EXCLUDED from manifest until clean text sourced.
- **Recording manifest (built E25/E27):** `Texts/recording_manifest.json` + `recording_script.tsv` — **12,487 half-verse segments, ~40h**, each with Devanagari, SLP1, precomputed **`gl` (laghu/guru chandas feature)**, meter, ref, est_sec, priority, long_split, is_pramana.

### Corpus audit thresholds (booster targets, §3.5 of recording_corpus_design)
Retroflex `W/Q/q/w/R` ≥300 each; long vowels ≥500; repha/R-clusters ≥200; hard conjuncts kṣ/jñ/tr/dr/dhr/hm/hn ≥150; all 57 SLP1 symbols ≥100. Tool: `corpus_coverage_audit.py` / `corpus_audit.py`.

---

## 8. DATA-PROCESSING PIPELINES (two distinct pipelines)

### 8a. Recording → TTS dataset (cut → prosody → pair → augment → train → eval)
1. **Cut clips:** per `[start,end]` → resample 24kHz → `librosa.effects.trim(top_db=30)` → `{verse_id}_styleX.wav`. **Two timestamp conventions** (confirm before cutting): (A) **direct wav-timeline** (new/preferred, contiguous, no offset); (B) **constant-offset** (old Prathosh sessions — Audacity records before the timestamp app; `OFFSET = st − first_block_start`; `et` = last-block end, NOT a linear rescale — linear scaling is the bug that cut late verses short). Snap-to-gap refine (HOP 0.01, WIN 0.025, gap-window 0.35, PAD 0.06). ⚠ **Never write into source `Wav/`/`Timestamp/`.**
2. **Verify pairing** (paired clips share text → disambiguate by mel-frame count; A/B durations must differ >0.4s).
3. **Extract per-syllable prosody** (`extract_prosody.py`, matra-proportional over voiced frames).
4. **Paired lookups** (swara/f0/dur, frame-keyed).
5. **Augment** (`f5_ds_rows_v2.json`; oversample paired ~3–5×, NOT 15× — overfits, E57).
6. **Train (DDP):** `accelerate launch --multi_gpu --num_processes 2`, lr 1–2e-5, frame-batch 12800 (≈28GB stable), save/500–2000, tmux, expandable_segments.
7. **Eval obedience:** F0-contour ↔ template correlation.
- **QC gate:** trim to ~0.1s, peak < −1dB, duration 4–15s, loudness −23 LUFS, forced-align sanity (MAS), manifest row `wav \t SLP1 \t meter \t session \t take`.

### 8b. Recording session → clean YouTube audio + sync files (7 stages, `Final_Files/Scripts/PIPELINE.md`)
Master text JSON (UUID-keyed blocks) + session timestamp JSON (per-block start/end, possible `Start time = X sec` offset on line 1, `redos[]` end-of-session retakes).
1. **ASR** (IndicConformer on A6000, env `nemo1`, model `indicconformer_stt_sa_hybrid_rnnt_large.nemo`, lang `sa`) → word timestamps.
2. **Parse + offset correction** (+ redo overrides, WAV-absolute).
3. **Slice** segments from WAV (ffmpeg; gaps = retakes, discarded).
4. **ASR verification** (RapidFuzz token_sort_ratio: ≥0.85 pass / 0.60–0.85 flag / <0.60 fail; never auto-dropped).
5. **Stitch + clean timestamps** (silence: 500ms after `॥`, 200ms after `।`/prose).
6. **Sync files** (SRT / VTT / app JSON).
7. **Speed normalize** (mora-count → target pace → `pyrubberband` pitch-preserving stretch, clamp [0.67,1.5×]).

### 8c. MFA setup (the v0_mfa backbone, reused for d_gt)
- env `mfa` (MFA 3.3.9 + openfst + kaldi). Dict = identity char-as-phoneme (52 SLP1 chars). AM `mfa_work/anu_acoustic.zip`; alignments `mfa_work/aligned/pilot_reciter/*.TextGrid` (1929 clips). Prathosh adapted AM `prathosh_adapted.zip` / `prathosh_mfa_new/prathosh_new_adapted.zip`; dict `prathosh_new_dict.dict`.
- MFA scripts: `parse_mfa_textgrids.py`, `styletts2_mfa_helpers.py`, `extract_dgt.py`. d_gt: guru:laghu ≈ 1.75× median (Prathosh), 1.33–1.59× (the pilot reciter, prose-like).

---

## 9. TEXT FRONTEND — `prep_text.py` (every rule)

Two entry points: **`model_text(src)`** = plain Deva→Kannada (champion path, no sandhi); **`model_text_sandhi(src, echo_final=)`** = + internal visarga sandhi. `align_slp1(src)` = SLP1 for MFA. Pipeline: `to_deva` → strip punct → Deva→SLP1 → (sandhi) → SLP1→Kannada.

- **Script routing:** Sanskrit is rendered as **Kannada** (IndicF5's strong script; Devanagari → Hindi schwa-deletion).
- **Internal visarga sandhi (E69/E80h):** apply **utva / rutva / lopa**; **satva** before c/ch→ಶ್, ṭ/ṭh→ಷ್, t/th→ಸ್ but KEEP PLAIN before ś/ṣ/s, k/kh, p/ph; keep **jihvāmūlīya/upadhmānīya plain** (model learned them acoustically). Pipeline order: utva/rutva/lopa → `_satva` → `_anusvara_m` → `_danda_fix`. Default ON; `--no_sandhi` for already-sandhified text (MBTN normals). **Citation verses (pramāṇa, quote-marked) → render WITH sandhi.**
- **Homorganic anusvara (`_anusvara_m`):** `ं` → homorganic nasal of the FOLLOWING consonant (looks past spaces): ka→ಙ್, ca→ಞ್, ṭa→ಣ್, ta→ನ್; pa-varga & y/r/l/v/ś/ṣ/s/h → ಮ್. (Fixes तारतम्यं→"tāratan" drop.)
- **Daṇḍa-final (`_danda_fix`):** trailing visarga `ಃ` after SHORT vowel → echo (`ha/hi/hu/hṛ`; inherent a→ಹ); after LONG vowel/diphthong → **bare** (elongates naturally). Trailing anusvara `ಂ` → `ಮ್`. ⚠ The bare long-vowel visarga can garble to "haha" on rare verses (MBTN adh01 v18) — handled per-clip, NOT globally (user prefers natural bare visarga).
- **`_hna_metathesis`:** ह्ण→ण्ह, ह्न→न्ह (गृह्णन्ति→गृण्हन्ति) — F5 struggles with ह्ण/ह्न; metathesis is cleaner + legit chant.
- **`_vocalic_l`:** कॢ→क्लृ.
- **Deergha-ṝ (long vocalic ṝ, ॄ/ॠ):** SLP1 `F` → `"rU"` at the SLP1 stage in model_text/model_text_sandhi/align_slp1 (IndicF5 mispronounces Kannada ೄ). gives ತ್ರೂ "trū" not ತರೂ.
- **Editorial parentheticals:** strip `\([^)]*\)` before hemi-split — `(नख)`,`(ऽ)` are not recited and corrupt meter detection (added to the build; carry into Bhāgavatam).
- **Schwa:** routing through Kannada (not raw Devanagari) prevents Hindi-style schwa deletion.

---

## 10. REFERENCE BANK & METERS

- **Bank:** `production/reference_bank/bank.json` + ~27 ref wavs (16+ meters + `repeat_primes` + `gadya`/`gadya_mbtn`). `--meter <name>` auto-loads ref wav + ref_text; nearest-meter fallback by syllable count. Source-of-truth bank also at `Final_Files/reference_bank_final/bank.json`, pushed to both boxes.
- **Per-meter `sec_per_syll`** (= ref pace × 1.10, baked into bank): e.g. anuṣṭubh **0.326** (current, v094 ref) ⚠ (earlier docs 0.277/0.259-era values — bank.json is authoritative), vasantatilakā 0.259, vaṃśastha 0.255, upajāti 0.273, sragdharā 0.31. Renderer `--sec_per_syll -1` = use bank value; `0` = speed-based; `>0` = explicit. `fix_duration = ref_len + n_syll × sec_per_syll` (F5 pads-not-stretches → over-budget = silence parked mid-verse).
- **★ HALF-REFERENCE RULE (E79):** `ref_text` must match the ref audio's spoken span on a clean word/daṇḍa boundary. The per-meter `References/` JSON lists the *full* verse but the `.wav` holds only what was chanted → a clean half-hemistich (~7s) beats a full 15s shloka. Databank `model_text` is always exact. Ref ≤15s (F5 clips audio>15s but keeps full text → garble); `clip_short=False` bypasses to ~44s (DiT 4096 frames) but quality degrades >25s. ★ Half-śloka ref repeated ×3 (~20s) markedly improves swara.
- **★ Reference selection = exact per-pāda L/G match** (4/4 > 3/4 > meter-name > syll-count), cleanest HNR, ends at a daṇḍa.
- **Repeat-prime rule (E80):** F5 drops adjacent identical tokens; in-context **repeat-priming** — clone from a ref whose audio+text demonstrates repeat depth ≥ target (rule: prime depth ≥ target). Banked `prime_jaya` (narasimha_20), `prime_chata` (narasimha_18), `prime_mono` (sumadhwa_10_44). Must be in-distribution. **Autoprime:** di-repeat ≥3 → di-prime; mono-repeat ≥2 → prime_mono. Render hemistich-wise (no pāda-resplit — resplit inserts a pause AND disrupts priming).
- **Meter detection** (`build_mbtn_adh_v2.py`): L/G signatures self-calibrated from bank refs; per-pāda match with pāda-final anceps ignored; pramāṇikā-aware; **per-hemistich** `hemis_meter[]` for mixed-vṛtta verses (~1,150 in MBTN). Reference databanks for picking clean refs: `prathosh_new/` (style B, `ds__*.wav`, manifest + `vq_qc.json` HNR) and old style A (`mbtn1__*.wav`).

---

## 11. TTS RENDER PIPELINE

- **`render_batch.py`** (user + GPU host B, md5-identical): loads DiT + BigVGAN once, loops a shard of per-hemistich clips. Bit-identical to gold one-shot (max_diff 0.0). Per-clip JSON fields: `id, meter, true_meter, padas[], seed, no_sandhi, sps, speed, ref_wav, ref_text, no_autoprime, out`. Flags: `--shard --outdir --results --voice --voc --bank --nfe 64 --cfg 3.0 --speed 0.90 --gap 0.55 --gap_halant 0.20 --dump_raw`.
- **4-GPU fan-out:** `build_clips_gen.py N` → per-hemistich clips + 4 round-robin shards. Shards 0,1 → user GPU0,1; 2,3 → GPU host B GPU0,1 (`turing_render.sh`, `.done` markers); rsync GPU host B→user to centralize. Drivers: `run_adhyaya.sh N` (full per-adhyāya), `marathon_serial.sh [from] [to]` (sequential 1..32, robust), `swap_shloka.sh N verse [seed]` (incremental one-shloka re-render + re-assemble), `consolidate_log.sh N` (per-adhyāya `adhNN_FULL.log`).
- **Gate (post):** head/tail RMS gate (removes vocoder onset transient + tail ring); **fricative-aware** (ś/ṣ/s/h onset → low floor + skip fade-in, fixes स्तुहि→"tuhi"); **halant-aware** (त्/क्/प् final → preserve stop burst). `compress_sil(>0.28→0.12s)`.
- **F5 facts that drive every fix:** non-AR flow-matching DiT, in-context cloning, **no duration predictor**; (1) clones reference prosody/relative timing; (2) `fix_duration` sets only total mel length, pads-not-stretches; (3) stochastic per-seed syllable drops; (4) full-verse ref ending on a cadence swallows the first generated syllable → use half-mode ref; (5) melody is seed-stochastic → best-of-N.

---

## 12. QC PIPELINE — MFA retrospective (dropped), duration gate (kept)

- **MFA QC (built E-series, run on MBTN, then RETROSPECTIVELY JUDGED LOW-VALUE):** 3-gate `qc_loop_gen.py` — (1) MFA word-DROP (two-tier sec/syll), (2) MFA-UNALIGNED, (3) duration-fidelity. Auto-reseed loop (≤3 iters). **Retrospective (MBTN, 10,220 clips):** drop+unaligned = ~1,648 flags, **~0 real** (15-clip sample: 0 real); only the **duration gate** caught the 2 genuine collapses. MFA's false-positive floor (conjuncts, fricative onsets, long meters, prime-swapped) dominated. **→ For Bhāgavatam: drop MFA; keep the cheap duration gate + ear-spot + structural/source validation** (where every real defect actually came from). The `unaligned` gate was already softened to report-only (no reseed).
- **swap_shloka** is the fix primitive: re-render one shloka + incremental re-assemble (only that segment re-encodes, rest `concat -c copy`).

---

## 13. VIDEO ASSEMBLY (MBTN-specific; Bhāgavatam is audio-only)

- **`assemble_modular.py N`:** per-shloka video-only segment mp4 (`-frames:v k`, k=round(dur·24)) → `concat -c copy` → continuous tanpura over concatenated voice → mux `-c:v copy`. Each segment's voice snapped to whole-frame (SR/fps = 24000/24 = **1000 samples**) → zero A/V drift. Incremental encode keyed on per-segment voice SHA1 (a style change needs `rm seg/*.sha`).
- **Card:** Devanagari + Kannada, **N-hemistich** verse cards (highlight advances per hemi across N−1 known split times; font auto-shrinks for n>2), cream highlight `&H0096E1F5&` / grey `&H00686868&`, FS 80/80, corner headers (`ಮ.ಭಾ.ತಾ.ನಿ — ಅಧ್ಯಾಯ N` / `म.भा.ता.नि — अध्यायः N`), verse number `॥ N/v ॥`. WrapStyle 0.
- **Structure:** invocation → atha-colophon → verses (v0 maṅgala if present) → iti-colophon → madhvārpaṇam. Prose (invocation/colophons) rendered with bank `gadya_mbtn`, `sps:0`. Colophons via `colophon_gen.py`. VGAP 0.95s inter-shloka, HGAP 0.55s within-verse. **Tanpura:** Pa-Sa-Sa-Sa drone, **Sa LOCKED to 170 Hz** (per-adhyāya F0 estimation drifted; the tonic is a voice constant), `mix = 0.92·voice + 0.04·tanpura`.
- **ffmpeg gotchas:** user ffmpeg 4.2.7 has no `gradients` filter (use PIL/numpy bg PNG); `-framerate 24` on image input + `-r 24` out (else 25fps default → drift); build on user (has libass + Lohit fonts); local mac ffmpeg lacks libass.
- **Anuvyakhyana video track (separate):** the pilot reciter's *actual recitation* → MP4s via **headless-Chrome bilingual cards** (`render_card.py` Adishila+Noto Kannada, aksharamukha translit; `make_adhyaya_video.py` cross-dissolves; title/credit/closing bookends). 4 adhyayas done (`Final_Files/Anuvyakhyana/output/`). **Tanpura abandoned there** (chanter Sa drifts ½–1 semitone across an unaccompanied multi-session recording → fixed drone would beat).

---

## 14. PROSODY CONTROL — the open problem (definitive)

- **The wall:** designed / verse-independent prosody is **not reachable with native F5**. Contour is **text-driven** (same-ref/diff-verse F0 corr ~0.13; diff-ref/same-verse 0.40–0.54).
- **Text-side conditioner = architecturally DEAD** (E59/E65/E68/E78): F5's self-infilling feeds the clip's own context mel (reveals pitch) → gaṇa/swara embedding is redundant → no gradient (norm stays ~0.02; obedience ON≈OFF; +0.0st even at cfg4). **Never re-attempt embedding tuning.**
- **cfg trades expression vs tremor** (coupled — both ride stochastic F0/jitter): cfg 1.2 expressive+tremor; cfg 3.0 clean (jitter dead-on real); cfg 5 flattens swara. NFE ≥64 for visarga.
- **The working levers:** (1) **the reference** — supply the prosody you want as a clean, exactly-matched (half-reference rule) reference; dramatic-contour ref transfers span ~1:1; (2) **voice-steering retrain** (E79) makes the voice more reference-responsive; (3) **WHOLE-VERSE rendering** for short meters (pāda1↔pāda2 corr 0.38→0.63). Arbitrary designed swara would need a different architecture (explicit Arch-B duration predictor + length regulator, designed in `vrutta_conditioning_arch.md`, gated behind a go/no-go probe, currently unbuilt).

---

## 15. MBTN PRODUCTION (shipped 2026-06-24)

- **Run:** `marathon_serial.sh 1 32` (sequential, 4-GPU), ~13 h, 32/32 clean. Cadence: full overnight batch → async morning QC via REVIEW sheets + surgical swaps.
- **Output:** 32 videos, `MBTN_videos/Final_production/` named `Mahabharata Tatparya Nirnaya — Adhyaya NN | Sri Madhvacharya.mp4`, 2.3 GB, **17 h 34 m**.
- **Post-ship fixes (all surgical swaps):** adh24 v44 recovered (split-file merge); 5 dup renumberings (adh11/13/21/25); adh17 v119 merged into a 4-hemistich verse (drove the N-hemistich card renderer); adh02 v160 collapse → pāda-split (adh21 v135 accepted); 6 editorial-bracket verses (adh01/02/03/10) → build strips `(…)`, adh10 v53 meter corrected anuṣṭubh→mālinī; adh07 v31 garble re-rendered (seed 66); adh01 v18 visarga "haha" → per-clip echo. Source numbering quirks logged in `MANIFEST_ISSUES.md`.
- **YouTube fact:** cannot replace a published video's file in place — a content fix = new upload (new URL) + playlist swap. So catch issues before publishing (soft-launch unlisted) when possible.
- IAST captions = backlog "add on demand" (non-destructive `.vtt` from `segments.json` + Deva→IAST).

---

## 16. ZERO-SHOT VOICE CLONING

Established via a teacher's test (clone his `sri_ref.wav` voice onto a Vāyustuti). See `Desktop/inference_test/`.
- **steered model** → imposes Prathosh's voice (not zero-shot).
- **base IndicF5** → **true zero-shot voice clone** of an unseen voice. Fix schwa via Kannada routing; prosody is flat (base, no chant).
- **armA champion** → has chant swara from the reference but is **voice-locked** (garbles a truly OOD voice).
- **Conclusion:** zero-shot **voice ✓**; zero-shot **voice + our chant ✗** (chant lives in the voice-locked champion; vocoder/reference swaps can't add prosody). Real path = a **voice-agnostic chant fine-tune** (train chant on many reciters).
- **Recipe (works today):** base IndicF5 DiT + Kannada routing + our BigVGAN + short chanted reference (≤7s, daṇḍa-ended) + split long target at pāda/daṇḍa.
- **Lesson:** the TTS is faithful — **proofread the source** (a `मर्णिमे`/`मणिर्मे` repha-transposition and a joined `उद्यद्विद्युत्प्रचंडां` both came straight through).

---

## 17. ŚRĪMAD BHĀGAVATAM — next project

- **Scope:** ~18,000 verses, 12 skandhas / ~335 adhyāyas. **AUDIO ONLY** (app) — drop the entire video half.
- **Drop MFA** (§12 retrospective). QC = duration gate + ear-spot + **structural/source validation** (the dominant defect source).
- **Pipeline:** source → structural-validate → `build_clips` → `render_batch` (4-GPU) → duration-QC → package.
- **App deliverable defaults (agreed):** per-**verse** wav · **dry masters + optional tanpura-mixed** · **Opus/m4a + WAV masters** · **JSON manifest** (skandha/adhyāya/verse → file, deva, kannada, meter, duration).
- **Gadya (prose) — the hard case** (design `Bhagavatam/audit/GADYA.md`; samples `sessions/malini_fidelity/gadya_test/`). **★ KEY CORRECTION (2026-06-24):** this source's `is_padya` flag means "carries a verse number," NOT "metrical" — the long Skandha-5 bhū-gola prose is stored as `is_padya:true` (e.g. S5 A20 v2 = 154 syll). **Gadya = blocks that FAIL meter detection**, not `is_padya:false` (the 1,374 `is_padya:false` are tiny "śuka uvāca—" connectives, median 6 syll, render as-is). Real prose set ≈ **290 blocks (S5=271)** → chunker yields **1,629 chunks** (avg 5.6/blk) with **106 monster compounds** (>22 syll no-space, worst 68). **Chunk rule LOCKED (ear pass 2026-06-24, A/B `audit/gadya_ab_listen/`):** strip colophon+`(…)` → hard-break at `, । ॥` → pack WHOLE words to TARGET=20 syll, merge trailing <8. **★ chunk boundaries fall ONLY on safe word boundaries — a word is NEVER split** ("split mid-word is wrong"); akṣara force-split REMOVED. **A space is NOT a safe break when the next token is vowel-initial** (de-sandhi'd junction, e.g. source `प्लक्ष उत्थितो`=प्लक्षोत्थितो — caught by ear, was 11% of boundaries) **or the prev token ends in a hyphen** (compound continuation); punctuation breaks before a vowel are kept. An over-20-syll single compound = its own intact chunk rendered whole (max 68 syll ≈18s, within F5 range — `g_b3_whole` accepted). **★ Break-point selection PREFERS a STRONG word-final** (visarga ः/anusvara ं/halant ्/long-vowel = complete word); a bare-inherent-'a' ending (सप्त, प्लक्ष) is a bound modifier → weak fallback only (ear: `…द्वीपं | सप्त वर्षाणि…` not `…सप्त | वर्षाणि…`). Result: 92% strong breaks, 0 vowel-junctions. Render: ref `gadya_mbtn` · **sps 0.26 (fix_duration)** · **gap 0.55**. ⚠ **sps 0 clips the final syllable** (F5 under-allocates the last vowel's decay → `…उपरराम` cut); 0.26 gives tail room, no parked silence. Verified 0 mid-word splits across all 290 prose blocks. Gadya fully resolved → clear to pilot.
- **Carry-over fixes:** strip `(…)` parentheticals; deergha-ṝ; the reference/half-reference rules.
- **★ Critical first step:** point the renderer at the Bhāgavatam source → run a **structural audit (completeness/numbering/meters) BEFORE rendering anything** → pilot one skandha.
- **★ PHASE 0 DONE (2026-06-24):** `Bhagavatam/audit/extract_audit.py` extracts the **16,017** `content_type:"Bhagavatam"` blocks (14,643 padya · 1,374 gadya · 1,397 verse=null) across 12 skandhas / 345 adhyāyas → `audit/verses.jsonl` (one row per rendering unit, collision-free seq-based `out_id`/`out_wav` `BhP_SS.AAA.NNN.wav`, 30,284 text lines ≈ hemistichs) + `audit/ISSUES.md` + `audit/SUMMARY.json`. **Source is exceptionally clean: 0 empty-text blocks → no audio unit is missing.** `verse` field verified trustworthy (0 skandha/adhyāya/verse mismatches vs embedded `s/a/v` colophon). Only-issues, all benign/cosmetic for an audio app: **23 unnumbered padya** (2-part `॥ s/a॥` colophon, verse text present but number elided — render fine, seq-id used), **13 verse-number gaps** (source editor's numbering, text contiguous, no missing audio), **1 dup** (S10 A100 v44 ×2 — renumber candidate, also gap at 43), **4 parentheticals** (strip-rule cases: `(सलिलस्य)`,`(ऽ)`, two S12 A12 anukramaṇikā glosses), **20 verse-range blocks** (one block = 2–3 verses, `verse_label` like `14-16`), ZWJ pervasive but legitimate.
- **★ PHASE 1 DONE (2026-06-25):** `audit/meter_census.py` (reuses build_mbtn_adh_v2 L/G + bank sigs) → `meter_census.jsonl`+`METER_CENSUS.md`. Classes: **VERSE 14,042 · PROSE 559 · SHORT 1,416** (SHORT=tiny connectives→single clip). Meters: anuṣṭubh 11,464, vasantatilakā 649, jagatī-upajāti 382, vaṃśastha 371, upajāti 337, etc. Mixed-vṛtta 1,348 (per-hemi refs). Detector fixes added: 12-syll jagatī-upajāti (vaṃśastha+indravaṃśā mix) + leading-pāda fallback for odd-syll scan miscounts (recovered ~300 verses from PROSE). **Coverage gap:** ~185 ardhasama verses (puṣpitāgrā/aupacchandasika, 12+13 hemis, e.g. Bhīṣma-stuti 1.9.39-46) — bank has NO ardhasama ref → **decision: nearest-length fallback ref for now** (defer recording a real ref). anuṣṭubh ref = `anu_v094.wav`, IDENTICAL to MBTN (verified in render logs).
- **★ PRONUNCIATION (ऋ), RENDER-ONLY:** vocalic ऋ kept everywhere EXCEPT **हृ→ह्रु when before a conjunct** (हृद्य→ह्रुद्य); bare हृ stays vocalic. (blanket ृ→्रु and हृ-everywhere both ear-rejected.) Display/manifest keep original text. `build_skandha.py:name_fix()`.
- **★ SEED-STOCHASTIC ARTIFACTS:** F5 randomly inserts mid-hemistich pauses (~0.3s) and consonant gemination (munik-**k**rute) — verse+seed dependent (MBTN hid these via seed60+attention-retry). QC stack: **duration gate (collapses) + internal-silence gate (pauses) → 3-seed best-of-N reseed [61,63,64] of flagged (~4.5%)**; **ASR-CER (IndicConformer-Sa) is the ONLY detector for gemination/drops/mispronunciation** (phase-2, env WIP — see `audit/pilot/ASR_FIX.md`). MFA stays dropped.
- **★★ FULL RENDER MARATHON RUNNING (2026-06-25, ~tonight):** `audit/marathon_full.sh 1 12` in tmux `bhag_chain` on user, 4-GPU (user sh0,1 + GPU host B sh2,3), seed 65, ~18-20h. Per-skandha: build(`build_skandha.py`)→render→detect/reseed→`assemble_skandha.py`. **Deliverable:** `~/Prathosh/bhag_full/audio/skandha_NN/adhyaya_AAA/BhP_SS.AAA.NNN.wav` (1:1 w/ text) + per-adhyaya `timings.json` (karaoke hemistich start/end) + per-skandha `manifest.json` (deva+kannada+segments+splits). Resumable (per-skandha DONE markers); local QC player = `audit/pilot/listen.html` (serve via `python3 -m http.server`, NOT file://).

---

## 18. DISTILLED KEY LEARNINGS

1. **Architecture journey:** StyleTTS2/VITS2/Matcha all hit a conjunct/prosody ceiling; **IndicF5 (Kannada-routed) cleared it at 4.6 MOS** with a 5h clone — *data wasn't the bottleneck, the backbone was.*
2. **F5 trade-off:** perfect content, locked prosody. Don't fight it text-side (conditioner is inert); steer the voice + supply prosody via the reference.
3. **Reference is everything** — voice, swara, pace; governed by the half-reference rule (exact ref_text↔audio span) and exact per-pāda L/G matching.
4. **Vocoder = artifact layer:** BigVGAN-v2 mandatory (vocos = long-vowel phase shiver). But wobble/jitter lives in the **mel/generator**, not the vocoder (fix via cfg, conditioning, best-of-N).
5. **Real defects come from the SOURCE, not the model** — numbering dups/gaps, merged entries, brackets, ZWJ, transcription typos. The TTS is faithful; validate source first.
6. **MFA QC was low-signal at scale** (~0 real of ~1,648 flags); the cheap duration gate + ear caught the real ones.
7. **Data-prep landmines:** the Audacity lead-in offset (constant, not linear), word-gap → "space=pause" learning, 15s reference cap, /tmp checkpoint fills.
8. **Modular video + surgical swap** = async QC is viable (fix one shloka without re-rendering an adhyāya).

---

## 19. LOCKED INFERENCE PARAMETERS (as-built)

| param | batch (`render_batch`, MBTN) | champion (`render.sh`) | established |
|---|---|---|---|
| architecture | IndicF5 DiT (dim1024/depth22/heads16, 337M) | same | E29–E32 |
| voice | `voice_steer_ema_2026-06-17.pt` | `voice_armA_ema_2026-06-11.pt` | E79 / E62 |
| vocoder | BigVGAN-v2 EMA (`voc_bigvgan_EMA_2026-06-11.pth`) | same | E59/E63/E76 |
| mel | vocos 100-band 24kHz | same | E76 |
| solver / nfe | euler / 64 | euler / 64 | E64/E67 |
| cfg_strength | **3.0** ⚠ | **1.2** ⚠ | E67 / E76 |
| sway_sampling_coef | −0.7 | −0.7 | E64 |
| seed | 60 (+att) | 50 | marathon / E64 |
| speed | 0.90 | 0.90 | E31 |
| sec_per_syll | per-meter from bank (`-1`); anuṣṭubh 0.326 | same | E80b/§10 |
| gap / gap_halant | 0.55 / 0.20 | 0.30 / 0.20 | E80h |
| conditioner | NONE | NONE | E59/E68/E78 |
| post | head/tail gate + compress_sil + daṇḍa breath + tanpura(Sa 170) | head/tail gate + 0.30s gap | E46/E61/E80 |

---

## 20. BACKLOG / OPEN PROBLEMS

- **Prosody design** (verse-independent contour) — needs explicit Arch-B (duration predictor + length regulator) or a different backbone; designed, unbuilt.
- **Voice-agnostic chant** (zero-shot chant in any voice) — needs a multi-reciter chant fine-tune.
- **Repeated-syllable depth >4** (`ಗಗಗಗಗಗ`) — unrecoverable (~4 cap) even with priming.
- **Pāda-end vocal fry** on some renders (amplitude shimmer) — open.
- **F0-descent syllables** occasionally swallowed.
- **Long-vowel bare visarga "haha"** on rare verses (handled per-clip).
- **25h Prathosh corpus** — recording incomplete (~3h done); paused (4.6 MOS from 5h made it lower priority).
- **Bhāgavatam gadya** parameters (VT vs gadya ref, chunk/gap) — open, needs ear.
- **Source data quality** — the 14 MBTN boundary verses pattern (dups/gaps/merges) will recur at 3.5× scale in Bhāgavatam.
- **MANIFEST_ISSUES.md** boundary verses — fix at review.
- **Public release** (decided 2026-06-24, see §1) — two-case-study report (MBTN + Bhāgavatam) + frontend code + weights + samples, on HF/arXiv/blog. Gating work: write Bhāgavatam app first; then draft scrubbed/anonymized; clear base-model licenses (NVIDIA BigVGAN-v2, IndicF5); confirm source-text edition provenance; add intended-use/ethics note.

---

## APPENDIX A — COMPLETE FILE & PATH INDEX

### GPU host `$PROD/`
- `production/` — `render_batch.py`, `render_production.py`, `render.sh`, `assemble_modular.py`, `build_clips_gen.py`, `qc_loop_gen.py`, `swap_shloka.sh`, `run_adhyaya.sh`, `marathon_serial.sh`, `consolidate_log.sh`, `colophon_gen.py`, `prose_gen.py`, `spotcheck_report.py`, `turing_render.sh`, `mfa_prep.py`, `mfa_flag.py`, `README.md`, `reference_bank/bank.json` (+wavs).
- `prep_text.py` (frontend). `CHAMPION_2026-06-11/` (gold models + MANIFEST.md + CHECKSUMS.md5). `idf5_kannada_FINAL.pt`, `armA_voice.pt`, `prathosh_champion_styleA.pt`, `gana_ckpt/`, `armB_mirror/`.
- `prathosh_new/` (style-B databank: `wavs/ds__*.wav`, `prathosh_new_manifest.plain.jsonl`, `vq_qc.json`, `prosody_bank_new.json`, `prathosh_new_dict.dict`), `prathosh_mfa_new/prathosh_new_adapted.zip`.
- `sanskrit-tts/` — `data/prathosh/`, `model/{StyleTTS2,VITS2,Matcha-TTS}/`, `mfa_work/{corpus,sanskrit_dict.txt,anu_acoustic.zip,aligned/pilot_reciter/*.TextGrid}`, `data/styletts2_data/{mfa_durations,conformer_alignments}/`, `scripts/`, `corpora/local_sanskrit/mbtn_split.txt`.
- HF cache: `~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/` (model.safetensors, checkpoints/vocab.txt).
- MBTN data: `$BIGDISK/mbtn_prod/` (`manifests/`, `work/adhNN/`, `videos/`, `colophons/`, `MARATHON_*.txt`, `MANIFEST_ISSUES.md`).

### GPU host B `$PROD_B/`
- `production/` (synced render deps), `CHAMPION/`, `mbtn_work/adhNN/`, `mbtn_videos/`, `pylibs/bigvgan`, `.cache/huggingface/`. env `/home/prathosh/miniconda3/envs/indicf5`.

### Local Mac `$REPO/`
- **This report:** `MASTER_TECH_REPORT.md`. **Full experiment ledger:** `EXPERIMENTS_DIGEST.md`.
- **⚠ Layout change 2026-06-24 (consolidation done):** the scattered source `.md` docs listed below were **moved into `_archive_pre_consolidation/`** (same relative paths preserved). A raw structure-preserved backup of **every** project `.md` (33 files) is in `_all_md_dump/`. So paths like `Final_Files/production.md` or `sessions/malini_fidelity/sessions_todays_data.md` now live under `_archive_pre_consolidation/…`. The non-`.md` assets (scripts, JSON, audio, video, venv) stayed in place; only the markdown docs moved.
- `Final_Files/` — `production.md`, `PRODUCTION_REFERENCE.md`, `experiment_render.md`, `Scripts/` (`PIPELINE.md`, `stage{1..7}_*.py`, `process_session.py`, `tts_*.py`, `render_dataset*.py`, `render_card.py`, `make_adhyaya_video.py`, `sanskrit_tts_design.md` [old]), `reference_bank_final/bank.json`, `CHAMPION_2026-06-11/MANIFEST.md`, `Prathosh_data/` (style A: DATA_PREP_PIPELINE.md, clips24/, aligned_qc/, dgt_v2.json), `Prathosh_data_new/` (References/, Experiments.md [old fork], mbtn_record_shlokas.json, clips_manifest.json), `Anuvyakhyana/` (SESSION_CONTEXT.md, Audio/, TS/, Text/, output/{TTS,YouTube,MP4}), `Listen/`, `PRODUCTION_REFERENCE.md`.
- `Sarvamoola/` — `sanskrit_tts_design.md` [current], `sessions/` (`Experiments.md` [master E0–E80h], `2026-05-30_recording_strategy.md`, `recording_corpus_design.md`, `2026-05-30_pilot_inference_findings.md`, `2026-05-31_vits2_mfa_setup.md`, `bigvgan_integration_plan.md`, `vrutta_conditioning_arch.md`, `2026-06-03_pilot-complete-data-pivot.md`), `scripts/` (`prep_text.py`, `chandas_labeler.py`, `gana_f5.py`, `gana_uni.py`, `extract_prosody.py`, `build_swara_lookup.py`, `extract_swara_templates.py`, `build_combined.py`), `Texts/` (`mbtn.json`, `mbtn_split.json`, `sumadhvavijayah_mula.json`, `sumadhvavijayah.json`, `vayustuti.json`, `recording_manifest.json`, `recording_script.tsv`, `recording_sheet_vruttas.md`), `production_assets/mrstft_v2_annealed/`, `fractal-engine/` [SEPARATE non-TTS tool — exclude].
- `MBTN_videos/Final_production/` (32 final mp4s), `MBTN_videos/MARATHON_LOG.md`.
- `sessions/malini_fidelity/` (MBTN pipeline scripts + `SESSION_2026-06-20.md`, `sessions_todays_data.md`, `gadya_test/`, `build_mbtn_adh_v2.py`).
- `pilot_dip/` (`chandas_labeler.py`, venv). `~/Desktop/inference_test/` (zero-shot test artifacts).

---

## APPENDIX B — EXPERIMENT INDEX (E0–E80h, one-liners)

E0 StyleTTS2 pilot MOS~3.0 · **E1 v0_mfa MOS4.2+ (PL-BERT+MFA+warmstart)** · E2 PAD (no change) · E3 cluster-scale (worse) · E4 MFA conjunct audit · E5 decoder-only FT (no gain) · E6 IndicConformer align · E7 conformer-dur FT (junk) · E8 pad-silence (killed). **E9 VITS2+PLBERT killed · E10/B1 VITS2+BigVGAN killed · E15 PLBERT-v2 stopped · E16 seed-explosion · E17 muffled=data · E18 DDP.** E19 Matcha scaffold · E20 vocoder bakeoff · E21 7-issues MOS4.2 · E22 Rung1 phase-2 voc · **E23 MR-STFT wins** · E24 Turing reframe+25h plan · E25 corpus audit · E26 chandas prior · E27 manifest 12,487 · E28 ठ→ट. **E29 IndicF5 zero-shot+GRN bug · E30 Kannada vs SLP1 · E31 long-verse split · E32 ★4.6 MOS clone · E33 vrutta pivot · E34 3 tells · E35 ref-prosody 1.6× · E36 vrutta arch · E37 cfg1.2/nfe64 · E38 gaṇa labeler · E39 GaṇaDiT · E40 phase-1 · E41 implicit INERT · E42 d_gt · E43 durpred · E44 BigVGAN-f5 · E45 recon · E46 head/tail gate · E47 Prathosh 3h · E48 prep_text · E49 FT launch · E50 data-fix saga · E51 2-path · E52 durpred v2 · E53 base-voice/wobble=mel · E54 swara works · E55 generalization · E56 unified cond · E57 any-style + wobble=generator.** E58 ArmA/B+no-pada-split · E59 vocoder fix/conditioner dropped · E60 deploy · E61 breath/pause · **E62 ★GOLD snapshot** · E63 vocoder locked · E64 serving config · E65 per-pāda L/G ref · E66 tempo · **E67 cfg=3.0** · **E68 swara NEGATIVE** · E69 visarga norm · E70 ×3 half-ref · E71 CosyVoice3 (dropped) · E72–E75 Indic-Parler (dropped) · **E76 BigVGAN-v2 phase fix LOCKED** · E77 prosody wall · E78 B-route dead · **E79 ★voice-steering SHIPPED + half-ref rule** · E80–E80h repeat-prime/chorus/tanpura/gap/visarga/satva.

---

## APPENDIX C — GLOSSARY

**vṛtta** meter · **gaṇa** L/G (laghu/guru) pattern · **laghu/guru** short/long syllable · **swara** melodic note/contour · **daṇḍa** verse punctuation (`।` half, `॥` full) · **yati** caesura · **pāda** quarter-verse · **hemistich** half-verse (2 pādas) · **saṃhitā** sandhified continuous text · **sandhi** euphonic joining (utva/rutva/lopa/satva) · **visarga** ः · **anusvāra** ं · **jihvāmūlīya/upadhmānīya** visarga allophones before k/p · **pramāṇa** scriptural citation · **SLP1** ASCII Sanskrit transliteration · **MFA** Montreal Forced Aligner · **CFM/OT-CFM** (optimal-transport) conditional flow matching · **DiT** diffusion transformer · **MOS** mean opinion score · **HNR** harmonics-to-noise ratio · **EMA** exponential moving average (model soup) · **MBTN** Mahābhārata Tātparya Nirṇaya · **SMV** Sumadhvavijaya.

---

## APPENDIX D — SOURCE DOCUMENTS CONSOLIDATED (this report replaces these for reading)

**DONE 2026-06-24:** originals were verified and **moved to `_archive_pre_consolidation/`** (relative paths preserved); a complete raw backup is in `_all_md_dump/`. The paths below are therefore now under `_archive_pre_consolidation/…`. **Authoritative master experiment log = (ex-)`Sarvamoola/sessions/Experiments.md` (E0–E80h); the `Final_Files/Prathosh_data_new/Experiments.md` is a stale subset (E0–E57)** — both now in the archive; the live distilled version is `EXPERIMENTS_DIGEST.md`.

Consolidated: `Sarvamoola/sanskrit_tts_design.md` (current) + `Final_Files/Scripts/sanskrit_tts_design.md` (old); both `Experiments.md`; `Final_Files/production.md`; `Final_Files/PRODUCTION_REFERENCE.md`; `Final_Files/experiment_render.md`; `Final_Files/Scripts/PIPELINE.md`; `Final_Files/Prathosh_data/DATA_PREP_PIPELINE.md`; `Final_Files/CHAMPION_2026-06-11/MANIFEST.md` (+ user copy); `Sarvamoola/sessions/{2026-05-30_recording_strategy, recording_corpus_design, 2026-05-30_pilot_inference_findings, 2026-05-31_vits2_mfa_setup, bigvgan_integration_plan, vrutta_conditioning_arch, 2026-06-03_pilot-complete-data-pivot}.md`; `Sarvamoola/Texts/recording_sheet_vruttas.md`; `Final_Files/Anuvyakhyana/SESSION_CONTEXT.md`; `sessions/malini_fidelity/{SESSION_2026-06-20, sessions_todays_data}.md`; `MBTN_videos/MARATHON_LOG.md`; user `production/README.md`. **Excluded** (not TTS): `Sarvamoola/fractal-engine/*`, venv/site-packages.

_End of report._
</content>
</invoke>
