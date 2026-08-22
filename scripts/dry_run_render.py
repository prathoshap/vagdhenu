# -*- coding: utf-8 -*-
"""Dry-run / noop render planner CLI (no GPU, no weights, no audio).

Examples:
  python scripts/dry_run_render.py --text "वसुदेवसुतं देवं ..."
  python scripts/dry_run_render.py --shard examples/vedic_four_vedas_shard.json -o /tmp/plan.json
  python scripts/dry_run_render.py --veda-samples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import limits
from render_plan import plan_render, DEFAULT_BANK

VEDA_SAMPLES = [
    {
        "id": "rv_1_1_1",
        "veda": "Rigveda",
        "meter": "__auto__",
        "text": (
            "अ\u0952ग्निमी\u0951ळे पु\u0952रोहि\u0951तं य\u0952ज्ञस्य\u0951 "
            "दे\u0952वमृ\u0951त्विजम् । हो\u0952तारं रत्नधा\u0951तमम्"
        ),
    },
    {
        "id": "yv_shanti",
        "veda": "Yajurveda",
        "meter": "__auto__",
        "text": (
            "ॐ\u0951 श\u0952ं नो मि\u0951त्रः श\u0952ं वरु\u0951णः । "
            "श\u0952ं नो भवत्वर्य\u0951मा । श\u0952ं न इ\u0951न्द्रो "
            "ब\u0952ृह\u0951स्पतिः । श\u0952ं नो विष्णुरुरुक्र\u0951मः"
        ),
    },
    {
        "id": "sv_agni",
        "veda": "Samaveda",
        "meter": "__auto__",
        "text": (
            "अ\u0952ग्न आ या\u0951हि वी\u0952तये\u0951 गृ\u0952णा\u0951नो "
            "ह\u0952व्यदा\u0951तये । नि होता सत्सि बर्हिषि"
        ),
    },
    {
        "id": "av_1_1_1",
        "veda": "Atharvaveda",
        "meter": "__auto__",
        "text": (
            "ये\u0952 त्रिष\u0951प्ताः प\u0952रिय\u0951न्ति वि\u0952श्वा "
            "रू\u0951पाणि बिभ्रतः । वा\u0952चस्पतिर्बला\u0951 तेषां "
            "तन्वो अद्य दधातु मे"
        ),
    },
]


def _plan_one(text, meter, no_sandhi, bank, seed):
    err = limits.validate_one_shloka(text)
    plan = plan_render(
        text, meter=meter, no_sandhi=no_sandhi, bank_path=bank, seed=seed,
    )
    if err:
        plan["ok"] = False
        plan["validation_error"] = err
    warn = limits.vedic_noop_warning(text)
    if warn and warn not in plan["warnings"]:
        plan["warnings"].append(warn)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vāgdhenu dry-run/noop text frontend planner")
    ap.add_argument("--text", help="single verse (any Indic script, accents OK)")
    ap.add_argument("--shard", help="JSON list of {id,text|padas,meter?} entries")
    ap.add_argument("--veda-samples", action="store_true", help="run built-in 4-Veda samples")
    ap.add_argument("--meter", default="__auto__")
    ap.add_argument("--sandhi", action="store_true", help="enable visarga sandhi (default off)")
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("-o", "--out", help="write full JSON manifest(s) here")
    ap.add_argument("--write-shard", action="store_true",
                    help="with --veda-samples, refresh examples/vedic_four_vedas_shard.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    jobs = []
    if args.text:
        jobs.append({"id": "stdin", "text": args.text, "meter": args.meter})
    if args.shard:
        raw = json.loads(Path(args.shard).read_text(encoding="utf-8"))
        for row in raw:
            text = row.get("text") or " । ".join(row.get("padas") or [])
            jobs.append({
                "id": row.get("id", f"row{len(jobs)}"),
                "text": text,
                "meter": row.get("meter", args.meter),
                "veda": row.get("veda"),
            })
    if args.veda_samples:
        jobs.extend(VEDA_SAMPLES)
    if not jobs:
        ap.error("provide --text, --shard, and/or --veda-samples")

    reports = []
    n_ok = 0
    for job in jobs:
        plan = _plan_one(
            job["text"], job.get("meter") or args.meter,
            no_sandhi=not args.sandhi, bank=args.bank, seed=args.seed,
        )
        plan["id"] = job["id"]
        if job.get("veda"):
            plan["veda"] = job["veda"]
        reports.append(plan)
        if plan.get("ok"):
            n_ok += 1
        if not args.quiet:
            tag = "OK" if plan.get("ok") else "FAIL"
            m = plan["meter"]
            print(
                f"[{tag}] {plan['id']}: syll={plan['total_syllables']} "
                f"padas={plan['n_padas']} vedic={plan['vedic_input']} "
                f"meter={m['resolved']} ({m['source']}) "
                f"accents_drive_audio={plan['accents_drive_audio']}"
            )
            for w in plan.get("warnings") or []:
                print(f"       ! {w}")
            for i, p in enumerate(plan["pieces"]):
                marked = p["n_accents_marked"]
                print(
                    f"       pāda{i}: syll={p['n_syll']} marked={marked} "
                    f"align={p['align_ok']} kn={p['kannada'][:48]}"
                    f"{'…' if len(p['kannada']) > 48 else ''}"
                )

    doc = reports[0] if len(reports) == 1 else {"noop": True, "n": len(reports), "plans": reports}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if not args.quiet:
            print(f"wrote {args.out}")

    # Optional refresh of the checked-in four-veda shard (off by default — no surprise diffs)
    if args.veda_samples and args.write_shard:
        shard_path = ROOT / "examples" / "vedic_four_vedas_shard.json"
        shard_path.write_text(
            json.dumps(
                [{"id": s["id"], "veda": s["veda"], "meter": s["meter"],
                  "padas": [p for p in s["text"].replace("॥", "।").split("।") if p.strip()],
                  "text": s["text"], "noop": True}
                 for s in VEDA_SAMPLES],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"wrote {shard_path}")

    print(f"dry-run {n_ok}/{len(reports)} ok (noop; no audio rendered)")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
