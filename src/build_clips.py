"""Expand mbtn_adh01_manifest.json -> per-hemistich QC clips + 4 round-robin shards.
clip: {id, verse, hi, meter, padas:[deva], text, seed, no_sandhi}
no_sandhi = NOT has_citation (citations get sandhi; normal verses already sandhified)."""
import json, sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NSHARD = 4
m = json.load(open("mbtn_adh01_manifest.json", encoding="utf-8"))
clips = []
for v in m:
    no_sandhi = not v.get("has_citation")
    for hi, hemi in enumerate(v["hemis_deva"]):
        clips.append({"id": f"{v['id']}_h{hi}", "verse": v["id"], "hi": hi,
                      "meter": v["meter"], "padas": [hemi], "text": hemi,
                      "seed": SEED, "no_sandhi": no_sandhi})
json.dump(clips, open("adh01_clips.json", "w"), ensure_ascii=False, indent=1)
shards = [[] for _ in range(NSHARD)]
for i, c in enumerate(clips):
    shards[i % NSHARD].append(c)
for s in range(NSHARD):
    json.dump(shards[s], open(f"adh01_qc_shard{s}.json", "w"), ensure_ascii=False, indent=1)
nc = sum(1 for v in m if v.get("has_citation"))
print(f"verses={len(m)} clips={len(clips)} citation_verses={nc} "
      f"shard_sizes={[len(s) for s in shards]} seed={SEED}")
