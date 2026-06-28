import json, os
b=json.load(open("reference_bank/bank.json",encoding="utf-8"))
print("=== bank refs ===")
for k,v in b.items():
    if isinstance(v,dict) and "wav" in v:
        print("%-30s class=%-8s mode=%-6s %s" % (k, str(v.get("class")), str(v.get("mode")), v.get("wav")))
print("\n=== manifest verse keys ===")
m=json.load(open("mbtn_adh01_manifest.json",encoding="utf-8"))
print("keys:", list(m[0].keys()))
print("n verses:", len(m))
# any colophon-ish field?
for kk in ("colophon","title","adhyaya_title","end","iti"):
    if kk in m[0]: print("HAS", kk, m[0][kk])
print("\n=== find mbtn_split.json on box ===")
