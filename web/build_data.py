"""DAE list (notebook 03 output) -> docs/dae.json for the KrishiGuard site."""
import json, re, pandas as pd

d = pd.read_csv("data/dae_registered_pesticides.csv", dtype=str).fillna("")
d = d.apply(lambda c: c.str.strip())
NOT_CROP = re.compile(r"Ltd|Limited|Company|Corporation|Enterprise|Pvt|Industr|Co\.|International|Chemical|Agro|Products|Works|Kit|Care|Seed|Vet|Biotech|Oil|Intefa|^\W*$", re.I)
PUBLIC_HEALTH = re.compile(r"mosquito|cockroach|house ?fly|bed ?bug|flea|lice|ant\b", re.I)
FIX = {"Chili": "Chilli", "Swee gourd": "Sweet gourd", "Sweet guard": "Sweet gourd", "Cucurbits": "Cucurbit",
       "Country Bean": "Country bean", "Soyabean": "Soybean", "Bettle Vine": "Betel vine", "Ground nut": "Groundnut"}
d["crop"] = d.crop.replace(FIX)
keep = ~d.crop.str.contains(NOT_CROP) & ~d.pest.str.contains(PUBLIC_HEALTH) & d.brand.ne("")
print(f"{len(d)} rows, keeping {keep.sum()} farm rows ({(~keep).sum()} public-health or broken-crop rows dropped)")
d = d[keep]
ings = sorted(set(d.common_name)); holders = sorted(set(d.holder))
rows = [[r.brand, ings.index(r.common_name), r.reg_no, holders.index(r.holder), r.crop, r.pest, r.dose] for r in d.itertuples()]
out = {"source": "DAE list of registered agricultural pesticides, up to the 81st PTAC meeting (parsed in KrishiGuard notebook 03)",
       "ingredients": ings, "holders": holders, "rows": rows,
       "stats": {"rows": len(rows), "products": d.brand.nunique(), "ingredients": len(ings)}}
json.dump(out, open("docs/dae.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(out["stats"])
assert any(r[0] == "Taba 25WDG" and r[4] == "Rice" and r[5] == "BPH" and r[6] == "100 g/ha" for r in rows)
