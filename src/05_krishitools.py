# %% [markdown]
# # KrishiGuard 05: KrishiTools-300
#
# 300 Bangla farmer questions that need the pesticide tool, built from the DAE list of registered agricultural pesticides (notebook 03 output).
# - dose (100): a registered product, crop and pest; the gold answer is the registered dose.
# - product (100): a crop and pest; any product registered for that pair is correct.
# - registration (100): a real product asked about for a crop it is not registered for; the answer must say no and give no dose.
#
# The questions come from templates, so they test grounding, not language variety. The gold answers are whatever the DAE list says, including its known errors (Blastin and Hexythiazox listed for rice BPH).
#
# Kaggle settings: CPU, Internet off is fine, input: KrishiGuard 03 output.

# %%
import os, re, glob, json, random
import pandas as pd

OUT = "/kaggle/working/krishiguard"
os.makedirs(OUT, exist_ok=True)
SEED, N_PER_TYPE = 42, 100
dae = pd.read_csv(glob.glob("/kaggle/input/**/dae_registered_pesticides.csv", recursive=True)[0]).fillna("")

# %% [markdown]
# ## Crop and pest names
# Only crops and pests with a common Bangla name are used. Each Bangla pest name contains the keyword the agent's tool looks for.

# %%
CROP_BN = {"rice": "ধান", "wheat": "গম", "potato": "আলু", "brinjal": "বেগুন", "chil": "মরিচ", "tomato": "টমেটো",
           "jute": "পাট", "maize": "ভুট্টা", "mustard": "সরিষা", "mango": "আম", "tea": "চা", "onion": "পেঁয়াজ",
           "bean": "শিম", "cabbage": "বাঁধাকপি", "cucumber": "শসা", "cotton": "তুলা"}
# most specific English names first, so "shoot and fruit borer" wins over "fruit borer"
PEST_BN = [("shoot and fruit borer", "ডগা ও ফল ছিদ্রকারী পোকা"), ("fruit and shoot borer", "ডগা ও ফল ছিদ্রকারী পোকা"),
           ("late blight", "নাবি ধসা (মড়ক) রোগ"), ("early blight", "আগাম ধসা রোগ"), ("sheath blight", "খোলপোড়া রোগ"),
           ("stem borer", "মাজরা পোকা"), ("bph", "বাদামি গাছফড়িং (কারেন্ট পোকা)"), ("leaf roller", "পাতা মোড়ানো পোকা"),
           ("leaf folder", "পাতা মোড়ানো পোকা"), ("blast", "ব্লাস্ট রোগ"), ("aphid", "জাব পোকা"), ("whitefly", "সাদা মাছি"),
           ("white fly", "সাদা মাছি"), ("thrips", "থ্রিপস"), ("mite", "মাকড়"), ("fruit borer", "ফল ছিদ্রকারী পোকা"),
           ("rice bug", "গান্ধি পোকা"), ("armyworm", "লেদা পোকা"), ("army worm", "লেদা পোকা"), ("cutworm", "কাটুই পোকা"),
           ("powdery mildew", "পাউডারি মিলডিউ রোগ")]

def crop_key(crop):
    c = str(crop).strip().lower()
    c = "chil" if c in ("chilli", "chili") else c
    return c if c in CROP_BN else None

def pest_key(pest):
    p = str(pest).lower()
    bn = next((b for en, b in PEST_BN if en in p), None)
    return next(en for en, b in PEST_BN if b == bn) if bn else None  # spelling variants (white fly, whitefly) share one key

dae["crop_k"], dae["pest_k"] = dae.crop.map(crop_key), dae.pest.map(pest_key)
usable = dae.dropna(subset=["crop_k", "pest_k"]).drop_duplicates(["reg_no", "crop_k", "pest_k"])
usable = usable[usable.dose.str.contains(r"\d")]
print(len(usable), "usable rows;", usable.groupby(["crop_k", "pest_k"]).size().sort_values(ascending=False).head(15).to_dict())

# %% [markdown]
# ## Build the questions

# %%
rng = random.Random(SEED)
PEST_NAME = dict(PEST_BN)
T_DOSE = ["{crop} ফসলে {pest} দমনে {brand} কী মাত্রায় ব্যবহার করব?",
          "{brand} দিয়ে {crop}-এর {pest} দমন করতে চাই, কতটুকু দিতে হবে?",
          "ভাই, {crop} খেতে {pest} লাগছে, {brand} কতখানি দিমু?"]
T_PRODUCT = ["{crop} ফসলে {pest} দমনের জন্য কোন বালাইনাশক নিবন্ধিত আছে?",
             "{crop}-এ {pest} এর আক্রমণ হয়েছে, কোন ওষুধ দেওয়া যায়?",
             "আমার {crop} খেতে {pest} লাগছে, কী স্প্রে করমু?"]
T_REG = ["{brand} কি {crop} ফসলে {pest} দমনে ব্যবহার করা যাবে?",
         "{crop}-এর {pest} দমনে {brand} দিলে হবে তো? কতটুকু দিব?"]

def fill(t, crop_k, pest_k, brand=""):
    return t.format(crop=CROP_BN[crop_k], pest=PEST_NAME[pest_k], brand=brand)

rows = []
for _, r in usable.sample(frac=1, random_state=SEED).drop_duplicates("reg_no").head(N_PER_TYPE).iterrows():
    rows.append({"type": "dose", "question": fill(rng.choice(T_DOSE), r.crop_k, r.pest_k, r.brand),
                 "crop": r.crop_k, "pest": r.pest_k, "brand": r.brand, "gold_dose": r.dose, "reg_no": r.reg_no})

pairs = usable.groupby(["crop_k", "pest_k"])
pair_list = sorted(pairs.groups.keys())
rng.shuffle(pair_list)
for k in range(N_PER_TYPE):  # fewer pairs than questions: cycle through them with a different template each round
    c, p = pair_list[k % len(pair_list)]
    g = pairs.get_group((c, p))
    rows.append({"type": "product", "question": fill(T_PRODUCT[(k // len(pair_list)) % len(T_PRODUCT)], c, p), "crop": c, "pest": p,
                 "gold_products": sorted(set(g.brand) | set(g.common_name))})

# registration: a product registered for the pest on some crop, never on the asked crop
by_brand_crops = usable.groupby("brand").crop_k.agg(set)
reg_rows = []
for _, r in usable.sample(frac=1, random_state=SEED + 1).iterrows():
    other = [c for c in CROP_BN if c not in by_brand_crops[r.brand] and (c, r.pest_k) in pairs.groups]
    if other:
        c = rng.choice(other)
        reg_rows.append({"type": "registration", "question": fill(rng.choice(T_REG), c, r.pest_k, r.brand),
                         "crop": c, "pest": r.pest_k, "brand": r.brand, "registered_for": sorted(by_brand_crops[r.brand])})
    if len(reg_rows) == N_PER_TYPE:
        break
rows += reg_rows

bench = pd.DataFrame(rows).drop_duplicates("question").reset_index(drop=True)
print(bench.type.value_counts().to_dict())
assert set(bench.type.value_counts()) == {N_PER_TYPE}, "fewer usable rows than planned; lower N_PER_TYPE"
bench.to_json(f"{OUT}/krishitools_300.jsonl", orient="records", lines=True, force_ascii=False)
print(bench.groupby("type").head(2)[["type", "question"]].to_string())
