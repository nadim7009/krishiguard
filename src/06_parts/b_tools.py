# %% [markdown]
# ## DAE lookup, copied from notebook 04 (v3) so the training prompts match what the agent sends

# %%
dae = pd.read_csv(find("dae_registered_pesticides.csv")).fillna("")
CROP_BN = {"ধান": "rice", "গম": "wheat", "আলু": "potato", "বেগুন": "brinjal", "মরিচ": "chilli",
           "টমেটো": "tomato", "পাট": "jute", "ভুট্টা": "maize", "সরিষা": "mustard", "আম": "mango",
           "চা": "tea", "আখ": "sugarcane", "পেঁয়াজ": "onion", "রসুন": "garlic", "শিম": "bean",
           "বাঁধাকপি": "cabbage", "ফুলকপি": "cauliflower", "শসা": "cucumber", "কলা": "banana",
           "লিচু": "litchi", "মসুর": "lentil", "ছোলা": "chickpea", "তুলা": "cotton", "ঢেঁড়স": "okra",
           "করলা": "bitter gourd", "লাউ": "bottle gourd", "পেঁপে": "papaya", "সয়াবিন": "soybean"}
PEST_BN = {"মাজরা": ["stem borer"], "কারেন্ট": ["bph"], "গাছফড়িং": ["bph", "hopper"], "পাতা মোড়ানো": ["leaf roller", "leaf folder"],
           "ব্লাস্ট": ["blast"], "খোল পোড়া": ["sheath blight"], "খোলপোড়া": ["sheath blight"], "পাতা ঝলসানো": ["bacterial leaf blight", "blight"],
           "নাবি ধসা": ["late blight"], "মড়ক": ["late blight"], "আগাম ধসা": ["early blight"], "জাব": ["aphid"],
           "সাদা মাছি": ["whitefly", "white fly"], "থ্রিপস": ["thrips"], "মাকড়": ["mite"], "ডগা ও ফল ছিদ্রকারী": ["shoot and fruit borer", "fruit and shoot borer"],
           "ফল ছিদ্রকারী": ["fruit borer"], "গান্ধি": ["rice bug"], "লেদা": ["armyworm", "army worm"], "কাটুই": ["cutworm", "cut worm"],
           "পাউডারি": ["powdery mildew"], "ঢলে পড়া": ["wilt"], "পচা": ["rot"]}
SHORT_SUFFIX = ("", "ে", "ের", "য়ের", "গাছ", "গাছে", "গাছের", "ক্ষেত", "ক্ষেতে")

def to_crop(crop):
    crop = CROP_BN.get(str(crop).strip(), str(crop).strip()).lower()
    return "chil" if crop in ("chilli", "chili") else crop  # the list spells it both ways

def crops_in(text):
    words = re.findall(r"[ঀ-৿]+", str(text))
    # short names like আম or চা hide inside other words (আমার, চাষ), so they must match a whole word plus a case ending
    return [en for bn, en in CROP_BN.items()
            if any(w.startswith(bn) if len(bn) > 2 else w in {bn + x for x in SHORT_SUFFIX} for w in words)]

def products_for(crop, pest, limit=3):
    d = dae[dae.crop.str.lower().str.contains(to_crop(crop), regex=False) & dae.pest.str.lower().str.contains(pest, regex=False)]
    return d.drop_duplicates("reg_no").head(limit).to_dict("records")

# v3: the DAE answer comes first. A product named in the question gets its registered dose for this crop and pest,
# or a plain "not registered" (for the crop, or for this pest on the crop) without any dose.
BRANDS = sorted({b for b in dae.brand if len(b) >= 5}, key=len, reverse=True)
CROP_EN_BN = {en: bn for bn, en in CROP_BN.items()}
row_line = lambda h: f"{h.brand} ({h.common_name}), {h.crop} - {h.pest}: {h.dose} [DAE {h.reg_no}]"

def alternatives(crops, pests, limit=3):
    hits = [h for c in crops[:1] for p in pests for h in products_for(c, p, limit)]
    return list({h["reg_no"]: h for h in hits}.values())[:limit]

def tool_lookup(q):
    """Returns (DAE rows for the model, fixed lead sentence shown first). Both empty when nothing matches."""
    crops = crops_in(q)
    pests = [en for bn, ens in PEST_BN.items() if bn in str(q) for en in ens]
    pest_bn = next((bn for bn in PEST_BN if bn in str(q)), "")
    crop_bn = CROP_EN_BN.get(crops[0], crops[0]) if crops else ""
    ql = str(q).lower()
    b = next((b for b in BRANDS if b.lower() in ql), None)
    if b:
        rows = dae[dae.brand == b]
        on_crop = rows[[any(to_crop(c) in str(x).lower() for c in crops) for x in rows.crop]] if crops else rows
        on_pest = on_crop[[any(p in str(x).lower() for p in pests) for x in on_crop.pest]] if pests else on_crop
        if not crops:  # no crop named: show its uses, the model asks which crop
            return "\n".join(row_line(h) for h in rows.drop_duplicates(["crop", "pest"]).head(5).itertuples()), ""
        if len(on_pest):
            h = on_pest.iloc[0]
            lead = f"{b} ({h.common_name}) {crop_bn} ফসলে {h.pest} দমনের জন্য নিবন্ধিত। নিবন্ধিত মাত্রা {h.dose} [DAE {h.reg_no}]।"
            return "\n".join(row_line(h) for h in on_pest.drop_duplicates(["crop", "pest"]).head(3).itertuples()), lead
        lead = (f"{b} {crop_bn} ফসলের জন্য নিবন্ধিত নয়, তাই এই ফসলে এটি ব্যবহার করা যাবে না।" if on_crop.empty
                else f"{b} {crop_bn} ফসলে {pest_bn} দমনের জন্য নিবন্ধিত নয়, তাই এই কাজে এটি ব্যবহার করা যাবে না।")
        alt = alternatives(crops, pests)
        ev = lead + f"\n{b} নিবন্ধিত আছে: " + ", ".join(sorted(set(rows.crop)))
        if alt:  # names only, so a "no" never carries a dose
            ev += f"\n{crop_bn} ফসলে {pest_bn} দমনে নিবন্ধিত পণ্য: " + ", ".join(f"{h['brand']} ({h['common_name']})" for h in alt)
        return ev, lead
    alt = alternatives(crops, pests)
    if not alt:
        return "", ""
    lead = f"{crop_bn} ফসলে {pest_bn} দমনে নিবন্ধিত বালাইনাশকের কয়েকটি: " + "; ".join(
        f"{h['brand']} ({h['common_name']}), মাত্রা {h['dose']} [DAE {h['reg_no']}]" for h in alt) + "।"
    return "\n".join(f"{h['brand']} ({h['common_name']}), {h['crop']} - {h['pest']}: {h['dose']} [DAE {h['reg_no']}]" for h in alt), lead

