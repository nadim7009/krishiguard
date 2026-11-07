# %% [markdown]
# ## DAE training examples
# Every answer is built from the same rows the tool returns, so the model only ever sees doses that are in its prompt.
# The site format mirrors the website's system prompt, including the amount already worked out for the farmer's land.

# %%
SYSTEM = ("তুমি বাংলাদেশের কৃষকদের জন্য একজন কৃষি পরামর্শক। সংক্ষেপে বাংলায় উত্তর দাও। "
          "ফসল বা সমস্যা পরিষ্কার না হলে আগে প্রশ্ন করে জেনে নাও। নিশ্চিত না হলে কীটনাশক বা মাত্রা বলবে না।")  # notebook 04
AGENT_CTX = "নিবন্ধিত বালাইনাশক (কৃষি সম্প্রসারণ অধিদপ্তরের তালিকা):\n{}"  # notebook 04 run_agent

def agent_msgs(question, context=None):
    user = question if context is None else (
        "নিচের তথ্যের ভিত্তিতে উত্তর দাও। তথ্যে না থাকলে বলো জানা নেই।\n\n"
        f"তথ্য:\n{context}\n\nপ্রশ্ন: {question}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

SITE_RULES = """তুমি কৃষিগার্ড, বাংলাদেশের কৃষকদের পাশে থাকা একজন বন্ধুর মতো সহকারী। নিয়মগুলো মেনে চলবে:
১. সহজ, ছোট বাংলায় উত্তর দেবে। কৃষককে "আপনি" বলবে। প্রশ্ন ইংরেজিতে হলে ইংরেজিতে উত্তর দেবে।
২. কৃষক যেকোনো বিষয়ে কথা বলতে পারেন: জমি, বাজার, পরিবার, টাকা, মন খারাপ বা একেবারে ব্যক্তিগত কথা। মন দিয়ে শুনবে, সহানুভূতি দেখাবে, বিচার করবে না, আর প্রয়োজনে একটা প্রশ্ন করে আরও জানতে চাইবে।
৩. বালাইনাশকের নাম, মাত্রা বা কোনো পরিমাণ কখনো নিজে বানিয়ে বলবে না। শুধু নিচে দেওয়া DAE তথ্য থেকে বলবে। তথ্য না থাকলে ফসল আর পোকা বা রোগের নাম জিজ্ঞেস করবে।
৪. নিশ্চিত না হলে অনুমান না করে জিজ্ঞেস করবে।
৫. কেউ নিজের ক্ষতি করার কথা বললে নরমভাবে কথা বলবে, পরিবারের কাউকে পাশে ডাকতে বলবে, বিপদে ৯৯৯ নম্বরে ফোন করতে বলবে, কীটনাশক দূরে তালা দিয়ে রাখতে বলবে। এমন সময় কোনো ওষুধ বা বিষের তথ্য দেবে না।
৬. কারও ক্ষতি করা, বেআইনি কাজ বা নিষিদ্ধ বালাইনাশক নিয়ে সাহায্য করবে না।
৭. উত্তর সাধারণত ১২০ শব্দের মধ্যে রাখবে। কোনো ইমোজি ব্যবহার করবে না। কৃষক নিজে না বললে "ভাই", "আপা" এমন সম্বোধন করবে না, নাম জানা থাকলে নাম ধরে ডাকবে।
৮. মাত্রার প্রশ্নে উত্তর হবে এভাবে: কোন পণ্য, তার নিবন্ধিত মাত্রা, আর কৃষকের জমির মোট পরিমাণ দেওয়া থাকলে সেটা। নিজে কোনো হিসাব করবে না।"""
BN_DIG = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
bn = lambda x: str(x).translate(BN_DIG)
NAMES = ["করিম", "রহিমা", "সালমা", "জব্বার", "হাসান", "মালেকা", "রফিক", "শাহানা", "আলমগীর", "নাসরিন", "কামাল", "পারভীন"]
DATES = ["২২ কার্তিক ১৪৩৩ বঙ্গাব্দ, রবি মৌসুম", "৫ ফাল্গুন ১৪৩৩ বঙ্গাব্দ, রবি মৌসুম", "১৮ বৈশাখ ১৪৩৪ বঙ্গাব্দ, খরিফ-১ মৌসুম",
         "৯ শ্রাবণ ১৪৩৪ বঙ্গাব্দ, খরিফ-২ মৌসুম", "৩০ আশ্বিন ১৪৩৩ বঙ্গাব্দ, খরিফ-২ মৌসুম"]
LAND = [("বিঘা", 0.13355), ("শতাংশ", 0.004047), ("কাঠা", 0.006677), ("একর", 0.4047)]

def site_msgs(question, evidence="", profile=None):
    prof = "; ".join(f"{k} {v}" for k, v in (profile or {}).items())
    parts = [SITE_RULES, f"আজ {rng.choice(DATES)}।",
             f"এই কৃষক নিজে জানিয়েছেন: {prof}। এগুলো মাথায় রেখে উত্তর দেবে।" if prof else "এই কৃষকের খামারের তথ্য জানা নেই।",
             f"DAE নিবন্ধিত বালাইনাশক তালিকা থেকে এই প্রশ্নের তথ্য (ওষুধের নাম ও মাত্রা শুধু এখান থেকে নেবে):\n{evidence}" if evidence
             else "এই প্রশ্নের জন্য DAE তালিকা থেকে কোনো তথ্য দেওয়া হয়নি, তাই কোনো বালাইনাশকের নাম বা মাত্রা বলবে না।"]
    return [{"role": "system", "content": "\n\n".join(parts)}, {"role": "user", "content": question}]

def parse_dose(d):  # same rule as the website's parseDose: grams or millilitres, per hectare or per litre
    m = re.search(r"([\d.]+)\s*(kg|gm|g|ml|l|litre|liter)\s*/\s*(ha|hectare|lof|l\b|litre|liter)", str(d), re.I)
    if not m:
        return None
    u = m.group(2).lower()
    unit, f = ("g", 1000) if u == "kg" else ("g", 1) if u.startswith("g") else ("ml", 1) if u == "ml" else ("ml", 1000)
    return float(m.group(1)) * f, unit, "ha" if m.group(3).lower().startswith("h") else "L"

def amount_text(x, unit):  # the website's amountText
    big = x >= 1000
    v = x / 1000 if big else x
    r = round(v) if v >= 10 else round(v * 10) / 10
    r = int(r) if float(r).is_integer() else r
    return f"{bn(r)} " + ({"g": "কেজি", "ml": "লিটার"} if big else {"g": "গ্রাম", "ml": "মিলি"})[unit]

assert parse_dose("1.0 L/ha") == (1000.0, "ml", "ha") and parse_dose("2 ml/L of water")[2] == "L"
assert amount_text(100 * 3 * 0.13355, "g") == "৪০ গ্রাম", amount_text(100 * 3 * 0.13355, "g")

def for_farm(dose, land, tank):  # (text for the prompt, text for the answer) or None
    p = parse_dose(dose)
    if not p:
        return None
    amt, unit, per = p
    if per == "ha" and land:
        t = f"আপনার {bn(land[0])} {land[1]} জমিতে মোট প্রায় {amount_text(amt * land[0] * land[2], unit)}"
        return t, t
    if per == "L":
        t = f"{bn(tank)} লিটারের ট্যাংকে {amount_text(amt * tank, unit)}"
        return t, t
    return None

# %%
kt = pd.read_json(find("krishitools_300.jsonl"), lines=True)
HELD_OUT = set(kt.brand.dropna()) - {""}
print(len(HELD_OUT), "products held out (KrishiTools-300 dose and registration questions)")
PEST_NAMES = {"stem borer": "মাজরা পোকা", "bph": "বাদামি গাছফড়িং (কারেন্ট পোকা)", "leaf roller": "পাতা মোড়ানো পোকা",
              "leaf folder": "পাতা মোড়ানো পোকা", "blast": "ব্লাস্ট রোগ", "sheath blight": "খোলপোড়া রোগ",
              "late blight": "নাবি ধসা (মড়ক) রোগ", "early blight": "আগাম ধসা রোগ", "aphid": "জাব পোকা", "whitefly": "সাদা মাছি",
              "white fly": "সাদা মাছি", "thrips": "থ্রিপস", "mite": "মাকড়", "shoot and fruit borer": "ডগা ও ফল ছিদ্রকারী পোকা",
              "fruit and shoot borer": "ডগা ও ফল ছিদ্রকারী পোকা", "fruit borer": "ফল ছিদ্রকারী পোকা", "rice bug": "গান্ধি পোকা",
              "armyworm": "লেদা পোকা", "army worm": "লেদা পোকা", "cutworm": "কাটুই পোকা", "powdery mildew": "পাউডারি মিলডিউ রোগ"}
pest_of = lambda p: next((k for k in sorted(PEST_NAMES, key=len, reverse=True) if k in str(p).lower()), None)
crop_of = lambda c: next((bn_ for bn_, en in CROP_BN.items() if to_crop(en) in str(c).lower()), None)

# templates differ from KrishiTools-300 (notebook 05)
T_DOSE = ["{brand} {crop}-এর {pest} দমনে কত মাত্রায় দিতে হয়?", "আমার {crop} জমিতে {pest} দেখা দিছে, {brand} কতটুকু মিশাবো?",
          "{crop} জমিতে {pest} দমনে {brand}-এর সঠিক মাত্রা কত?", "{brand}-এর মাত্রা জানতে চাই, {crop}-এ {pest} লাগছে।",
          "{crop} গাছে {pest}, {brand} দেওয়ার নিয়ম কী?", "{pest}-এর জন্য {crop} জমিতে {brand} কতখানি লাগবে?"]
T_PRODUCT = ["{crop} জমিতে {pest} লাগছে, কোন ওষুধ ভালো হবে?", "{pest} থেকে {crop} বাঁচাতে কী দেব?",
             "{crop}-এ {pest}-এর জন্য সরকার অনুমোদিত কোন কীটনাশক আছে?", "আমার {crop}-এ {pest}-এর আক্রমণ, কী স্প্রে করলে কাজ হবে?"]
T_REG = ["{brand} কি {crop}-এ {pest}-এর জন্য দেওয়া যায়?", "দোকানদার {crop} জমির {pest}-এর জন্য {brand} দিতে বলছে, এটা কি ঠিক?",
         "{crop}-এর {pest} মারতে {brand} চলবে?"]
CLOSE = ["বোতলের লেবেল পড়ে নিন, মাস্ক ও গ্লাভস পরে স্প্রে করুন।",
         "বাতাস কম থাকলে সকালে বা বিকেলে স্প্রে করুন, আর স্প্রের পর হাত-মুখ ভালো করে ধুয়ে নিন।",
         "একই উপাদান বারবার না দিয়ে বদলে বদলে দিন, তাতে পোকার প্রতিরোধ ক্ষমতা কম তৈরি হয়।",
         "সন্দেহ থাকলে উপজেলা কৃষি অফিসে মিলিয়ে নিন।"]

usable = dae.assign(pk=dae.pest.map(pest_of), cb=dae.crop.map(crop_of))
usable = usable.dropna(subset=["pk", "cb"])
usable = usable[usable.dose.str.contains(r"\d") & ~usable.brand.isin(HELD_OUT) & (usable.brand.str.len() >= 5)
                & usable.common_name.str.contains(r"[A-Za-z]{3}")]  # a few rows lost their ingredient names in parsing
usable = usable.drop_duplicates(["brand", "cb", "pk"]).sample(frac=1, random_state=SEED)
print(len(usable), "usable DAE rows for training")

def dose_answer(q, crop, pest_bn, row, name=None, farm=None):
    who = f"{name}, " if name else ""
    s = (f"{who}{crop} ফসলে {pest_bn} দমনে {row.brand} ({row.common_name}) নিবন্ধিত। নিবন্ধিত মাত্রা {row.dose}"
         + (f", {farm}" if farm else "") + f"। নিবন্ধন নম্বর DAE {row.reg_no}।")
    return s + " " + rng.choice(CLOSE)

def make_dae_examples(n_dose=450, n_product=250, n_reg=250):
    out = []
    for row in usable.head(n_dose).itertuples():
        q = rng.choice(T_DOSE).format(brand=row.brand, crop=row.cb, pest=PEST_NAMES[row.pk])
        ev, lead = tool_lookup(q)
        if f"DAE {row.reg_no}]" not in lead:  # the tool must pick this very row, as it would in the agent
            continue
        out.append({"kind": "dae_dose", "fmt": "agent", "messages": agent_msgs(q, AGENT_CTX.format(ev)),
                    "answer": dose_answer(q, row.cb, PEST_NAMES[row.pk], row)})
        if rng.random() < 0.6:  # website format with a farm profile
            name = rng.choice(NAMES)
            land = (rng.choice([1, 2, 3, 5, 10, 20, 33]), *rng.choice(LAND))  # (count, unit name, hectares per unit)
            ff = for_farm(row.dose, land, 16)
            line = f"পণ্য {row.brand} ({row.common_name} · {row.crop}, {row.pest}): নিবন্ধিত মাত্রা {row.dose}" + (f"; এই কৃষকের জন্য হিসাব করা: {ff[0]}" if ff else "")
            ev_site = line + f"\nনিবন্ধন নম্বর: DAE {row.reg_no}"
            prof = {"নাম": name, "ফসল": row.cb, "জমি": f"{bn(land[0])} {land[1]}", "স্প্রে ট্যাংক": "16 লিটার"}
            out.append({"kind": "dae_dose", "fmt": "site", "messages": site_msgs(q, ev_site, prof),
                        "answer": dose_answer(q, row.cb, PEST_NAMES[row.pk], row, name, ff[1] if ff else None)})
    pairs = usable.groupby(["cb", "pk"])
    keys = sorted(pairs.groups)
    rng.shuffle(keys)
    for k, ((c, p), t) in enumerate([(kp, t) for t in T_PRODUCT for kp in keys][:n_product]):  # few pairs, so each gets several phrasings
        q = t.format(crop=c, pest=PEST_NAMES[p])
        ev, lead = tool_lookup(q)
        alt = alternatives(crops_in(q), [en for b_, ens in PEST_BN.items() if b_ in q for en in ens])
        if not ev or not alt:
            continue
        a = (f"{c} ফসলে {PEST_NAMES[p]} দমনে নিবন্ধিত কয়েকটি বালাইনাশক:\n"
             + "\n".join(f"- {h['brand']} ({h['common_name']}): মাত্রা {h['dose']} [DAE {h['reg_no']}]" for h in alt)
             + "\nএকই উপাদান বারবার না দিয়ে বদলে বদলে দিন। " + rng.choice(CLOSE[:2]))
        out.append({"kind": "dae_product", "fmt": "agent", "messages": agent_msgs(q, AGENT_CTX.format(ev)), "answer": a})
    by_crops = usable.groupby("brand").cb.agg(set)
    n = 0
    for row in usable.itertuples():
        other = [c for (c, p) in keys if p == row.pk and c not in by_crops[row.brand]
                 and not dae[(dae.brand == row.brand)].crop.str.lower().str.contains(to_crop(CROP_BN[c]), regex=False).any()]
        if not other:
            continue
        c = rng.choice(other)
        q = rng.choice(T_REG).format(brand=row.brand, crop=c, pest=PEST_NAMES[row.pk])
        ev, lead = tool_lookup(q)
        if "নিবন্ধিত নয়" not in lead:
            continue
        alt = alternatives(crops_in(q), [en for b_, ens in PEST_BN.items() if b_ in q for en in ens])
        a = f"না, {row.brand} {c} ফসলের জন্য নিবন্ধিত নয়, তাই {c} জমিতে এটি দেবেন না।"
        if alt:
            a += f" {c} ফসলে {PEST_NAMES[row.pk]} দমনে নিবন্ধিত আছে: " + ", ".join(f"{h['brand']} ({h['common_name']})" for h in alt) + "। মাত্রা জানতে এর একটির নাম দিয়ে জিজ্ঞেস করুন।"
        else:
            a += " এই কাজে কোন পণ্য নিবন্ধিত, তা উপজেলা কৃষি অফিসে জেনে নিন।"
        out.append({"kind": "dae_not_registered", "fmt": "agent", "messages": agent_msgs(q, AGENT_CTX.format(ev)), "answer": a})
        n += 1
        if n == n_reg:
            break
    return out

dae_ex = make_dae_examples()
print(pd.Series([(e["kind"], e["fmt"]) for e in dae_ex]).value_counts().to_dict())
for e in dae_ex:  # every number in an answer must already be in its prompt (Bangla digits count as the same number)
    to_en = lambda s: s.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    allowed = set(re.findall(r"\d+(?:\.\d+)?", to_en(" ".join(m["content"] for m in e["messages"]))))
    extra = set(re.findall(r"\d+(?:\.\d+)?", to_en(e["answer"]))) - allowed
    assert not extra, (extra, e["answer"])
    if e["kind"] == "dae_not_registered":
        assert not re.search(r"\d+(\.\d+)?\s*(ml|g|gm|kg|l)\s*/", e["answer"], re.I), e["answer"]
print(dae_ex[0]["answer"], "\n", next(e for e in dae_ex if e["fmt"] == "site")["answer"])
