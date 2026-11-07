# %% [markdown]
# ## Lookup v4: read messy product, pest and crop names
# A thin step in front of the v3 lookup: it finds the product the farmer meant and adds the standard pest and crop words to the question, then calls `tool_lookup` unchanged.
# - Product: exact name, name without the formulation code ("Rekazeb"), joined or lower case ("rekazeb80wp"), or a close spelling (difflib ratio at least 0.8, same first letter). When several products share a base name, the one registered for this crop and pest wins.
# - Pest and crop: Bangla spelling is normalised (ী/ি, ূ/ু, ধ্ব/ধ, ষ্ট/স্ট, ড়/র, hasanta, spaces), plus a short list of English loanwords and dialect crop names that we wrote.

# %%
import difflib

BASE, FULL = {}, {nrm(b): b for b in BRANDS}
for b in BRANDS:
    BASE.setdefault(nrm(FORM_CODE.sub("", b)), []).append(b)
KEYS = sorted(set(BASE) | set(FULL))
PEST_LEX = {"লেট ব্লাইট": "নাবি ধসা", "আর্লি ব্লাইট": "আগাম ধসা", "কারেন": "কারেন্ট", "শিথ ব্লাইট": "খোলপোড়া", "এফিড": "জাব",
            "হোয়াইট ফ্লাই": "সাদা মাছি", "মাইট": "মাকড়", "আর্মিওয়ার্ম": "লেদা", "পাতা মোড়া": "পাতা মোড়ানো"}
CROP_LEX = {"বাইগন": "বেগুন", "টমাটো": "টমেটো", "বাধাকপি": "বাঁধাকপি", "পিঁয়াজ": "পেঁয়াজ", "পিয়াজ": "পেঁয়াজ", "ঢেড়স": "ঢেঁড়স", "মরিছ": "মরিচ"}
BN_NORM = [("ী", "ি"), ("ূ", "ু"), ("ধ্ব", "ধ"), ("ষ্ট", "স্ট"), ("ষ", "শ"), ("ড়", "র"), ("য়", "য"), ("্", ""), (" ", ""), ("-", "")]

def bn_norm(t):
    t = str(t)
    for a, b in BN_NORM:
        t = t.replace(a, b)
    return t

def find_brand(q):
    """(text span in q, canonical brand) for a messy product name, or None."""
    for ch in re.findall(r"[A-Za-z][A-Za-z0-9.\-]*(?:\s+[A-Za-z0-9.\-]+){0,3}", q):
        words = ch.split()
        for n in range(len(words), 0, -1):
            span = " ".join(words[:n])
            k = nrm(span)
            if len(k) < 4:
                continue
            hit = FULL.get(k) and [FULL[k]] or BASE.get(k)
            if not hit:
                m = [c for c in difflib.get_close_matches(k, KEYS, n=3, cutoff=0.8) if c[0] == k[0]]
                hit = m and ([FULL[m[0]]] if m[0] in FULL else BASE[m[0]])
            if hit:
                return span, hit
    return None

def canonicalize(q):
    q = str(q)
    qn = bn_norm(q)
    extra = [bn_ for bn_ in PEST_BN if bn_ not in q and bn_norm(bn_) in qn]
    extra += [v for k, v in PEST_LEX.items() if k in q and v not in q]
    extra += [v for k, v in CROP_LEX.items() if re.search(rf"(^|\s){k}", q)]
    if extra:
        q = q + " (" + ", ".join(dict.fromkeys(extra)) + ")"  # tool_lookup matches these standard words
    if not any(b.lower() in q.lower() for b in BRANDS):
        found = find_brand(q)
        if found:
            span, cands = found
            crops = crops_in(q)
            pests = [en for b_, ens in PEST_BN.items() if b_ in q for en in ens]
            def fits(b):
                rows = dae[dae.brand == b]
                return any(any(to_crop(c) in str(x).lower() for c in crops) and any(p in str(y).lower() for p in pests)
                           for x, y in zip(rows.crop, rows.pest))
            q = re.sub(r"\s+".join(map(re.escape, span.split())), next((b for b in cands if fits(b)), cands[0]), q, count=1)
    return q

tool_lookup_v4 = lambda q: tool_lookup(canonicalize(q))
assert "Rekazeb 80WP" in canonicalize("আলু খেতে লেট ব্লাইট, rekazeb80wp কতখানি দিমু?")
assert "নাবি ধসা" in canonicalize("আলু খেতে লেট ব্লাইট, Rekazeb কতখানি দিমু?")
assert "100 g/ha" in tool_lookup_v4("ধানে কারেন পোকা লাগছে, Tabba কতটুকু দিব?")[1] or "Taba" in canonicalize("ধানে কারেন পোকা লাগছে, Tabba কতটুকু দিব?")

ASK_CROP = "কোন ফসলে দেবেন জানালে নিবন্ধিত মাত্রা বলতে পারব। {brand} এই কাজে নিবন্ধিত আছে এই ফসলগুলোতে: {crops}। আপনার কোন ফসল?"

def tool_lookup_v5(q):
    """v4, plus, when no crop is named: a product registered for this pest on several crops gets a fixed question back
    (the crops, no dose), which run_agent sends as the whole reply; a product registered on one crop only gets that crop's dose, with the crop named."""
    cq = canonicalize(q)
    ev, lead = tool_lookup(cq)
    if crops_in(cq):
        return ev, lead
    b = next((b for b in BRANDS if b.lower() in cq.lower()), None)
    pests = [en for b_, ens in PEST_BN.items() if b_ in cq for en in ens]
    if not (b and pests):
        return ev, lead
    rows = dae[(dae.brand == b) & dae.pest.str.lower().map(lambda p: any(x in p for x in pests))]
    if rows.crop.nunique() > 1:
        return ev, ASK_CROP.format(brand=b, crops=", ".join(rows.crop.unique()[:6]))
    crop_bn = next((c for c, en in CROP_BN.items() if len(rows) and to_crop(en) in rows.crop.iloc[0].lower()), None)
    return tool_lookup(f"{cq} {crop_bn} ফসলে") if crop_bn else (ev, lead)  # one crop only: the dose is for that crop, and the lead names it

assert tool_lookup_v5("ধানে কারেন্ট পোকা লাগছে, Taba 25WDG কতটুকু দিব?")[1] == tool_lookup_v4("ধানে কারেন্ট পোকা লাগছে, Taba 25WDG কতটুকু দিব?")[1]
