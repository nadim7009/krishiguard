# %% [markdown]
# ## KrishiTools-Messy: the same 300 questions, written the way farmers type
# Each KrishiTools-300 question gets one change, picked by seed. The gold answer stays the same, except that a question with its crop removed should get a follow-up question instead of a dose.
#
# | change | example | applies to |
# |---|---|---|
# | drop_code | Rekazeb 80WP -> Rekazeb | dose, registration |
# | lower_join | Rekazeb 80WP -> rekazeb80wp | dose, registration |
# | typo | Rekazeb -> Rekzaeb (one swap or deletion) | dose, registration |
# | pest_word | নাবি ধসা -> লেট ব্লাইট, ব্লাস্ট -> ব্লাষ্ট | all |
# | crop_word | বেগুন -> বাইগন, টমেটো -> টমাটো | all |
# | no_crop | crop name removed, so the right reply asks which crop | dose |
#
# The pest and crop variants were written by us from common Bangla spellings and English loanwords; they are not collected from farmers, and the paper says so.

# %%
FORM_CODE = re.compile(r"\s*\d+(?:\.\d+)?\s?(?:G|GR|WG|WDG|WP|EC|SC|SL|SG|SP|FS|DF|EW|ME|CS|WS|OD|SE|ZC|DP|EG|ULV)\b", re.I)
PEST_VARIANTS = {"নাবি ধসা (মড়ক) রোগ": ["লেট ব্লাইট রোগ", "নাবী ধ্বসা রোগ"], "আগাম ধসা রোগ": ["আর্লি ব্লাইট রোগ", "আগাম ধ্বসা রোগ"],
                 "বাদামি গাছফড়িং (কারেন্ট পোকা)": ["কারেন পোকা", "বাদামী গাছ ফড়িং"], "ব্লাস্ট রোগ": ["ব্লাষ্ট রোগ"],
                 "খোলপোড়া রোগ": ["খোল পোড়া রোগ", "শিথ ব্লাইট"], "পাতা মোড়ানো পোকা": ["পাতা মোড়া পোকা"], "জাব পোকা": ["এফিড", "জাবপোকা"],
                 "সাদা মাছি": ["সাদামাছি", "হোয়াইট ফ্লাই"], "মাকড়": ["মাইট"], "ফল ছিদ্রকারী পোকা": ["ফল ছিদ্রকারি পোকা"],
                 "ডগা ও ফল ছিদ্রকারী পোকা": ["ডগা ও ফল ছিদ্রকারি পোকা"], "মাজরা পোকা": ["মাজড়া পোকা"],
                 "পাউডারি মিলডিউ রোগ": ["পাউডারী মিলডিউ"], "কাটুই পোকা": ["কাটুইপোকা"], "লেদা পোকা": ["আর্মিওয়ার্ম", "লেদাপোকা"],
                 "গান্ধি পোকা": ["গান্ধী পোকা"], "থ্রিপস": ["থ্রিপ্স"]}
CROP_VARIANTS = {"বেগুন": ["বাইগন"], "টমেটো": ["টমাটো"], "বাঁধাকপি": ["বাধাকপি"], "পেঁয়াজ": ["পিঁয়াজ", "পিয়াজ"], "ঢেঁড়স": ["ঢেড়স"],
                 "মরিচ": ["মরিছ"], "ধান": ["বোরো ধান", "আমন ধান"], "ভুট্টা": ["ভুট্টা গাছ"], "আলু": ["গোল আলু"], "কলা": ["কলা গাছ"]}

def typo(word, r):
    i = r.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:] if r.random() < 0.5 else word[:i] + word[i + 1:]

def perturb(row, r):
    """One farmer-style change to a KrishiTools-300 question: (new question, change name)."""
    q, b = row.question, row.brand if isinstance(row.brand, str) else ""
    pest = next((p for p in sorted(PEST_VARIANTS, key=len, reverse=True) if p in q), None)
    crop = next((c for c in sorted(CROP_VARIANTS, key=len, reverse=True) if re.search(rf"(^|\s){re.escape(c)}(?=[\s\-]|ফসল|খেত)", q)), None)
    opts = (["drop_code", "lower_join", "typo", "pest_word", "crop_word", "no_crop"] if row.type == "dose"
            else ["drop_code", "lower_join", "typo", "pest_word", "crop_word"] if row.type == "registration" else ["pest_word", "crop_word"])
    opts = [o for o in opts if (o != "pest_word" or pest) and (o not in ("crop_word", "no_crop") or crop)
            and (o not in ("drop_code", "lower_join", "typo") or (b and len(FORM_CODE.sub("", b)) >= 5))
            and (o != "drop_code" or FORM_CODE.sub("", b) != b)]
    kind = r.choice(opts)
    base = FORM_CODE.sub("", b).strip()
    if kind == "drop_code":
        return q.replace(b, base), kind
    if kind == "lower_join":
        return q.replace(b, re.sub(r"\s+", "", b).lower()), kind
    if kind == "typo":
        return q.replace(b, typo(base, r)), kind
    if kind == "pest_word":
        return q.replace(pest, r.choice(PEST_VARIANTS[pest])), kind
    if kind == "crop_word":
        return re.sub(rf"(^|\s){re.escape(crop)}(?=[\s\-]|ফসল|খেত)", lambda m: m.group(1) + r.choice(CROP_VARIANTS[crop]), q, count=1), kind
    no = re.sub(rf"(^|\s){re.escape(crop)}(?:\s*ফসলে|\s*খেতে|-এর|-এ)?\s*", r"\1", q, count=1)  # no_crop
    return re.sub(r"\s{2,}", " ", no).strip(), kind

def build_messy(kt, seed=SEED):
    r = random.Random(seed)
    out = []
    for row in kt.itertuples():
        q, kind = perturb(row, r)
        assert q != row.question, (kind, row.question)
        out.append({**row._asdict(), "question_orig": row.question, "question": q, "change": kind,
                    "expect": "ask" if kind == "no_crop" else row.type})
    return pd.DataFrame(out).drop(columns=["Index"])

# gold product lists rebuilt with the lookup's own rule (crop and pest as substrings of the DAE row);
# notebook 05's lists missed rows that name several pests at once, such as "GLH Thrips, Hispa" or "Sheath blight & Blast"
def gold_products_v2(crop, pest):
    d = dae[dae.crop.str.lower().str.contains(to_crop(crop), regex=False) & dae.pest.str.lower().str.contains(str(pest).lower(), regex=False)]
    return sorted(set(d.brand) | set(d.common_name))

def tool_correct_v4(pred, typ, gold_dose, gold_products):
    """v3 scoring, plus: an ingredient counts as unregistered only if it is not part of a registered one
    ("metalaxyl" inside "metalaxyl m" was flagged before)."""
    pred = FORM.sub(" ", str(pred))
    if typ != "product":
        return tool_correct(pred, typ, gold_dose, gold_products)
    names = [re.split(r"\s*\d", g)[0] for g in gold_products]
    named = any(len(nrm(n)) >= 4 and nrm(n) in nrm(pred) for n in names)
    allowed = {w.strip().lower() for g in gold_products for w in re.split(r"[+(),]", g)}
    return named and not any(i in pred.lower() and not any(i in a for a in allowed) for i in INGREDIENTS)
