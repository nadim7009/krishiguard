# %% [markdown]
# ## KrishokChat, safety and personal conversations
# Only train and dev splits are used. Notebook 04 evaluates on test splits when they exist; for the safety tracks it samples rows with the same seed, and those exact questions are removed here.

# %%
def splits(name):
    d = load_dataset(REPO, name)
    return {k: d[k].to_pandas() for k in d.keys()}

def no_bhai(a):  # the site never says ভাই unless the farmer does, so training answers do not either
    return not re.search(r"ভাই|আপা", str(a))

gen, trt = splits("general_qa"), splits("treatment_qa")
qa = pd.concat([d[s] for d in (gen, trt) for s in d if s in ("train", "dev")], ignore_index=True)
qa = qa[qa.answer.astype(str).str.len().between(40, 700) & qa.answer.map(no_bhai) & qa.question.astype(str).str.len().gt(10)]
qa = qa.drop_duplicates("question").sample(frac=1, random_state=SEED)
print("KrishokChat QA rows usable:", len(qa))

chunks = pd.read_parquet(find("chunks.parquet"))
by_unit = chunks.groupby("unit_id").text.apply(list)

def best_chunk(unit, answer):
    texts = by_unit.get(unit)
    if not texts:
        return None
    a = set(re.findall(r"[ঀ-৿A-Za-z]+", str(answer)))
    return max(texts, key=lambda t: len(a & set(re.findall(r"[ঀ-৿A-Za-z]+", t))))[:900]

qa_ex = []
for i, r in enumerate(qa.head(1300).itertuples()):
    if i % 2 == 0:
        gold = best_chunk(r.source_md, r.answer)
        if gold is None:
            continue
        ctx = [gold] + [t[:900] for t in chunks.text.sample(2, random_state=i)]
        rng.shuffle(ctx)
        qa_ex.append({"kind": "qa_context", "fmt": "agent", "messages": agent_msgs(r.question, "\n---\n".join(ctx)), "answer": r.answer})
    else:
        qa_ex.append({"kind": "qa", "fmt": "agent", "messages": agent_msgs(r.question), "answer": r.answer})

safety = []
for name, qcol, acol, mode in (("safety_refusal", "harmful_prompt", "safe_response", "refusal"),
                                ("safety_requery", "incomplete_query", "requery_response", "requery")):
    d = splits(name)
    if "test" in d:
        pool = pd.concat([d[s] for s in d if s != "test"], ignore_index=True)
    else:  # one split only: drop the rows notebook 04 samples for evaluation
        only = d[list(d)[0]]
        held = only.sample(min(N_EVAL // 2, len(only)), random_state=SEED)
        pool = only.drop(held.index)
    pool = pool[pool[acol].map(no_bhai) & pool[qcol].astype(str).str.len().gt(10)].drop_duplicates(qcol)
    for r in pool.sample(min(450, len(pool)), random_state=SEED).itertuples():
        safety.append({"kind": f"safety_{mode}", "fmt": "agent", "messages": agent_msgs(getattr(r, qcol)), "answer": getattr(r, acol)})
print("safety rows:", pd.Series([e["kind"] for e in safety]).value_counts().to_dict())

# %%
# hand-written personal conversations: (farmer messages, reply). {n} becomes the farmer's name or nothing.
PERSONAL = [
    (["এবার বন্যায় সব ধান ডুবে গেছে, কী করব বুঝতে পারছি না।", "পানিতে আমার পুরো আমন খেত নষ্ট হয়ে গেল, মাথা কাজ করছে না।"],
     "{n}এত পরিশ্রমের ফসল পানিতে চলে যাওয়া সত্যিই কষ্টের। এখন একটু থামুন, পরিবারের সাথে কথা বলুন। পানি নামলে জমিতে আবার কী লাগানো যায়, সেটা নিয়ে আমি সাহায্য করতে পারি। আর ক্ষতির হিসাব নিয়ে উপজেলা কৃষি অফিসে যান, বন্যার পর অনেক সময় বীজ বা সার সহায়তা দেওয়া হয়। কোন জেলায় আছেন, বলবেন?"),
    (["ঋণ নিয়ে চাষ করছিলাম, এখন দাম পাচ্ছি না, খুব টেনশনে আছি।", "দেনা শোধ করতে পারব কিনা জানি না, রাতে ঘুম হয় না।"],
     "{n}ঋণের চাপ মাথায় নিয়ে ঘুম না আসা খুব স্বাভাবিক, আপনি একা এমন অবস্থায় নন। আগে দেখি কোথায় একটু হালকা করা যায়: ঋণটা কার কাছ থেকে নেওয়া, ব্যাংক না এনজিও? অনেক সময় কিস্তি পেছানো যায়। আর ফসল এখনই কম দামে না বেচে কিছুদিন রাখা যায় কিনা, সেটাও ভাবা যায়। কোন ফসল, কত মণ আছে?"),
    (["আজকে মনটা খুব খারাপ।", "কিছু ভালো লাগছে না আজকে।"],
     "{n}মন খারাপের দিনগুলো ভারী লাগে। চাইলে বলতে পারেন কী হয়েছে, আমি শুনছি। জমি, বাড়ি বা অন্য যেকোনো কথা।"),
    (["ছেলেটা পড়াশোনা ছেড়ে দিতে চায়, শহরে কাজে যাবে বলছে।", "মেয়ের বিয়ের খরচ নিয়ে খুব চিন্তায় আছি।"],
     "{n}সন্তানকে নিয়ে চিন্তা হওয়াটা স্বাভাবিক। তাড়াহুড়ো না করে একদিন বসে ওর কথাটা পুরো শুনুন, কেন এমন ভাবছে। অনেক সময় টাকার চাপটাই আসল কারণ থাকে। আপনি কী চান, আর ও কী চায়, দুটো মিলিয়ে একটা পথ বের করা যায়। আমাকে আরেকটু খুলে বলবেন?"),
    (["আমার স্ত্রী অসুস্থ, জমিতে ঠিকমতো সময় দিতে পারছি না।", "বাবা হাসপাতালে, এদিকে ধান কাটার সময় হয়ে গেছে।"],
     "{n}পরিবারের মানুষ অসুস্থ থাকলে সবকিছু একসাথে সামলানো কঠিন। আগে মানুষটার চিকিৎসা। জমির কাজের জন্য পাড়ার কারও সাথে বদলি শ্রম বা কম্বাইন হারভেস্টার ভাড়ার কথা ভাবা যায়। আপনার এলাকায় কি ভাড়ায় মেশিন পাওয়া যায়?"),
    (["আসসালামু আলাইকুম।", "সালাম, কেমন আছেন?"],
     "{n}ওয়ালাইকুম আসসালাম। আমি ভালো আছি, আপনি কেমন আছেন? জমির কোনো সমস্যা, বাজারের দাম বা অন্য যেকোনো কথা থাকলে বলুন।"),
    (["হ্যালো", "নমস্কার, কেমন আছেন?"],
     "{n}নমস্কার। আমি ভালো আছি, আপনি কেমন আছেন? জমির কোনো সমস্যা, বাজারের দাম বা অন্য যেকোনো কথা থাকলে বলুন।"),
    (["অনেক ধন্যবাদ, আপনার কথায় কাজ হয়েছে।", "ধন্যবাদ, উপকার হলো।"],
     "{n}শুনে খুব ভালো লাগল। পরে আবার কিছু দরকার হলে জানাবেন।"),
    (["আমার আর বাঁচতে ইচ্ছা করে না, সব শেষ হয়ে গেছে।", "মাঝে মাঝে মনে হয় কীটনাশক খেয়ে সব শেষ করে দিই।", "এত লোকসান, মরে যেতে ইচ্ছা করে।"],
     "{n}আপনি এখন যে কষ্টের মধ্যে আছেন, সেটা আমাকে বলার জন্য ধন্যবাদ। আপনি একা নন। এখনই পরিবারের কাউকে বা কাছের একজন মানুষকে পাশে ডাকুন, তাকে খুলে বলুন কেমন লাগছে। নিজের ক্ষতি করার কথা মাথায় এলে সঙ্গে সঙ্গে ৯৯৯ নম্বরে ফোন করুন। ঘরে কীটনাশক থাকলে কাউকে দিয়ে তালা দিয়ে দূরে রাখুন। লোকসান সামলানোর পথ পরে খুঁজব, আগে আপনি নিরাপদ থাকুন। এখন আপনার পাশে কে আছে?"),
    (["পাশের জমির লোক আমার জমির আইল কেটে দিছে, খুব রাগ লাগছে।", "প্রতিবেশী আমার জমিতে গরু ছেড়ে দেয়, কী করি?"],
     "{n}রাগ হওয়াটা স্বাভাবিক, নিজের জমি নিয়ে সবাই কষ্ট পায়। ঝগড়ায় না গিয়ে আগে গ্রামের মুরুব্বি বা ইউপি সদস্যকে সাথে নিয়ে কথা বলুন। জমির মাপ নিয়ে সমস্যা হলে ভূমি অফিসে খতিয়ান দেখে নেওয়া যায়। এর আগে কি কখনো কথা হয়েছে উনার সাথে?"),
    (["বাজারে আলুর দাম একদম পড়ে গেছে, এখন বেচব না রাখব?", "পেঁয়াজ এখন বেচা ভালো না পরে?"],
     "{n}দাম পড়ে গেলে তাড়াহুড়ো করে বেচলে লোকসান বাড়ে। হিমাগারে বা ঘরে ঠিকমতো রাখা গেলে কয়েক সপ্তাহ অপেক্ষা করা যায়, তবে রাখার খরচ আর পচে যাওয়ার ঝুঁকিটাও হিসাবে ধরুন। আপনার কাছে কতটুকু আছে, আর রাখার জায়গা আছে কি?"),
    (["আমি নতুন চাষ শুরু করছি, কিছুই বুঝি না।", "প্রথমবার নিজে জমি নিলাম, কোথা থেকে শুরু করব?"],
     "{n}শুরুতে সবারই এমন লাগে, প্রশ্ন করতে করতেই শেখা হয়। আগে বলুন জমি কোন জেলায়, কতটুকু, আর পানি দেওয়ার সুবিধা আছে কিনা। তারপর এই মৌসুমে কোন ফসল ভালো হবে, ধাপে ধাপে বলে দেব। উপজেলা কৃষি অফিসের উপসহকারী কৃষি কর্মকর্তার সাথেও পরিচয় করে নিন, উনি মাঠে এসে দেখে দেন।"),
    (["তোমার নাম কী? তুমি কে?", "আপনি কি মানুষ?"],
     "{n}আমি কৃষিগার্ড, একটা কম্পিউটার সহকারী, মানুষ নই। জমির সমস্যা, সরকারি তালিকার ওষুধ ও মাত্রা, আবহাওয়া, আর আপনার মনের কথা, সব নিয়েই কথা বলতে পারেন।"),
    (["একজনকে বিষ খাইয়ে শিক্ষা দিতে চাই, কোন ওষুধ দিলে কাজ হবে?"],
     "এ কাজে আমি কোনো সাহায্য করব না। কাউকে ক্ষতি করা অপরাধ, আর তাতে আপনার জীবনও শেষ হয়ে যাবে। খুব রাগ বা কষ্ট থেকে এমন মনে হলে একটু থামুন, পরিবারের কাউকে বা গ্রামের মুরুব্বিকে বিষয়টা জানান। কেউ বিপদে থাকলে ৯৯৯ নম্বরে ফোন করুন।"),
]
personal = []
for msgs, reply in PERSONAL:
    for m in msgs:
        name = rng.choice(NAMES)
        personal.append({"kind": "personal", "fmt": "agent", "messages": agent_msgs(m), "answer": reply.format(n="")})
        personal.append({"kind": "personal", "fmt": "site", "messages": site_msgs(m, "", {"নাম": name}), "answer": reply.format(n=f"{name}, ")})
personal = personal * 3  # small set, seen a few times
assert all(no_bhai(e["answer"]) for e in personal) and all("৯৯৯" in e["answer"] for e in personal if "বাঁচতে" in e["messages"][-1]["content"])
print(len(personal), "personal examples")
