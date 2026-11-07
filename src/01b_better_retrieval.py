# %% [markdown]
# ## Coverage check and a stronger encoder (bge-m3)

# %%
for name, df in eval_sets.items():
    print(name, "gold units with train/dev answers in their passage:",
          round(df["source_md"].isin(kb.index).mean(), 3))

def make_dense(model, prefix_q="", prefix_p=""):
    e = model.encode([prefix_p + t for t in chunks["text"]], batch_size=32,
                     normalize_embeddings=True, show_progress_bar=True).astype("float32")
    ix = faiss.IndexFlatIP(e.shape[1]); ix.add(e)
    def search(q, k=10):
        v = model.encode([prefix_q + q], normalize_embeddings=True).astype("float32")
        return to_units(ix.search(v, k * 5)[1][0], k)
    return search, e

def make_hybrid(dense_fn, rrf_k=60):
    def search(q, k=10):
        score = {}
        for ranked in (search_bm25(q, 50), dense_fn(q, 50)):
            for r, u in enumerate(ranked):
                score[u] = score.get(u, 0) + 1 / (rrf_k + r + 1)
        return sorted(score, key=score.get, reverse=True)[:k]
    return search

def run(systems):
    out = []
    for set_name, df in eval_sets.items():
        df = df[df["source_md"].isin(units[ID_COL])]
        for name, fn in systems.items():
            res, r = evaluate(df, fn, name)
            res["set"] = set_name
            out.append(res); per_q[(set_name, name)] = (df, r)
    return pd.DataFrame(out)[["set", "retriever", "n", "R@1", "R@5", "R@10", "MRR@10"]]

bge = SentenceTransformer("BAAI/bge-m3", device="cuda")
bge.max_seq_length = 512
bge.half()
search_bge, emb_bge = make_dense(bge)
all_res = pd.concat([res_df, run({"bge-m3": search_bge, "hybrid bm25+bge": make_hybrid(search_bge)})],
                    ignore_index=True)
print(all_res.round(3).to_string())

free_gpu("bge", "search_bge", "emb_bge")  # the search closure also holds the model

# %% [markdown]
# ## Fine-tune a Bangla retriever on the train questions
# Each train question is paired with the chunk of its source passage that holds its answer (answers were added to passages word for word). In-batch negatives, one epoch.

# %%
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

pairs = pd.concat([to_df(gen, "train"), to_df(trt, "train")], ignore_index=True)[["question", "answer", "source_md"]]
chunk_lookup = chunks.groupby("unit_id")["text"].apply(list)

def positive(ans, unit):
    cands = chunk_lookup.get(unit, [])
    for c in cands:
        if str(ans)[:60] in c:
            return c
    return cands[0] if cands else None

pairs["pos"] = [positive(a, u) for a, u in zip(pairs.answer, pairs.source_md)]
pairs = pairs.dropna(subset=["pos"]).sample(frac=1, random_state=42)
pairs = pairs.groupby("pos").head(3)  # fewer same-chunk pairs in a batch, fewer false negatives
print("training pairs:", len(pairs))

examples = [InputExample(texts=["query: " + q, "passage: " + p]) for q, p in zip(pairs.question, pairs.pos)]
ft = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
ft.max_seq_length = 256  # 512 tokens x batch 64 ran out of memory on a T4
ft.fit(train_objectives=[(DataLoader(examples, shuffle=True, batch_size=32),
                          losses.MultipleNegativesRankingLoss(ft))],
       epochs=1, warmup_steps=100, show_progress_bar=True)
ft.save(f"{OUT}/e5-small-krishi")

search_ft, emb_ft = make_dense(ft, "query: ", "passage: ")
all_res = pd.concat([all_res, run({"e5-small-ft": search_ft, "hybrid bm25+e5-ft": make_hybrid(search_ft)})],
                    ignore_index=True)
all_res.to_csv(f"{OUT}/retrieval_results.csv", index=False)
np.save(f"{OUT}/chunk_emb_e5small_ft.npy", emb_ft)
print(all_res.round(3).to_string())

dialect_r5 = pd.DataFrame({name: pd.Series((r <= 5).values, index=df["dialect"].values).groupby(level=0).mean()
                           for (s_, name), (df, r) in per_q.items() if s_ == "general_test"}).round(3)
dialect_r5.to_csv(f"{OUT}/retrieval_by_dialect.csv")
print(dialect_r5.to_string())

# %% [markdown]
# ## Fine-tune bge-m3 and tune fusion weights on dev
# Caveat: dev answers are part of the rebuilt passages, so dev slightly favours lexical matching (BM25). Weights are picked on dev only; test is scored once.

# %%
import itertools

free_gpu()
bge_ft = SentenceTransformer("BAAI/bge-m3", device="cuda")
bge_ft.max_seq_length = 256
bge_ft[0].auto_model.gradient_checkpointing_enable()
# full fine-tune of 568M params needs ~9 GB for weights/grads/Adam; the cached loss keeps 32 in-batch
# negatives but runs them in pieces of 8, so activations fit on a T4 (plain loss, batch 16 ran out of memory)
bge_ft.fit(train_objectives=[(DataLoader([InputExample(texts=[q, p]) for q, p in zip(pairs.question, pairs.pos)],
                                         shuffle=True, batch_size=32),
                              losses.CachedMultipleNegativesRankingLoss(bge_ft, mini_batch_size=8))],
           epochs=1, warmup_steps=50, use_amp=True, show_progress_bar=True)
bge_ft.save(f"{OUT}/bge-m3-krishi")
bge_ft.half()
search_bgeft, emb_bgeft = make_dense(bge_ft)
np.save(f"{OUT}/chunk_emb_bgem3_ft.npy", emb_bgeft)

def make_fusion(fns, weights, rrf_k=60):
    def search(q, k=10):
        score = {}
        for fn, w in zip(fns, weights):
            for r, u in enumerate(fn(q, 50)):
                score[u] = score.get(u, 0) + w / (rrf_k + r + 1)
        return sorted(score, key=score.get, reverse=True)[:k]
    return search

def fused_mrr(rank_lists, golds, w, rrf_k=60):
    total = 0.0
    for i, gold in enumerate(golds):
        score = {}
        for s, wt in w.items():
            for r, u in enumerate(rank_lists[s][i]):
                score[u] = score.get(u, 0) + wt / (rrf_k + r + 1)
        top = sorted(score, key=score.get, reverse=True)[:10]
        total += 1 / (top.index(gold) + 1) if gold in top else 0
    return total / len(golds)

systems = {"bm25": search_bm25, "bge-ft": search_bgeft, "e5-ft": search_ft}
dev = {n: to_df(d, "dev").sample(300, random_state=42) for n, d in (("general", gen), ("treatment", trt)) if "dev" in d}
ranked = {n: {s: [fn(q, 50) for q in df.question] for s, fn in systems.items()} for n, df in dev.items()}
grid = [dict(zip(systems, w)) for w in itertools.product([0, 0.5, 1, 2], repeat=len(systems)) if any(w)]
scores = pd.DataFrame([{**w, **{n: fused_mrr(ranked[n], dev[n].source_md.tolist(), w) for n in dev}} for w in grid])
scores["mean"] = scores[list(dev)].mean(axis=1)
best_w = scores.sort_values("mean").iloc[-1][list(systems)].to_dict()
print(scores.sort_values("mean").tail(5).round(3).to_string())
print("dev-tuned weights:", best_w)

fused = make_fusion([systems[s] for s in best_w], list(best_w.values()))
all_res = pd.concat([all_res, run({"bge-m3-ft": search_bgeft, "fusion (dev-tuned)": fused})], ignore_index=True)
all_res.to_csv(f"{OUT}/retrieval_results.csv", index=False)
json.dump(best_w, open(f"{OUT}/fusion_weights.json", "w"))
print(all_res.round(3).to_string())

dialect_r5 = pd.DataFrame({name: pd.Series((r <= 5).values, index=df["dialect"].values).groupby(level=0).mean()
                           for (s_, name), (df, r) in per_q.items() if s_ == "general_test"}).round(3)
dialect_r5.to_csv(f"{OUT}/retrieval_by_dialect.csv")
print(dialect_r5.to_string())
