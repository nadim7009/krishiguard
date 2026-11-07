# %% [markdown]
# # KrishiGuard 01: data and retrieval
#
# Loads KrishokChat, builds three retrievers over the 2,946 semantic units (BM25, multilingual-e5-small, hybrid) and measures how often the gold source unit comes back in the top k.
#
# Kaggle settings: GPU T4 (CPU also works, just slower for embeddings), Internet on, secret `HF_TOKEN`.

# %%
!pip install -q rank_bm25 faiss-cpu sentence-transformers

# %%
import os, re, json, random
import numpy as np, pandas as pd
from datasets import load_dataset

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass  # public dataset, token only avoids rate limits

REPO = "RaiyanKhaan/KrishokChat"
OUT = "/kaggle/working/krishiguard"
os.makedirs(OUT, exist_ok=True)
random.seed(42); np.random.seed(42)


def free_gpu(*names):
    """Drop the named globals and release their GPU memory: free_gpu("bge", "search_bge")."""
    import gc, sys, torch
    for n in names:
        globals().pop(n, None)
    sys.last_traceback = sys.last_value = sys.last_type = None  # a failed cell's traceback pins GPU tensors
    gc.collect(); torch.cuda.empty_cache()
    print("GPU in use:", [f"{torch.cuda.memory_allocated(i) / 1e9:.2f} GB" for i in range(torch.cuda.device_count())])

# %% [markdown]
# ## Load the subsets
# The HF viewer fails on `semantic_units`, so the loader falls back to reading the raw json files from the repo.

# %%
from huggingface_hub import list_repo_files, hf_hub_download

def load_subset(name):
    try:
        return load_dataset(REPO, name)
    except Exception as e:
        print(f"{name}: load_dataset failed ({type(e).__name__}), reading raw files")
        files = [f for f in list_repo_files(REPO, repo_type="dataset")
                 if name in f and f.endswith((".json", ".jsonl"))]
        print("  files:", files)
        dfs = [pd.read_json(hf_hub_download(REPO, f, repo_type="dataset"),
                            lines=f.endswith(".jsonl")) for f in files]
        return {"train": pd.concat(dfs, ignore_index=True)}

def to_df(ds, split):
    d = ds[split]
    return d if isinstance(d, pd.DataFrame) else d.to_pandas()

gen = load_subset("general_qa")
trt = load_subset("treatment_qa")
units_ds = load_subset("semantic_units")
farmer = load_subset("farmer_eval_350")

gen_test = to_df(gen, "test")
units = to_df(units_ds, "train")
print("general test:", gen_test.shape, "| units:", units.shape)
print("unit columns:", list(units.columns))
print("treatment splits:", list(trt.keys()), "| farmer splits:", list(farmer.keys()))

# %%
trt_test = to_df(trt, "test") if "test" in trt else None
if trt_test is not None:
    print("treatment test:", trt_test.shape)
print(gen_test.groupby("dialect").size())
print(gen_test["category"].value_counts().head(12))

# %% [markdown]
# ## Rebuild the passage text
# `semantic_units` only ships a 512-character preview (`sorted_content`, marked "[truncated]"); full passages average tens of thousands of characters and are not in the repo.
# KrishokChat answers are verbatim extracts from those passages, so each passage is rebuilt from its title, its preview and every train/dev answer drawn from it. Test answers are never used.

# %%
ID_COL = "file_path"  # QA rows point to units through source_md == file_path
assert gen_test["source_md"].isin(units[ID_COL]).all(), "gold units missing from corpus"

def answers_by_unit(ds):
    # train/dev only; "full" contains test rows, "test" is held out
    parts = [to_df(ds, s) for s in ds.keys() if s in ("train", "dev")]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby("source_md")["answer"].apply(lambda a: "\n".join(dict.fromkeys(a)))

kb = pd.concat([answers_by_unit(gen), answers_by_unit(trt)]).groupby(level=0).apply("\n".join)
units = units.drop_duplicates(ID_COL).reset_index(drop=True)
preview = units["sorted_content"].astype(str).str.replace(r"!\[.*?\]\(.*?\)|<!--.*?-->|\[truncated\]", " ", regex=True)
units["text"] = (units["title"].astype(str) + "\n" + preview + "\n" + units[ID_COL].map(kb).fillna("")).str.strip()
print("units with rebuilt text:", units[ID_COL].isin(kb.index).mean().round(3))
print(units["text"].str.len().describe().round(0))

# %% [markdown]
# ## Chunking
# Long units are split into ~1,200 character windows with overlap. A chunk hit counts as a hit for its parent unit.

# %%
def chunk(text, size=1200, overlap=200):
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text) - overlap, size - overlap)]

rows = [(u, c) for u, t in zip(units[ID_COL], units["text"]) for c in chunk(t)]
chunks = pd.DataFrame(rows, columns=["unit_id", "text"])
print("chunks:", len(chunks))

# %% [markdown]
# ## Retrievers

# %%
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def tokenize(s):
    s = s.translate(BN_DIGITS).lower()
    return re.findall(r"[ঀ-৿]+|[a-z0-9.]+", s)

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([tokenize(t) for t in chunks["text"]])

from sentence_transformers import SentenceTransformer
import faiss, torch
enc = SentenceTransformer("intfloat/multilingual-e5-small",
                          device="cuda" if torch.cuda.is_available() else "cpu")
emb = enc.encode(["passage: " + t for t in chunks["text"]], batch_size=64,
                 normalize_embeddings=True, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

def to_units(chunk_ids, k):
    seen = []
    for i in chunk_ids:
        u = chunks.unit_id[i]
        if u not in seen:
            seen.append(u)
        if len(seen) == k:
            break
    return seen

def search_bm25(q, k=10):
    return to_units(np.argsort(-bm25.get_scores(tokenize(q)))[:k * 5], k)

def search_dense(q, k=10):
    v = enc.encode(["query: " + q], normalize_embeddings=True).astype("float32")
    return to_units(index.search(v, k * 5)[1][0], k)

def search_hybrid(q, k=10, rrf_k=60):
    score = {}
    for ranked in (search_bm25(q, 50), search_dense(q, 50)):
        for r, u in enumerate(ranked):
            score[u] = score.get(u, 0) + 1 / (rrf_k + r + 1)
    return sorted(score, key=score.get, reverse=True)[:k]

# %% [markdown]
# ## Retrieval evaluation

# %%
def evaluate(df, fn, name):
    ranks = []
    for q, gold in zip(df["question"], df["source_md"]):
        got = fn(q, 10)
        ranks.append(got.index(gold) + 1 if gold in got else None)
    r = pd.Series(ranks, dtype="float")
    return {"retriever": name, "n": len(r),
            "R@1": (r <= 1).mean(), "R@5": (r <= 5).mean(), "R@10": (r <= 10).mean(),
            "MRR@10": (1 / r).fillna(0).mean()}, r

results, per_q = [], {}
eval_sets = {"general_test": gen_test}
if trt_test is not None:
    eval_sets["treatment_test"] = trt_test
for set_name, df in eval_sets.items():
    df = df[df["source_md"].isin(units[ID_COL])]
    for name, fn in [("bm25", search_bm25), ("e5-small", search_dense), ("hybrid", search_hybrid)]:
        res, r = evaluate(df, fn, name)
        res["set"] = set_name
        results.append(res); per_q[(set_name, name)] = (df, r)

res_df = pd.DataFrame(results)[["set", "retriever", "n", "R@1", "R@5", "R@10", "MRR@10"]]
res_df.to_csv(f"{OUT}/retrieval_results.csv", index=False)
res_df.round(3)

# %%
# recall@5 by dialect, hybrid, general test
df, r = per_q[("general_test", "hybrid")]
by_dialect = (pd.DataFrame({"dialect": df["dialect"].values, "hit5": (r <= 5).values})
              .groupby("dialect")["hit5"].agg(["mean", "count"]).round(3))
by_dialect.to_csv(f"{OUT}/retrieval_by_dialect.csv")
by_dialect

# %% [markdown]
# ## Save artifacts for later notebooks
# After this finishes: Save Version, then "New Dataset" from the output and name it `krishiguard-index`.

# %%
chunks.to_parquet(f"{OUT}/chunks.parquet")
np.save(f"{OUT}/chunk_emb_e5small.npy", emb)
units[[ID_COL, "text"]].rename(columns={ID_COL: "unit_id"}).to_parquet(f"{OUT}/units.parquet")
json.dump({"id_col": ID_COL, "text": "title + preview + train/dev answers", "encoder": "intfloat/multilingual-e5-small",
           "chunk_size": 1200, "overlap": 200}, open(f"{OUT}/config.json", "w"), indent=2)
print(os.listdir(OUT))
