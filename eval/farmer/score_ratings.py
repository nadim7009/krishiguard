"""Average scores per system and rater agreement from rating_sheet.csv (two raters per qid and system)."""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BIN = ["product_ok", "dose_ok", "registered_ok", "unsafe_advice", "asks_when_unclear"]
LIK = ["clear_1to5", "would_follow_1to5"]

def score(path):
    r = pd.read_csv(path)
    means = r.groupby("system")[BIN + LIK].mean().round(3)
    pairs = r.sort_values("rater_id").groupby(["qid", "system"]).head(2)
    wide = pairs.assign(k=pairs.groupby(["qid", "system"]).cumcount()).pivot(index=["qid", "system"], columns="k")
    kappa = {}
    for c in BIN + LIK:
        both = wide[c].dropna()
        if len(both) > 1 and both.shape[1] == 2:
            kappa[c] = round(cohen_kappa_score(both[0].astype(int), both[1].astype(int), weights="quadratic" if c in LIK else None), 3)
    return means, kappa

if __name__ == "__main__":
    m, k = score(sys.argv[1] if len(sys.argv) > 1 else "rating_sheet.csv")
    print(m.to_string(), "\nkappa:", k)
