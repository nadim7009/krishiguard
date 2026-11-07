"""src/NN_name.py (# %% cells) -> NN_name.ipynb at the repo root, for Kaggle's import-by-link."""
import json, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
cells = []
for blk in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = blk.partition("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").split("\n"))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.strip("\n").splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cells), "cells ->", sys.argv[2])
