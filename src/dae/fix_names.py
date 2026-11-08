"""Fill in active ingredient names the first DAE parse lost (notebook 03).

In the DAE list (up to the 81st PTAC meeting) the "Common Name" column is a merged cell. The notebook 03
parser took the nearest non-number cell, so 608 products got a strength like "(64% + 4%)" as their name,
and 58 got a brand name ("89 Zeen 1 EC"). This script reads the same PDF again and rebuilds each block's
name: a name starts at a cell with letters, continues over following rows while it ends with "+", and
percentage-only text inside a block belongs to one product, not to a new name. Where a name's strengths
do not add up to the number in the brand (Diamond 68WP vs "Mancozeb (64%) + Metalaxyl (8%)"), only the
ingredient names are kept. Checked against the 4,088 products the first parse named: the two agree on
4,027; in the other 61 the first parse was wrong (brand names or the next block's name).

    python src/dae/fix_names.py dae_registered_pesticides.pdf docs/dae.json
"""
import json, re, sys
import pdfplumber

REG = re.compile(r"^AP-?\s?\d+")
clean = lambda c: re.sub(r"\s+", " ", (c or "")).strip()
alpha = lambda s: bool(re.search(r"[A-Za-z]{3}", s or ""))
pct_only = lambda s: bool(s) and not alpha(s) and "%" in s
names_only = lambda n: re.sub(r"\s*/\s*\+", " +", re.sub(r"\s+", " ", re.sub(r"\(?\s*[\d.]+\s*%\s*\)?", "", n))).replace(" )", "").strip(" +")


def block_names(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for t in page.extract_tables():
                for row in t:
                    row = [clean(c) for c in row]
                    if any("Common Name" in c for c in row):
                        continue
                    rows.append((row, next((i for i, c in enumerate(row) if REG.match(c)), None)))
    cur, group, out, own, ncol = None, [], {}, {}, None

    def flush():
        for g in group:
            out[g] = cur

    for row, r in rows:
        if r is not None and r >= 3:
            ncol = r - 3
        if ncol is None or ncol >= len(row):
            continue
        cell, cont = row[ncol], cur is not None and cur.rstrip().endswith("+")
        if r is not None:
            reg, sl = row[r].replace(" ", ""), row[r - 2]
            if alpha(cell):
                if cont:
                    cur = cur + " " + cell
                else:
                    flush(); cur, group = cell, []
            elif sl == "1" and not cont:
                flush(); cur, group = None, []
                if pct_only(cell):
                    own[reg] = cell
            elif pct_only(cell):
                if cont:
                    cur = cur + " " + cell
                else:
                    own[reg] = cell
            group.append(reg)
        elif cell and (alpha(cell) or cont):
            cur = cell if cur is None else cur + " " + cell
    flush()
    return out, own


def strength(brand):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:EC|WP|SC|WDG|WG|G|GR|SL|EW|CS|OD|SP|DF|FS|ME|ZC|DP|SG|WS|SE|FW|D|L)\b", brand, re.I)
    return float(m.group(1)) if m else None


def main(pdf_path, dae_path):
    names, own = block_names(pdf_path)
    d = json.load(open(dae_path, encoding="utf-8"))
    ing, key = d["ingredients"], lambda n: re.sub(r"[^a-z]", "", names_only(n or "").lower())
    first = {}
    for r in d["rows"]:
        first.setdefault(r[2], (r[0], ing[r[1]]))
    fixed = {}
    s_ok = lambda n, brand: (lambda p, st: bool(p) and st is not None and abs(sum(p) - st) < 0.6)([float(v) for v in re.findall(r"([\d.]+)\s*%", n)], strength(brand))
    for reg, (brand, old) in first.items():
        new = names.get(reg)
        if not new:
            continue
        junk = not alpha(old) or re.search(r"\d+\s*(EC|WP|SC|SL|G)\b", old)
        cut_short = old.rstrip().endswith("+") and s_ok(new, brand)
        if junk or cut_short:  # otherwise the first parse's name stays as it was
            fixed[reg] = new if (reg not in own and s_ok(new, brand)) else names_only(new)
    idx = {n: i for i, n in enumerate(ing)}
    for r in d["rows"]:
        if r[2] in fixed:
            n = fixed[r[2]]
            if n not in idx:
                idx[n] = len(ing); ing.append(n)
            r[1] = idx[n]
    d["source"] += "; ingredient names rebuilt by src/dae/fix_names.py"
    json.dump(d, open(dae_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    left = sum(1 for r in d["rows"] if not alpha(ing[r[1]]))
    print(f"{len(fixed)} products renamed; {left} rows still without a name")


if __name__ == "__main__":
    main(*sys.argv[1:3])
