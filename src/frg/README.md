# Fertilizer tables from the BARC Fertilizer Recommendation Guide 2024

The website's fertilizer answers come from chapter 7 of Hossain et al. (2024), *Fertilizer Recommendation Guide-2024*, Bangladesh Agricultural Research Council. The PDF itself is not in this repo.

Run from the repo root:

    pdftotext -layout FRG_English_30_10_2024.pdf src/frg/frg.txt
    python3 src/frg/parse.py    # 107 crop tables -> src/frg/tables.json, with checks
    python3 src/frg/build.py    # -> docs/frg.json (101 tables, 85 crops), with Bangla crop names

`parse.py` reads each table by the column position of its N, P, K, S, Mg, Zn and B headers, so rows that the PDF splits over two lines still land in the right column. Every table must have all four soil classes and one value per nutrient. It also checks that the ranges rise from Optimum to Very low. Ten tables fail that check because the guide's own ranges overlap or have typos (for example brinjal P "61-60"); they are kept as printed. Betel leaf, the seed-crop tables and fodder maize are left out. Fruit trees (chapter 7.8, dosed per plant by age) and the cropping-pattern tables (chapter 8, scanned images) are not parsed.

Without a soil test the website assumes medium fertility and shows the middle of each range, converted to fertilizer with the guide's appendix 10 factors (urea 2.17 × N, TSP 5 × P, MoP 2 × K, gypsum 5.6 × S, zinc sulphate monohydrate 2.78 × Zn, boric acid 5.88 × B, magnesium sulphate 10.5 × Mg).
