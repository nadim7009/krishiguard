import re, json
L = open('src/frg/frg.txt', encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(L) if l.strip() == '7.1 Cereal Crops')
end = next(i for i, l in enumerate(L) if i > start and re.match(r'\s*Cropping pattern', l))
TOK = re.compile(r'\d+(?:\.\d+)?\s*[–-]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])[–-](?![\w.])')
NUT = re.compile(r'(?<![A-Za-z])(N|P|K|S|Mg|Zn|B|Mo|Ca|Cu|Fe|Mn)(?![A-Za-z])')
CLASSES = ['Very low', 'Optimum', 'Medium', 'Low']
SKIP_SUB = ('Method', 'Optimum', 'Medium', 'Low', 'Very', 'Soil', 'Interpretation', 'Yield', 'Fertilizer', 'Usually', 'OF', 'Methods', 'Note')

def cls_of(s):
    return next((c for c in CLASSES if re.match(rf'^{c}\b', s.strip())), None)

def rng(t):
    if t in ('–', '-'): return (0.0, 0.0)
    a = re.split(r'\s*[–-]\s*', t); return (float(a[0]), float(a[-1]))

tables, crop, sub, section, yg = [], None, None, None, None
var, var_open = None, False
i = start
while i < end:
    raw = L[i]; s = raw.strip()
    if re.match(r'^7\.\d+ ', s): section = s; i += 1; continue
    m = re.match(r'^([A-Z][A-Z .,&/\'-]{2,}?)\s*(\(.*)?$', s)
    if m and len(m.group(1).strip()) > 2 and not NUT.fullmatch(m.group(1).strip()) and not s.startswith(('OF', 'LIV', 'FYM')):
        crop, sub, yg, var, var_open = m.group(1).strip(), None, None, None, False; i += 1; continue
    if re.match(r'^[A-Z][a-zA-Z]*\.?(?: [A-Za-z][A-Za-z.]*)*(?: \(.*\))?$', s) and len(s) < 45 and not s.startswith(SKIP_SUB):
        sub, var, var_open = s, None, False; i += 1; continue
    if s.startswith('Yield Goal'): yg = s[len('Yield Goal'):].strip(' :'); var_open = False; i += 1; continue
    if re.search(r'\(\s*(Var|Variety|LIV|All )|^\[?\s*\(?\s*Var', s) or (var is not None and var_open):
        var = (var + ' ' if var_open and var else '') + s; var_open = not re.search(r'\)\s*\]?\s*$', s); i += 1; continue
    if s.startswith('Yield Goal'): yg = s.split(':', 1)[-1].strip(); var_open = False; i += 1; continue
    if s.startswith('Interpretation') or (s.startswith('Soil Analysis') and 'Interpretation' not in L[i+1]):
        # header: nutrient names may sit on this line or the next one or two; OF/FYM column on any of them
        hdr = [L[j] for j in range(i, i + 4) if not cls_of(L[j])]
        cols, of_x = [], None
        for h in hdr:
            found = [(mm.group(1), (mm.start() + mm.end()) / 2) for mm in NUT.finditer(h)]
            if len(found) >= 3 and not cols: cols = found
        for h in [L[i - 1]] + hdr:
            for mm in re.finditer(r'OF\*?|FYM\*?|\(t/ha\)', h): of_x = max(of_x or 0, (mm.start() + mm.end()) / 2)
        j = i + 1
        while j < end and not cls_of(L[j]): j += 1
        rows, cur = {}, None
        while j < end:
            c = cls_of(L[j]); line = L[j]
            if c: cur = c; rows[c] = []; line = line.replace(c, ' ' * len(c), 1)
            elif re.search(r'[A-Za-z]', line) or (not line.strip() and len(rows) == 4): break
            if cur:
                for mm in TOK.finditer(line): rows[cur].append((mm.group(0), (mm.start() + mm.end()) / 2))
            j += 1
            if len(rows) == 4 and j < end and not L[j].strip(): break
        t = {'section': section, 'crop': crop, 'sub': sub, 'var': var, 'yield': yg, 'cols': None, 'rows': {}, 'line': i + 1, 'err': []}
        centers = [x for _, x in cols] + ([of_x] if of_x else [])
        names = [c for c, _ in cols]; t['cols'] = names
        if 'P' not in names and names[:1] == ['N'] and rows and all(len([1 for _, x in v if not of_x or abs(x - of_x) > 4]) == len(names) + 1 for v in rows.values()):
            names.insert(1, 'P')  # the PDF header lost the P column (onion)
        for c, toks in rows.items():
            vals = [tok for tok, x in toks if not of_x or abs(x - of_x) > 4]
            if len(vals) == len(names) and len(names) != len(cols):
                t['rows'][c] = dict(zip(names, vals)); continue
            got = {}
            for tok, x in toks:
                k = min(range(len(centers)), key=lambda k: abs(centers[k] - x))
                if k == len(cols) and c == 'Optimum': t['of'] = tok
                if k < len(cols):
                    if cols[k][0] in got: t['err'].append(f'{c}: two values for {cols[k][0]}')
                    got[cols[k][0]] = tok
            if len(got) < len(cols) and len(vals) == len(cols):  # columns drift on some pages: read the row in order
                t['err'] = [e for e in t['err'] if not e.startswith(c + ':')]; got = dict(zip(names, vals))
            t['rows'][c] = got
        for c in ['Optimum', 'Medium', 'Low', 'Very low']:
            if c not in t['rows']: t['err'].append(f'missing {c}'); continue
            miss = [n for n in t['cols'] if n not in t['rows'][c]]
            if miss: t['err'].append(f'{c}: no value for {miss}')
        if not t['err']:  # ranges must rise from Optimum to Very low
            for n in t['cols']:
                his = [rng(t['rows'][c][n])[1] for c in ['Optimum', 'Medium', 'Low', 'Very low']]
                los = [rng(t['rows'][c][n])[0] for c in ['Medium', 'Low', 'Very low']]
                if any(his[k + 1] < his[k] for k in range(3)) or any(los[k] < his[k] - 0.01 for k in range(3)):
                    t['err'].append(f'{n} ranges do not rise: {[t["rows"][c][n] for c in ["Optimum","Medium","Low","Very low"]]}')
        meth, k = [], j
        while k < end and k < j + 30:
            ln = L[k].strip()
            if meth and (cls_of(ln) or ln.startswith(('Soil Analysis', 'Yield Goal')) or re.match(r'^[A-Z][A-Z .]{3,}(\s*\(|$)', ln) or re.match(r'^\d{2,3}$', ln) and len(meth) > 8): break
            if ln.startswith('Method') or meth:
                if ln and not re.match(r'^\d{2,3}$', ln): meth.append(ln)
            k += 1
        t['method'] = ' '.join(meth)[:900]
        tables.append(t); var = None; i = j; continue
    i += 1
bad = [t for t in tables if t['err']]
print(len(tables), 'tables;', len(bad), 'flagged')
for t in bad: print('FLAG', t['line'], t['crop'], '|', t['sub'], '|', t['err'][:3])
json.dump(tables, open('src/frg/tables.json', 'w'), ensure_ascii=False, indent=0)
