import json, re
T = json.load(open("src/frg/tables.json"))
BN = {  # FRG crop name -> Bangla name and the words farmers use (longest match wins in the app)
 'RICE': ('ধান', ['ধান', 'ধানের', 'ধানে']), 'WHEAT': ('গম', ['গম']), 'MAIZE': ('ভুট্টা', ['ভুট্টা', 'ভুট্টায়']), 'BARLEY': ('যব', ['যব']),
 'PROSO MILLET': ('চিনা', ['চিনা']), 'FOXTAIL MILLET': ('কাউন', ['কাউন']), 'JUTE': ('পাট', ['পাট']), 'KENAF': ('কেনাফ', ['কেনাফ']), 'MESTA': ('মেস্তা', ['মেস্তা']),
 'COTTON': ('তুলা', ['তুলা']), 'LENTIL': ('মসুর', ['মসুর']), 'CHICKPEA': ('ছোলা', ['ছোলা']), 'MUNGBEAN': ('মুগ', ['মুগ']), 'BLACKGRAM': ('মাষকলাই', ['মাষকলাই', 'মাসকলাই']),
 'GRASSPEA': ('খেসারি', ['খেসারি']), 'COWPEA': ('ফেলন', ['ফেলন']), 'MUSTARD': ('সরিষা', ['সরিষা', 'সরিষায়', 'সর্ষে']), 'SESAME': ('তিল', ['তিল']),
 'GROUNDNUT': ('চিনাবাদাম', ['চিনাবাদাম', 'বাদাম']), 'SOYBEAN': ('সয়াবিন', ['সয়াবিন']), 'SUNFLOWER': ('সূর্যমুখী', ['সূর্যমুখী']), 'SAFFLOWER': ('কুসুম ফুল', ['কুসুম ফুল']),
 'LINSEED': ('তিসি', ['তিসি']), 'NIGER': ('গুজিতিল', ['গুজিতিল']), 'POTATO': ('আলু', ['আলু', 'আলুতে']), 'SWEET POTATO': ('মিষ্টি আলু', ['মিষ্টি আলু', 'মিষ্টিআলু']),
 'MUKHI KACHU': ('মুখীকচু', ['মুখীকচু', 'মুখী কচু']), 'PANI KACHU': ('পানিকচু', ['পানিকচু', 'পানি কচু']), 'CAULIFLOWER': ('ফুলকপি', ['ফুলকপি']), 'BROCCOLI': ('ব্রকলি', ['ব্রকলি', 'ব্রোকলি']),
 'CABBAGE': ('বাঁধাকপি', ['বাঁধাকপি', 'বাধাকপি']), 'CHINESE CABBAGE': ('চায়নিজ বাঁধাকপি', ['চায়নিজ বাঁধাকপি', 'চীনা কপি']), 'BATISAK': ('বাটিশাক', ['বাটিশাক']),
 'KANGKONG': ('গিমাকলমি', ['গিমাকলমি', 'কলমি']), 'INDIAN SPINACH': ('পুঁইশাক', ['পুঁইশাক', 'পুইশাক']), 'SPINACH': ('পালংশাক', ['পালংশাক', 'পালং']),
 'AMARANTHUS': ('ডাঁটা', ['ডাঁটা', 'ডাটা']), 'RED AMARANTH': ('লালশাক', ['লালশাক', 'লাল শাক']), 'TOMATO': ('টমেটো', ['টমেটো', 'টমাটো']), 'BRINJAL': ('বেগুন', ['বেগুন']),
 'OKRA': ('ঢেঁড়স', ['ঢেঁড়স', 'ঢেড়স', 'ঢেরস']), 'RADISH': ('মুলা', ['মুলা']), 'CARROT': ('গাজর', ['গাজর']), 'COUNTRY BEAN': ('শিম', ['শিম']),
 'YARDLONG BEAN': ('বরবটি', ['বরবটি']), 'FRENCH BEAN': ('ঝাড়শিম', ['ঝাড়শিম', 'ঝাড় শিম']), 'GARDEN PEA': ('মটরশুঁটি', ['মটরশুঁটি', 'মটর']),
 'SWEET GOURD': ('মিষ্টিকুমড়া', ['মিষ্টিকুমড়া', 'মিষ্টি কুমড়া']), 'ASH GOURD': ('চালকুমড়া', ['চালকুমড়া', 'চাল কুমড়া']), 'TEASLE GOURD': ('কাঁকরোল', ['কাঁকরোল', 'কাকরোল']),
 'BITTER GOURD': ('করলা', ['করলা']), 'POINTED GOURD': ('পটল', ['পটল']), 'SPONGE GOURD': ('ধুন্দুল', ['ধুন্দুল']), 'CUCUMBER': ('শসা', ['শসা', 'শশা']),
 'SQUASH': ('স্কোয়াশ', ['স্কোয়াশ']), 'ONION': ('পেঁয়াজ', ['পেঁয়াজ', 'পিঁয়াজ', 'পিয়াজ', 'পেয়াজ']), 'SUMMER ONION': ('গ্রীষ্মকালীন পেঁয়াজ', ['গ্রীষ্মকালীন পেঁয়াজ']),
 'GARLIC': ('রসুন', ['রসুন']), 'GINGER': ('আদা', ['আদা']), 'TURMERIC': ('হলুদ', ['হলুদ চাষ', 'হলুদের']), 'CHILLI': ('মরিচ', ['মরিচ']),
 'SWEET PEPPER': ('মিষ্টি মরিচ', ['মিষ্টি মরিচ', 'ক্যাপসিকাম']), 'CORIANDER': ('ধনিয়া', ['ধনিয়া', 'ধনে']), 'BLACK CUMIN': ('কালোজিরা', ['কালোজিরা']),
 'FENUGREEK': ('মেথি', ['মেথি']), 'PINEAPPLE': ('আনারস', ['আনারস']), 'STRAWBERRY': ('স্ট্রবেরি', ['স্ট্রবেরি']), 'WATERMELON': ('তরমুজ', ['তরমুজ']),
 'NETTED MELON': ('মাস্কমেলন', ['মাস্কমেলন', 'মেলন']), 'SUGARCANE': ('আখ', ['আখ', 'ইক্ষু']), 'SUGAR BEET': ('সুগারবিট', ['সুগারবিট']),
 'MARIGOLD': ('গাঁদা ফুল', ['গাঁদা']), 'ROSE': ('গোলাপ', ['গোলাপ']), 'TUBEROSE': ('রজনীগন্ধা', ['রজনীগন্ধা']), 'GERBERA': ('জারবেরা', ['জারবেরা']),
 'CHRYSANTHEMUM': ('চন্দ্রমল্লিকা', ['চন্দ্রমল্লিকা']), 'ORCHID': ('অর্কিড', ['অর্কিড']), 'GYPSOPHILA': ('জিপসোফিলা', ['জিপসোফিলা']),
 'NAPIER GRASS': ('নেপিয়ার ঘাস', ['নেপিয়ার']), 'PARA GRASS': ('পারা ঘাস', ['পারা ঘাস']), 'GERMAN GRASS': ('জার্মান ঘাস', ['জার্মান ঘাস']),
 'DHAL GRASS': ('ঢল ঘাস', ['ঢল ঘাস']), 'GUINEA GRASS': ('গিনি ঘাস', ['গিনি ঘাস']), 'JUMBO': ('জাম্বো ঘাস', ['জাম্বো']), 'ROZI': ('রুজি ঘাস', ['রুজি ঘাস']), 'OAT': ('ওট', ['ওট']),
}
SUB_BN = {'Boro rice': 'বোরো', 'Aus rice': 'আউশ', 'T. Aman rice': 'রোপা আমন', 'B. Aman rice': 'বোনা আমন'}
CLS = ['Optimum', 'Medium', 'Low', 'Very low']
def rng(t):
    if t in ('–', '-'): return [0, 0]
    a = [float(x) for x in re.split(r'\s*[–-]\s*', t)]; return [a[0], a[-1]]
out, last_method = [], {}
for i, t in enumerate(T):
    if t['crop'] not in BN or t['err'] and not all('do not rise' in e for e in t['err']) or 'Seed' in (t['yield'] or ''):
        continue
    if t['crop'] == 'MAIZE' and t['section'].startswith('7.14'): continue  # fodder maize: kept out so "ভুট্টা" means grain maize
    if t['crop'] in ('JUTE', 'MESTA') and 'kg/ha' in (t['yield'] or ''): continue  # seed crops
    meth = re.split(r'\s(?=[A-Z][a-z]+ [A-Za-z ]*\((?:Var|All))', t['method'])[0].strip() or last_method.get(t['crop'], '')
    if meth: last_method[t['crop']] = meth
    var = re.sub(r'^[\[\s(]*(Var(iety)?[.:]?|LIV:?)\s*', '', (t['var'] or '').strip()).rstrip(')] ') if t['var'] else ''
    bn, kw = BN[t['crop']]
    out.append({'c': t['crop'], 'bn': bn, 'kw': kw, 's': SUB_BN.get(t['sub'], ''), 'v': var, 'y': t['yield'] or '',
                'n': {k: [rng(t['rows'][c][k]) for c in CLS] for k in t['cols']}, 'of': t.get('of'), 'm': meth, 'sec': t['section'][4:].strip()})
# hybrid tomato and tossa jute need a word to pick them
for o in out:
    if o['c'] == 'TOMATO' and 'Hybrid' in o['v']: o['s'] = 'হাইব্রিড'
    if o['c'] == 'JUTE': o['s'] = 'তোষা' if 'Tossa' in o['v'] or 'O 4' in o['v'] else 'দেশি'
json.dump({'source': 'BARC Fertilizer Recommendation Guide 2024 (Hossain et al., 2024), chapter 7', 'tables': out},
          open('docs/frg.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print(len(out), 'tables,', len({o['c'] for o in out}), 'crops')
assert next(o for o in out if o['c'] == 'RICE' and o['s'] == 'বোরো')['n']['N'][1] == [59, 116]
assert next(o for o in out if o['c'] == 'POTATO')['n']['K'][1] == [46, 90]
