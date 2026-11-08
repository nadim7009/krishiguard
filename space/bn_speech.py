"""Bangla text as MMS-TTS (facebook/mms-tts-ben) can say it.

The model only knows Bangla letters: digits, Latin letters and punctuation are silently dropped.
So numbers become Bangla words, product names and codes in Latin letters are written in Bangla letters,
and the text is cut into short phrases at punctuation (each phrase is spoken separately, with a pause between).
"""
import re, unicodedata

ONES = ("শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ "
        "বিশ একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ ঊনত্রিশ ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ "
        "চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ ঊনপঞ্চাশ পঞ্চাশ একান্ন বাহান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন ঊনষাট "
        "ষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি ঊনসত্তর সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর ঊনআশি "
        "আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি অষ্টাশি ঊননব্বই নব্বই একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই").split()
assert len(ONES) == 100
HUNDREDS = ["", "একশো", "দুইশো", "তিনশো", "চারশো", "পাঁচশো", "ছয়শো", "সাতশো", "আটশো", "নয়শো"]

def int_words(n):
    if n < 100:
        return ONES[n]
    parts = []
    for size, name in ((10**7, "কোটি"), (10**5, "লাখ"), (1000, "হাজার")):
        if n >= size:
            parts.append(f"{int_words(n // size)} {name}"); n %= size
    if n >= 100:
        parts.append(HUNDREDS[n // 100]); n %= 100
    if n:
        parts.append(ONES[n])
    return " ".join(parts)

def num_words(m):
    s = m.group(0).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).replace(",", "")
    whole, _, frac = s.partition(".")
    out = int_words(int(whole or 0))
    return out + (" দশমিক " + " ".join(ONES[int(d)] for d in frac) if frac else "")

LETTER = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "এ বি সি ডি ই এফ জি এইচ আই জে কে এল এম এন ও পি কিউ আর এস টি ইউ ভি ডাব্লিউ এক্স ওয়াই জেড".split()))
CONS = [("sh", "শ"), ("ch", "চ"), ("th", "থ"), ("ph", "ফ"), ("kh", "খ"), ("gh", "ঘ"), ("ck", "ক"), ("qu", "কু"),
        ("b", "ব"), ("c", "ক"), ("d", "ড"), ("f", "ফ"), ("g", "গ"), ("h", "হ"), ("j", "জ"), ("k", "ক"), ("l", "ল"), ("m", "ম"), ("n", "ন"),
        ("p", "প"), ("q", "ক"), ("r", "র"), ("s", "স"), ("t", "ট"), ("v", "ভ"), ("w", "ও"), ("x", "ক্স"), ("y", "ই"), ("z", "জ")]
VOW = {"a": ("আ", "া"), "e": ("এ", "ে"), "i": ("ই", "ি"), "o": ("ও", "ো"), "u": ("উ", "ু")}

def latin_word(w):
    if w.isupper() and len(w) <= 4 or len(w) == 1:  # codes like WDG, EC: spell the letters
        return " ".join(LETTER.get(c, "") for c in w.upper())
    w, out, prev_cons, i = w.lower(), "", False, 0
    while i < len(w):
        c = w[i]
        if c in VOW:
            out += VOW[c][1] if prev_cons else VOW[c][0]; prev_cons = False; i += 1; continue
        for k, v in CONS:
            if w.startswith(k, i):
                out += ("্" if prev_cons else "") + v; prev_cons = True; i += len(k); break
        else:
            i += 1
    return out

def speakable(text):
    """Bangla text -> list of short phrases using only letters the model knows."""
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)  # 25WDG -> 25 WDG
    t = t.replace("%", " শতাংশ ")
    t = re.sub(r"[০-৯0-9][০-৯0-9,]*(?:\.[০-৯0-9]+)?", num_words, t)
    t = re.sub(r"[A-Za-z]+", lambda m: latin_word(m.group(0)), t)
    phrases = [p.strip() for p in re.split(r"[।?!.,;:\n()\[\]\"“”]+", t)]
    # the tokenizer keeps decomposed letters (ড + ়), and drops anything outside its vocabulary
    return [unicodedata.normalize("NFD", p) for p in phrases if re.search(r"[ঀ-৿]", p)]

if __name__ == "__main__":
    assert int_words(0) == "শূন্য" and int_words(16) == "ষোলো" and int_words(100) == "একশো" and int_words(250) == "দুইশো পঞ্চাশ"
    assert int_words(1500) == "এক হাজার পাঁচশো" and int_words(4706) == "চার হাজার সাতশো ছয়"
    assert speakable("০.৫ মিলি") == [unicodedata.normalize("NFD", "শূন্য দশমিক পাঁচ মিলি")]
    for s in ["নিবন্ধিত মাত্রা প্রতি হেক্টরে ১০০ গ্রাম। ১৬ লিটারের ট্যাংকে ৮ মিলি।", "Taba ২৫WDG, Tared 280SC, Amistar Top: ৯৮৩টি পণ্য। বৃষ্টির সম্ভাবনা ৭০%।"]:
        print([unicodedata.normalize("NFC", p) for p in speakable(s)])
