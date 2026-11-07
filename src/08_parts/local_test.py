import re, glob, random, pandas as pd, numpy as np
src=open("/home/claude/krishiguard/src/04_agent.py").read().split("\n")
SEED=42
find=lambda n: glob.glob(f'/tmp/claude-0/-home-claude/fc937821-ee15-5b53-b21d-baf42b676966/scratchpad/t06/in/**/{n}',recursive=True)[0]
exec("\n".join(src[43:84]))
i=61; j=next(k for k in range(i+1,len(src)) if src[k] and not src[k].startswith(" ")); exec("\n".join(src[i:j]))
exec(open("/home/claude/krishiguard/src/06_parts/b_tools.py").read())
exec("\n".join(src[371:393]).split("assert")[0]); exec("\n".join(src[391:404])); exec(src[430])
kt=pd.read_json(find("krishitools_300.jsonl"),lines=True)
exec(open("/home/claude/krishiguard/src/08_parts/b_messy.py").read())
exec(open("/home/claude/krishiguard/src/08_parts/c_canon.py").read())
M=build_messy(kt, seed=int(__import__("os").environ.get("MSEED", 42)))
print(M.change.value_counts().to_dict())
ASK=lambda p: not p  # lead-level: no dose sentence given
def lead_ok(lead,r):
    if r.expect=="ask": return not DOSE.search(lead)
    if r.expect=="product": return tool_correct_v4(lead,"product",None,gold_products_v2(r.crop,r.pest))
    return tool_correct(FORM.sub(" ",lead),r.expect,r.gold_dose,None)
res=[]
for r in M.itertuples():
    l3=tool_lookup(r.question)[1]; l4=tool_lookup_v4(r.question)[1]
    o3=tool_lookup(r.question_orig)[1]
    res.append((r.type,r.change,lead_ok(o3,r) if r.expect!="ask" else np.nan,lead_ok(l3,r),lead_ok(l4,r)))
R=pd.DataFrame(res,columns=["type","change","orig_v3","messy_v3","messy_v4"])
print(R.groupby("change")[["orig_v3","messy_v3","messy_v4"]].mean().round(2).to_string())
print(R[["orig_v3","messy_v3","messy_v4"]].mean().round(3).to_dict())
for r in M.sample(8,random_state=3).itertuples(): print(r.change,"|",r.question)
R["q"]=M.question.values; R["qo"]=M.question_orig.values
print("--- orig fails"); 
for x in R[R.orig_v3==0].itertuples(): print(x.type, x.change, "|", x.qo)
print("--- v4 messy fails")
for x in R[R.messy_v4==0].itertuples(): print(x.type, x.change, "|", x.q, "||", canonicalize(x.q)[-60:])
