from __future__ import annotations
import io, math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st
from collections import Counter

st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

# -----------------------
# Core helpers & signals
# -----------------------
VTRAC: Dict[int, int] = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]: return "quint"
    if counts == [4,1]: return "quad"
    if counts == [3,2]: return "triple_double"
    if counts == [3,1,1]: return "triple"
    if counts == [2,2,1]: return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s).strip() if ch.isdigit()]

def parse_list(text: str, to_int: bool=False) -> List:
    raw = text.replace("\\n", ",").replace(" ", ",").replace("\\t", ",").replace(";", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if to_int:
        out = []
        for p in parts:
            try: out.append(int(p))
            except: pass
        return out
    return parts

def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    sd = digits_of(seed)
    sd2 = digits_of(prev_seed)
    sd3 = digits_of(prev_prev_seed)
    base = {
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd),
        "prev_sum_cat": sum_category(sum(sd)),
        "seed_vtracs": set(VTRAC[d] for d in sd),
        "mirror": MIRROR,
        "VTRAC": VTRAC,
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),
        "Counter": Counter, "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted,
        "seed_value": int(seed) if seed else None,
        "winner_structure": classify_structure(sd) if sd else ""
    }
    return base

def combo_env(base_env: Dict, combo: str) -> Dict:
    cd = digits_of(combo)
    env = dict(base_env)
    env.update({
        "combo": combo,
        "combo_digits": sorted(cd),
        "combo_sum": sum(cd),
        "combo_sum_cat": sum_category(sum(cd)),
        "combo_vtracs": set(VTRAC[d] for d in cd),
        "combo_structure": classify_structure(cd),
        "last_digit": cd[-1] if cd else None,
        "spread": (max(cd) - min(cd)) if cd else 0
    })
    return env

def load_pool_from_csv(f, col_hint="Result") -> List[str]:
    df = pd.read_csv(f)
    cols = {c.lower(): c for c in df.columns}
    if col_hint in df.columns:
        s = df[col_hint]
    elif "result" in cols:
        s = df[cols["result"]]
    elif "combo" in cols:
        s = df[cols["combo"]]
    else:
        raise ValueError("CSV must have 'Result' or 'Combo'.")
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_winners_csv(path: str) -> List[str]:
    try: df = pd.read_csv(path)
    except: return []
    cols = {c.lower(): c for c in df.columns}
    if "result" in cols:
        s = df[cols["result"]]
    elif "combo" in cols:
        s = df[cols["combo"]]
    else:
        return []
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_filters_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    if "id" not in df.columns and "fid" in df.columns: df["id"]=df["fid"]
    if "id" not in df.columns: df["id"]=range(1,len(df)+1)
    if "expression" not in df.columns: raise ValueError("Filters CSV needs 'expression'.")
    if "name" not in df.columns: df["name"]=df["id"].astype(str)
    if "parity_wiper" not in df.columns: df["parity_wiper"]=False
    if "enabled" not in df.columns: df["enabled"]=True
    return df

def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict) -> Tuple[set,int]:
    expr = str(row["expression"])
    try: code = compile(expr,"<expr>","eval")
    except: return set(),0
    elim=set()
    for c in pool:
        env=combo_env(base_env,c)
        try:
            if bool(eval(code,{"__builtins__":{}},env)): elim.add(c)
        except: pass
    return elim,len(elim)

def greedy_plan(candidates: pd.DataFrame, pool: List[str], base_env: Dict, beam_width:int, max_steps:int):
    remain=set(pool); chosen=[]
    for _ in range(max_steps):
        if not remain: break
        scored=[]
        for _,r in candidates.iterrows():
            elim,cnt=eval_filter_on_pool(r,list(remain),base_env)
            if cnt>0: scored.append((cnt,elim,r))
        if not scored: break
        scored.sort(key=lambda x:x[0],reverse=True)
        best_cnt,best_elim,best_row=max(scored[:beam_width],key=lambda x:x[0])
        remain-=best_elim
        chosen.append({"id":best_row["id"],"name":best_row.get("name",""),
                       "expression":best_row["expression"],
                       "eliminated_this_step":best_cnt,
                       "remaining_after":len(remain)})
        if best_cnt==0: break
    return chosen,sorted(list(remain))

# Sidebar
with st.sidebar:
    mode=st.radio("Mode",["Playlist Reducer","Safe Filter Explorer"],index=1)
    if mode=="Playlist Reducer":
        dmin,dbeam,dsteps=120,5,15
    else:
        dmin,dbeam,dsteps=60,6,18
    min_elims=st.number_input("Min eliminations for 'Large'",1,99999,value=dmin)
    beam_width=st.number_input("Beam width",1,50,value=dbeam)
    max_steps=st.number_input("Max steps",1,50,value=dsteps)
    exclude_parity=st.checkbox("Exclude parity-wipers",True)

# Seed & H/C/D
st.subheader("Seed context")
c1,c2,c3=st.columns(3)
seed=c1.text_input("Seed",value="")
prev_seed=c2.text_input("Prev Seed",value="")
prev_prev=c3.text_input("Prev Prev Seed",value="")
st.subheader("Hot/Cold/Due digits")
c4,c5,c6=st.columns(3)
hot=parse_list(c4.text_input("Hot digits"),True)
cold=parse_list(c5.text_input("Cold digits"),True)
due=parse_list(c6.text_input("Due digits"),True)

# Pool
st.subheader("Combo Pool")
pool_text=st.text_area("Paste combos:",height=120)
pool_file=st.file_uploader("Or upload pool CSV",type=["csv"])
if pool_text.strip(): pool=parse_list(pool_text,False)
elif pool_file: pool=load_pool_from_csv(pool_file)
else: st.stop()
st.write(f"Pool size: {len(pool)}")

# History
st.subheader("Winners History")
hist=st.text_input("History CSV path",value="DC5_Midday_Full_Cleaned_Expanded.csv")
winners_list=load_winners_csv(hist)

# Filters
st.subheader("Filters")
ids_txt=st.text_area("Paste Filter IDs (optional)",height=80)
filters_up=st.file_uploader("Or upload Filters CSV",type=["csv"])
filters_src=filters_up if filters_up else "lottery_filters_batch_10.csv"
filters_df=load_filters_csv(filters_src)
if exclude_parity and "parity_wiper" in filters_df: filters_df=filters_df[~filters_df["parity_wiper"]]
if "enabled" in filters_df: filters_df=filters_df[filters_df["enabled"]==True]
if ids_txt.strip():
    ids=set(parse_list(ids_txt,False))
    filters_df=filters_df[filters_df["id"].astype(str).isin(ids)]

# Base env
base=make_base_env(seed,prev_seed,prev_prev,hot,cold,due)

# Score
scored=[]
for _,r in filters_df.iterrows():
    elim,cnt=eval_filter_on_pool(r,pool,base)
    scored.append({"id":r["id"],"name":r.get("name",""),"expression":r["expression"],
                   "elim_count_on_pool":cnt,"elim_pct_on_pool":cnt/len(pool)*100 if pool else 0})
scored_df=pd.DataFrame(scored)
large=scored_df[scored_df["elim_count_on_pool"]>=min_elims].sort_values(by="elim_count_on_pool",ascending=False)
st.write(f"Large filters: {len(large)}")
st.dataframe(large,use_container_width=True)

plan,kept=greedy_plan(large[["id","name","expression"]],pool,base,int(beam_width),int(max_steps))
st.write(f"Kept combos: {len(kept)}/{len(pool)}")
if plan: st.dataframe(pd.DataFrame(plan))

import io
kept_csv=io.StringIO(); pd.DataFrame({"Result":kept}).to_csv(kept_csv,index=False)
removed=[x for x in pool if x not in kept]
removed_csv=io.StringIO(); pd.DataFrame({"Result":removed}).to_csv(removed_csv,index=False)
st.download_button("Download KEPT CSV",kept_csv.getvalue(),"kept_combos.csv","text/csv")
st.download_button("Download REMOVED CSV",removed_csv.getvalue(),"removed_combos.csv","text/csv")
