
from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page & defaults
# -------------------------------------------------
st.set_page_config(page_title="Archetype Helper — Large Filters Planner", layout="wide")
st.title("Archetype Helper — Large Filters, Triggers & Plans")

# -----------------------
# Core helpers
# -----------------------
VTRAC: Dict[int, int] = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs)
    counts = sorted(c.values(), reverse=True)
    if counts == [5]: return "quint"
    if counts == [4,1]: return "quad"
    if counts == [3,2]: return "triple_double"
    if counts == [3,1,1]: return "triple"
    if counts == [2,2,1]: return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s).strip() if ch.isdigit()]

def parse_list(text: str, to_int: bool=False) -> List[str]:
    """Split on commas, spaces, or newlines."""
    raw = (
        text.replace("\n", ",")
            .replace(" ", ",")
            .replace("\t", ",")
            .replace(";", ",")
    )
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if to_int:
        out = []
        for p in parts:
            try: out.append(int(p))
            except: pass
        return out
    return parts

def norm_id(s: str) -> str:
    """Normalize an ID for robust matching (upper, strip, drop trailing S)."""
    s = str(s).strip().upper()
    for ch in (" ", ",", "'", '"'): s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch.isalnum())
    if s.endswith("S"): s = s[:-1]
    return s

# ---------------------------------------
# Seed / context env builders
# ---------------------------------------
def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    sd  = digits_of(seed) if seed else []
    sd2 = digits_of(prev_seed) if prev_seed else []
    sd3 = digits_of(prev_prev_seed) if prev_prev_seed else []

    return {
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "seed_sum": sum(sd) if sd else 0,
        "prev_seed_sum": sum(sd2) if sd2 else 0,
        "prev_prev_seed_sum": sum(sd3) if sd3 else 0,
        "seed_structure": classify_structure(sd) if sd else "",
        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),
        "mirror": MIRROR, "VTRAC": VTRAC,
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),
        "Counter": Counter, "any": any, "all": all, "len": len, "sum": sum, "max": max, "min": min, "set": set, "sorted": sorted
    }

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

# ---------------------------------------
# CSV loaders
# ---------------------------------------
def load_pool_from_csv(f, col_name_hint: str) -> List[str]:
    df = pd.read_csv(f)
    cols_lower = {c.lower(): c for c in df.columns}
    if col_name_hint and col_name_hint in df.columns:
        s = df[col_name_hint]
    elif "result" in cols_lower:
        s = df[cols_lower["result"]]
    elif "combo" in cols_lower:
        s = df[cols_lower["combo"]]
    else:
        raise ValueError("Pool CSV must contain 'Result' or 'Combo' column.")
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_filters_csv(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)): df = pd.read_csv(source)
    else: df = pd.read_csv(source)
    if "id" not in df.columns and "fid" in df.columns:
        df["id"] = df["fid"]
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)
    df["id_norm"] = df["id"].astype(str).map(norm_id)
    if "enabled" in df.columns:
        df = df[df["enabled"] == True]
    return df

# ---------------------------------------
# Filter evaluation
# ---------------------------------------
def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict) -> Tuple[Set[str], int]:
    expr = str(row["expression"])
    try: code = compile(expr, "<filter_expr>", "eval")
    except: return set(), 0
    eliminated = set()
    for c in pool:
        try:
            if bool(eval(code, {"__builtins__": {}}, combo_env(base_env, c))):
                eliminated.add(c)
        except: pass
    return eliminated, len(eliminated)

def greedy_plan(candidates: pd.DataFrame, pool: List[str], base_env: Dict,
                beam_width: int, max_steps: int) -> Tuple[List[Dict], List[str]]:
    remaining = set(pool); chosen = []
    for _ in range(max_steps):
        if not remaining: break
        scored = []
        for _, r in candidates.iterrows():
            elim, cnt = eval_filter_on_pool(r, list(remaining), base_env)
            if cnt > 0: scored.append((cnt, elim, r))
        if not scored: break
        scored.sort(key=lambda x: x[0], reverse=True)
        best_cnt, best_elim, best_row = max(scored[:beam_width], key=lambda x: x[0])
        remaining -= best_elim
        chosen.append({
            "id": best_row["id"], "name": best_row.get("name",""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining)
        })
        if best_cnt == 0: break
    return chosen, sorted(list(remaining))

# -------------------------------------------------
# SIDEBAR: Planner Mode
# -------------------------------------------------
with st.sidebar:
    st.header("Planner Mode")
    mode = st.radio("Select Mode", ["Playlist Reducer", "Safe Filter Explorer"], index=1)
    default_min_elims = 120 if mode=="Playlist Reducer" else 60
    default_beam = 5 if mode=="Playlist Reducer" else 6
    default_steps = 15 if mode=="Playlist Reducer" else 18
    min_elims = st.number_input("Min eliminations to call it ‘Large’", 1, 99999, default_min_elims)
    beam_width = st.number_input("Beam width (search breadth)", 1, 50, default_beam)
    max_steps = st.number_input("Max steps (search depth)", 1, 50, default_steps)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# -------------------------------------------------
# Inputs
# -------------------------------------------------
st.subheader("Seed context")
c1, c2, c3 = st.columns(3)
seed = c1.text_input("Seed (1-back, 5 digits)", "")
prev_seed = c2.text_input("Prev seed (2-back, optional)", "")
prev_prev = c3.text_input("Prev-prev seed (3-back, optional)", "")
if seed and len(seed.strip())!=5:
    st.error("Seed must be exactly 5 digits."); st.stop()

st.subheader("Hot/Cold/Due digits")
c4,c5,c6 = st.columns(3)
hot_digits = parse_list(c4.text_input("Hot digits", ""), to_int=True)
cold_digits = parse_list(c5.text_input("Cold digits", ""), to_int=True)
due_digits = parse_list(c6.text_input("Due digits", ""), to_int=True)

# -------------------------------------------------
# Combo Pool
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma/space/newline separated):", height=130)
pool_file = st.file_uploader("Or upload pool CSV", type=["csv"])
pool_col_hint = st.text_input("Pool column name (optional hint)", "Result")
pool=[]
if pool_text.strip(): pool = parse_list(pool_text)
elif pool_file: pool = load_pool_from_csv(pool_file, pool_col_hint)
st.write(f"Pool size: {len(pool)}")

# -------------------------------------------------
# Filters
# -------------------------------------------------
st.subheader("Filters")
ids_text = st.text_area("Paste applicable Filter IDs (optional):", height=90)
filters_file_up = st.file_uploader("Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)", type=["csv"])
filters_source = filters_file_up if filters_file_up else "lottery_filters_batch_10.csv"

try:
    filters_df_full = load_filters_csv(filters_source)
    loaded_ok = True
except Exception as e:
    loaded_ok = False
    filters_df_full = pd.DataFrame(columns=["id","id_norm","expression","name"])
    st.warning(f"Could not load filters CSV ({e}). You can upload a CSV above.")

if exclude_parity and "parity_wiper" in filters_df_full.columns:
    filters_df_full = filters_df_full[~filters_df_full["parity_wiper"]]

applicable_ids = set(norm_id(x) for x in parse_list(ids_text)) if ids_text.strip() else set()
if applicable_ids:
    filters_df = filters_df_full[filters_df_full["id_norm"].isin(applicable_ids)].copy()
    st.caption(f"Matched {len(filters_df)} of {len(applicable_ids)} pasted IDs.")
    if len(filters_df) == 0 and loaded_ok:
        known_sample = ", ".join(filters_df_full["id"].astype(str).head(8).tolist())
        st.warning("None of the pasted IDs matched the Filters CSV. "
                   "I normalize for spaces/commas/trailing 'S', but if it still fails, "
                   f"check ID formats. Sample CSV IDs: {known_sample}")
else:
    filters_df = filters_df_full.copy()

st.write(f"Filters loaded: {len(filters_df)}")

# -------------------------------------------------
# RUN
# -------------------------------------------------
if st.button("▶ Run Planner", use_container_width=False):
    if not len(pool):
        st.error("No pool loaded. Paste combos or upload a pool CSV first."); st.stop()
    if filters_df.empty:
        st.error("No filters to evaluate. Paste valid IDs or load a filters CSV."); st.stop()

    base_env = make_base_env(seed,prev_seed,prev_prev,hot_digits,cold_digits,due_digits)
    scored_rows=[]
    for _,r in filters_df.iterrows():
        elim,cnt=eval_filter_on_pool(r,pool,base_env)
        scored_rows.append({
            "id":r["id"], "name":r.get("name",""), "expression":r["expression"],
            "elim_count_on_pool":cnt, "elim_pct_on_pool": (cnt/len(pool)*100) if pool else 0
        })
    scored_df=pd.DataFrame(scored_rows)

    if scored_df.empty:
        st.warning("No filters evaluated."); st.stop()

    large_df=scored_df[scored_df["elim_count_on_pool"]>=int(min_elims)].copy()
    large_df=large_df.sort_values(by=["elim_count_on_pool","elim_pct_on_pool"], ascending=False)
    st.write(f"Large filters ≥{min_elims}: {len(large_df)}")
    st.dataframe(large_df[["id","name","elim_count_on_pool","elim_pct_on_pool"]], use_container_width=True)

    if large_df.empty:
        kept_after=pool; plan=[]
    else:
        plan, kept_after = greedy_plan(
            large_df[["id","name","expression"]],
            pool, base_env, int(beam_width), int(max_steps))

    st.write(f"Kept combos after plan: {len(kept_after)} / {len(pool)}")
    if plan: st.dataframe(pd.DataFrame(plan), use_container_width=True)

    st.subheader("Downloads")
    kept_df = pd.DataFrame({"Result": kept_after})
    removed_df = pd.DataFrame({"Result": sorted(set(pool) - set(kept_after))})
    cA, cB = st.columns(2)
    cA.download_button("Download KEPT CSV", kept_df.to_csv(index=False), "kept_combos.csv", mime="text/csv")
    cB.download_button("Download REMOVED CSV", removed_df.to_csv(index=False), "removed_combos.csv", mime="text/csv")
