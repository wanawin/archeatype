from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from collections import Counter

# -------------------------------------------------
# Page & defaults
# -------------------------------------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

# -----------------------
# Core helpers & signals
# -----------------------
VTRAC: Dict[int, int] = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s).strip() if ch.isdigit()]

def parse_list(text: str, to_int: bool=False) -> List[str]:
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

def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    sd  = digits_of(seed) if seed else []
    sd2 = digits_of(prev_seed) if prev_seed else []
    sd3 = digits_of(prev_prev_seed) if prev_prev_seed else []
    base = {
        "seed_digits": sd,
        "seed_digits_1": sd,
        "prev_seed_digits": sd2,
        "seed_digits_2": sd2,
        "prev_prev_seed_digits": sd3,
        "seed_digits_3": sd3,
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd) if sd else 0,
        "prev_seed_sum": sum(sd2) if sd2 else 0,
        "prev_prev_seed_sum": sum(sd3) if sd3 else 0,
        "prev_sum_cat": sum_category(sum(sd) if sd else 0),
        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),
        "mirror": MIRROR,
        "VTRAC": VTRAC,
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),
        "Counter": Counter,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted,
        "seed_value": int(seed) if seed else None,
        "nan": float("nan"),
        "winner_structure": classify_structure(sd) if sd else "",
    }
    return base

def combo_env(base_env: Dict, combo: str) -> Dict:
    cd = digits_of(combo)
    env = dict(base_env)
    env.update({
        "combo": combo,
        "combo_digits": sorted(cd),
        "combo_digits_list": sorted(cd),
        "combo_sum": sum(cd),
        "combo_sum_cat": sum_category(sum(cd)),
        "combo_sum_category": sum_category(sum(cd)),
        "combo_vtracs": set(VTRAC[d] for d in cd),
        "combo_structure": classify_structure(cd),
        "last_digit": cd[-1] if cd else None,
        "spread": (max(cd) - min(cd)) if cd else 0,
        "seed_spread": (max(base_env["seed_digits"]) - min(base_env["seed_digits"]))
                        if base_env["seed_digits"] else 0,
    })
    return env

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
        raise ValueError("Pool CSV must contain a 'Result' column.")
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_winners_csv(path: str) -> List[str]:
    if not path:
        return []
    try:
        df = pd.read_csv(path)
    except:
        return []
    cols_lower = {c.lower(): c for c in df.columns}
    if "result" in cols_lower:
        s = df[cols_lower["result"]]
    elif "combo" in cols_lower:
        s = df[cols_lower["combo"]]
    else:
        return []
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_filters_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    if "id" not in df.columns and "fid" in df.columns:
        df["id"] = df["fid"]
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)
    if "parity_wiper" not in df.columns:
        df["parity_wiper"] = False
    if "enabled" not in df.columns:
        df["enabled"] = True
    return df

def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict):
    expr = str(row["expression"])
    try: code = compile(expr, "<filter_expr>", "eval")
    except: return set(), 0
    eliminated = set()
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except:
            pass
    return eliminated, len(eliminated)

def greedy_plan(candidates: pd.DataFrame, pool: List[str], base_env: Dict,
                beam_width: int, max_steps: int):
    remaining = set(pool)
    chosen: List[Dict] = []
    for step in range(max_steps):
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
            "id": best_row["id"],
            "name": best_row.get("name", ""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining),
        })
        if best_cnt == 0: break
    return chosen, sorted(list(remaining))

# -------------------------------------------------
# Sidebar Mode Toggle
# -------------------------------------------------
st.sidebar.header("Planner Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    index=1,
    help="Playlist Reducer = original fast, bigger eliminations.\nSafe Filter Explorer = lower threshold, more candidates."
)
if mode == "Playlist Reducer":
    default_min_elims = 120; default_beam = 5; default_steps = 15
else:
    default_min_elims = 60; default_beam = 6; default_steps = 18

min_elims = st.sidebar.number_input("Min eliminations to call it ‘Large’", 1, 99999, value=default_min_elims)
beam_width = st.sidebar.number_input("Beam width (search breadth)", 1, 50, value=default_beam)
max_steps = st.sidebar.number_input("Max steps (search depth)", 1, 50, value=default_steps)
exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=True)

# -------------------------------------------------
# Seed + Hot/Cold/Due
# -------------------------------------------------
st.subheader("Seed context")
c1, c2, c3 = st.columns(3)
seed = c1.text_input("Seed (prev draw)", "")
prev_seed = c2.text_input("Prev Seed (2-back)", "")
prev_prev = c3.text_input("Prev Prev Seed (3-back)", "")

st.subheader("Hot/Cold/Due digits")
c4, c5, c6 = st.columns(3)
hot_digits  = parse_list(c4.text_input("Hot digits", ""), to_int=True)
cold_digits = parse_list(c5.text_input("Cold digits", ""), to_int=True)
due_digits  = parse_list(c6.text_input("Due digits", ""), to_int=True)

# -------------------------------------------------
# Combo Pool
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos", height=130)
pool_file = st.file_uploader("Or upload combo pool CSV", type=["csv"])
pool_col_hint = st.text_input("Pool column name", value="Result")

pool = []
if pool_text.strip():
    pool = parse_list(pool_text)
elif pool_file: pool = load_pool_from_csv(pool_file, pool_col_hint)
else: st.stop()
pool = [p for p in pool if p]
st.write(f"Pool size: {len(pool)}")

# -------------------------------------------------
# Winners CSV
# -------------------------------------------------
st.subheader("Winners History")
history_path = st.text_input("Path to winners history CSV", value="DC5_Midday_Full_Cleaned_Expanded.csv")
winners_list = load_winners_csv(history_path)

# -------------------------------------------------
# Filters
# -------------------------------------------------
st.subheader("Filters")
ids_text = st.text_area("Paste applicable Filter IDs (optional)", height=90)
filters_file_up = st.file_uploader("Or upload Filters CSV", type=["csv"])
filters_csv_path_default = "lottery_filters_batch_10.csv"
filters_source = filters_file_up if filters_file_up else filters_csv_path_default
filters_df_full = load_filters_csv(filters_source)
applicable_ids = set(parse_list(ids_text))
if applicable_ids:
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(applicable_ids)]
else: filters_df = filters_df_full
if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]]
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True]
st.write(f"Filters loaded: {len(filters_df)}")

# -------------------------------------------------
# Build environment & evaluate
# -------------------------------------------------
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

scored_rows = []
for _, r in filters_df.iterrows():
    elim_set, cnt = eval_filter_on_pool(r, pool, base_env)
    scored_rows.append({
        "id": r["id"],
        "name": r.get("name", ""),
        "expression": r["expression"],
        "elim_count_on_pool": cnt,
        "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
        "_elim_set": elim_set,
    })
scored_df = pd.DataFrame(scored_rows)

large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
st.write(f"Large filters (≥ {min_elims}): {len(large_df)}")
st.dataframe(large_df[["id", "name", "elim_count_on_pool", "elim_pct_on_pool"]])

# -------------------------------------------------
# Greedy plan
# -------------------------------------------------
if not large_df.empty:
    plan_df = large_df[["id", "name", "expression"]].copy()
    plan, kept_after = greedy_plan(plan_df, pool, base_env, int(beam_width), int(max_steps))
else:
    plan, kept_after = [], pool
st.write(f"Kept combos after plan: {len(kept_after)} / {len(pool)}")
if plan: st.dataframe(pd.DataFrame(plan))

# -------------------------------------------------
# Downloads
# -------------------------------------------------
kept_df = pd.DataFrame({"Result": kept_after})
removed_df = pd.DataFrame({"Result": sorted(set(pool)-set(kept_after))})
cA, cB = st.columns(2)
cA.download_button("Download KEPT combos (CSV)", kept_df.to_csv(index=False), "kept_combos.csv")
cA.download_button("Download KEPT combos (TXT)", "\n".join(kept_after), "kept_combos.txt")
cB.download_button("Download REMOVED combos (CSV)", removed_df.to_csv(index=False), "removed_combos.csv")
cB.download_button("Download REMOVED combos (TXT)", "\n".join(sorted(set(pool)-set(kept_after))), "removed_combos.txt")
