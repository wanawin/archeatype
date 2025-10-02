
# 1_Large_Filters_Planner.py
# Fully merged single-file Streamlit app
# - Mode selector (Playlist Reducer vs Safe Filter Explorer) to prefill thresholds
# - Combo pool: paste (comma/space/newline) OR upload CSV
# - Filters CSV: upload OR default to lottery_filters_batch_10.csv
# - Winner history CSV: upload OR default to DC5_Midday_Full_Cleaned_Expanded.csv
# - Manual Hot/Cold/Due overrides wired into the filter evaluation environment
# - Includes the core helper definitions so filters referencing these symbols evaluate cleanly
#
# NOTE: This file is self-contained; drop it into your repo as pages/1_Large_Filters_Planner.py

from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# -----------------------
# Core helpers & signals
# -----------------------
VTRAC = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def spread_band(spread: int) -> str:
    if spread <= 3: return "0-3"
    if spread <= 5: return "4-5"
    if spread <= 7: return "6-7"
    if spread <= 9: return "8-9"
    return "10+"

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
    return [int(ch) for ch in str(s)]

# -----------------------
# UI: Title & Mode
# -----------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio(
        "Select mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        help=(
            "Playlist Reducer = original logic, larger default cut thresholds, used for building a loser list.\n"
            "Safe Filter Explorer = lower thresholds, explores more filters."
        ),
        index=1
    )
    if mode == "Playlist Reducer":
        default_min_elims = 200
        default_beam = 5
        default_steps = 15
    else:
        default_min_elims = 60
        default_beam = 6
        default_steps = 18

    min_elims = st.number_input(
        "Min eliminations to call it ‘Large’", min_value=1, max_value=99999, value=default_min_elims
    )
    beam_width = st.number_input("Beam width (search breadth)", min_value=1, max_value=50, value=default_beam)
    max_steps = st.number_input("Max steps (search depth)", min_value=1, max_value=50, value=default_steps)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# -----------------------
# Seed context (optional)
# -----------------------
st.subheader("Seed context (optional)")
seed_str = st.text_input("Seed (prev draw)")
seed2_str = st.text_input("Prev Seed (2-back)")
seed3_str = st.text_input("Prev Prev Seed (3-back)")

def parse_seed_digits(s: str) -> List[int]:
    s = s.strip()
    return digits_of(s) if s else []

seed_digits = parse_seed_digits(seed_str)
prev_seed_digits = parse_seed_digits(seed2_str)
prev_prev_seed_digits = parse_seed_digits(seed3_str)

# -----------------------
# Combo Pool
# -----------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma, space, or newline separated):", height=140)
pool_upload = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_column_hint = st.text_input("Pool column name (optional hint)", value="Result")

pool: List[str] = []
if pool_text.strip():
    raw = pool_text.replace("\n", ",").replace(" ", ",")
    pool = [p.strip() for p in raw.split(",") if p.strip()]
elif pool_upload:
    df_pool = pd.read_csv(pool_upload)
    col = pool_column_hint if pool_column_hint in df_pool.columns else None
    if not col:
        # try to find likely column
        for c in df_pool.columns:
            if c.lower() in ("result", "combo", "combos", "number", "numbers"):
                col = c; break
    if not col:
        st.error("CSV must have a 'Result' (or similar) column.")
        st.stop()
    pool = df_pool[col].astype(str).tolist()
else:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

pool_size = len(pool)

# -----------------------
# Winner History & Filters inputs (override or defaults)
# -----------------------
st.subheader("Winner History")
hist_upload = st.file_uploader(
    "Upload winner history CSV (optional — leave blank to use default)",
    type=["csv"], key="hist_upload"
)
# Default can be changed to your repo path/file
default_history_path = "DC5_Midday_Full_Cleaned_Expanded.csv"
if hist_upload:
    history_df = pd.read_csv(hist_upload)
else:
    # Try to read default if present, otherwise keep empty
    try:
        history_df = pd.read_csv(default_history_path)
    except Exception:
        history_df = pd.DataFrame()

st.subheader("Filters")
filters_text = st.text_area("Paste filter CSV content here (optional):", height=140, key="filters_text")
filters_upload = st.file_uploader(
    "Or upload filters CSV (optional — leave blank to use default lottery_filters_batch_10.csv)",
    type=["csv"], key="filters_upload"
)
default_filters_path = "lottery_filters_batch_10.csv"
if filters_text.strip():
    from io import StringIO
    filters_df = pd.read_csv(StringIO(filters_text))
elif filters_upload:
    filters_df = pd.read_csv(filters_upload)
else:
    try:
        filters_df = pd.read_csv(default_filters_path)
    except Exception:
        st.error("No filters provided, and default 'lottery_filters_batch_10.csv' not found.")
        st.stop()

# -----------------------
# Manual Hot/Cold/Due overrides
# -----------------------
st.subheader("Hot / Cold / Due Overrides")
st.caption("If you enter values, they will be used. Leave blank to ignore (filters will still run).")
manual_hot = st.text_input("Manual Hot digits (comma or space separated)", value="")
manual_cold = st.text_input("Manual Cold digits (comma or space separated)", value="")
manual_due = st.text_input("Manual Due digits (comma or space separated)", value="")

def parse_digit_list(s: str) -> List[int]:
    if not s or not s.strip():
        return []
    return [int(x) for x in re.split(r"[\s,]+", s.strip()) if x.strip().isdigit()]

hot_override = parse_digit_list(manual_hot)
cold_override = parse_digit_list(manual_cold)
due_override  = parse_digit_list(manual_due)

# -----------------------
# Build base evaluation context (seed + overrides)
# -----------------------
base_env: Dict[str, object] = {
    'seed_digits': seed_digits,
    'prev_seed_digits': prev_seed_digits,
    'prev_prev_seed_digits': prev_prev_seed_digits,
    'new_seed_digits': list(set(seed_digits) - set(prev_seed_digits)),
    'seed_counts': Counter(seed_digits),
    'seed_sum': sum(seed_digits) if seed_digits else 0,
    'prev_sum_cat': sum_category(sum(seed_digits)) if seed_digits else "",
    'seed_vtracs': set(VTRAC.get(d, d) for d in seed_digits),
    'mirror': MIRROR,
    'VTRAC': VTRAC,
    'MIRROR': MIRROR,
    'sum_category': sum_category,
    'spread_band': spread_band,
    'classify_structure': classify_structure,
    'Counter': Counter,
    'any': any,
    'all': all,
    'len': len,
    'sum': sum,
    'max': max,
    'min': min,
    'set': set,
    'sorted': sorted,
    # Convenience names sometimes referenced
    'seed_value': int("".join(map(str, seed_digits))) if seed_digits else 0,
    'nan': float('nan'),
    # Hot/Cold/Due (override or blanks – do NOT auto-compute here)
    'hot_digits': hot_override,
    'cold_digits': cold_override,
    'due_digits': due_override,
}
# Back-compat aliases occasionally used in older filters
base_env['seed_digits_1'] = prev_seed_digits
base_env['seed_digits_2'] = prev_prev_seed_digits
base_env['prev_seed_sum'] = sum(prev_seed_digits) if prev_seed_digits else 0
base_env['prev_prev_seed_sum'] = sum(prev_prev_seed_digits) if prev_prev_seed_digits else 0
base_env['winner_structure'] = classify_structure(seed_digits or [0])
base_env['combo_structure'] = classify_structure(seed_digits or [0])

# Expose history if any expression wants it
if not history_df.empty:
    base_env['history_df'] = history_df

# -----------------------
# Compile and evaluate filters on the pool
# -----------------------
required_cols = {'name', 'expression'}
if not required_cols.issubset(set(c.lower() for c in filters_df.columns)):
    # Try case-insensitive mapping
    rename_map = {}
    for req in ('name','expression','enabled','applicable_if','parity_wiper'):
        for c in filters_df.columns:
            if c.lower() == req:
                rename_map[c] = req
                break
    if rename_map:
        filters_df = filters_df.rename(columns=rename_map)
# Normalize presence
for c in ('name','expression'):
    if c not in filters_df.columns:
        st.error(f"Filters CSV must include column '{c}'.")
        st.stop()
if 'enabled' not in filters_df.columns:
    filters_df['enabled'] = True
if 'applicable_if' not in filters_df.columns:
    filters_df['applicable_if'] = ""

# Evaluate each filter: count eliminations from the pool
results = []
safe_globals = {'__builtins__': {}}

def eval_expr(expr: str, local_env: Dict[str, object]) -> bool:
    # Returns True if expression is True (i.e., eliminate)
    try:
        return bool(eval(expr, safe_globals, local_env))
    except Exception:
        return False

for _, row in filters_df.iterrows():
    if not bool(row.get('enabled', True)):
        continue
    name = str(row['name'])
    expr = str(row['expression']).strip()
    applicable_if = str(row.get('applicable_if', "")).strip()
    parity_wiper = bool(row.get('parity_wiper', False))

    # First check applicability once (context-only)
    applicable = True
    if applicable_if:
        applicable = eval_expr(applicable_if, dict(base_env))
    if not applicable:
        elim_count = 0
    else:
        elim_count = 0
        # iterate pool
        for combo in pool:
            cd = digits_of(combo)
            local_env = dict(base_env)
            local_env.update({
                'combo': combo,
                'combo_digits': cd,
                'combo_digits_list': cd,
                'combo_sum': sum(cd),
                'combo_sum_cat': sum_category(sum(cd)),
                'combo_sum_category': sum_category(sum(cd)),
                'combo_structure': classify_structure(cd),
                'combo_vtracs': set(VTRAC.get(d, d) for d in cd),
            })
            if eval_expr(expr, local_env):
                elim_count += 1

    results.append({
        'name': name,
        'expression': expr,
        'applicable_if': applicable_if,
        'elim_count_on_pool': elim_count,
        'parity_wiper': parity_wiper,
    })

df = pd.DataFrame(results).sort_values(['elim_count_on_pool','name'], ascending=[False, True]).reset_index(drop=True)

# Apply "Large" threshold & optional parity exclusion
large_df = df[df['elim_count_on_pool'] >= int(min_elims)].copy()
if exclude_parity and 'parity_wiper' in large_df.columns:
    large_df = large_df[~large_df['parity_wiper']].copy()

st.subheader("Candidate Large Filters")
st.write(f"Pool size: **{pool_size}** | Large threshold: **{min_elims}** | Candidates: **{len(large_df)}**")
st.dataframe(large_df, use_container_width=True, height=420)

# -----------------------
# Downloads
# -----------------------
import datetime as _dt
ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

csv_bytes = large_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv_bytes, file_name=f"large_filters_{ts}.csv", mime="text/csv")

names_text = "\n".join(large_df['name'].tolist()).encode('utf-8')
st.download_button("Download filter names (TXT)", data=names_text, file_name=f"large_filter_names_{ts}.txt", mime="text/plain")

st.success("Ready. Adjust sliders or inputs and the table updates live.")
