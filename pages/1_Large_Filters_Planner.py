# pages/1_Large_Filters_Planner.py
from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from collections import Counter

# -----------------------
# Page & defaults
# -----------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

DEFAULT_HISTORY_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
DEFAULT_FILTERS_CSV = "lottery_filters_batch_10.csv"

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
    if spread <= 3: return "0–3"
    if spread <= 5: return "4–5"
    if spread <= 7: return "6–7"
    if spread <= 9: return "8–9"
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

def digits_of(x: str|int) -> List[int]:
    return [int(ch) for ch in str(x)]

def parse_int_list(text: str) -> List[int]:
    """Parse digits like '9,8,3' or '9 8 3' to [9,8,3]. Empty -> []."""
    if not text or not text.strip():
        return []
    raw = re.split(r"[,\s]+", text.strip())
    out = []
    for t in raw:
        if t == "": continue
        if not t.isdigit(): continue
        d = int(t)
        if 0 <= d <= 9:
            out.append(d)
    return out

def parse_combos(text: str) -> List[str]:
    """Accept comma/space/newline separated combos; returns list of strings."""
    if not text or not text.strip():
        return []
    raw = re.split(r"[,\s]+", text.strip())
    return [r for r in raw if r]

# -----------------------
# Build evaluation context
# -----------------------
def env_for_combo(
    combo: str,
    seed: str,
    prev_seed: str,
    prev_prev_seed: str,
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int],
) -> Dict:
    sd = digits_of(seed) if seed else []
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev_seed) if prev_prev_seed else []

    cd = sorted(digits_of(combo))
    new_digits = set(sd) - set(pdigs)

    prev_pattern = []
    for digs in (ppdigs, pdigs, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    ctx = {
        # seeds
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': new_digits,
        'seed_counts': Counter(sd),
        'seed_sum': sum(sd) if sd else 0,
        'prev_sum_cat': sum_category(sum(sd)) if sd else "",
        'seed_vtracs': set(VTRAC[d] for d in sd) if sd else set(),

        # combo
        'combo': combo,
        'combo_digits': cd,
        'combo_digits_list': cd,
        'combo_sum': sum(cd),
        'combo_sum_cat': sum_category(sum(cd)),
        'combo_sum_category': sum_category(sum(cd)),
        'combo_structure': classify_structure(cd) if cd else "",
        'spread_band': spread_band(max(cd)-min(cd) if cd else 0),

        # hot/cold/due
        'hot_digits': sorted(hot_digits) if hot_digits else [],
        'cold_digits': sorted(cold_digits) if cold_digits else [],
        'due_digits': sorted(due_digits) if due_digits else [],

        # signals
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted,

        # aliases often used in old filters
        'seed_value': int(seed) if seed else 0,
        'winner_structure': classify_structure(sd) if sd else "",
        'nan': float('nan'),
        'prev_pattern': tuple(prev_pattern),
    }
    return ctx

def apply_filter_expression_to_pool(
    expression: str,
    pool: List[str],
    seed: str,
    prev_seed: str,
    prev_prev_seed: str,
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int]
) -> Tuple[List[str], List[str]]:
    """
    Return (eliminated_list, kept_list) for given filter expression.
    Expression should evaluate to True to ELIMINATE the combo.
    """
    eliminated = []
    kept = []
    for c in pool:
        env = env_for_combo(c, seed, prev_seed, prev_prev_seed, hot_digits, cold_digits, due_digits)
        try:
            # Evaluate in a sandboxed globals (no builtins)
            res = eval(expression, {"__builtins__": {}}, env)
        except Exception:
            # If expression breaks on this combo, treat as NOT eliminating to be safe
            res = False
        if bool(res):
            eliminated.append(c)
        else:
            kept.append(c)
    return eliminated, kept

# -----------------------
# UI — Mode selector
# -----------------------
mode = st.sidebar.radio(
    "Select mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    help="Playlist Reducer = original intent, but you use its 'kept' list as losers.\n"
         "Safe Filter Explorer = lower threshold and deeper breadth to surface more candidates."
)

if mode == "Playlist Reducer":
    default_min_elims = 120
    default_beam = 5
    default_steps = 15
else:
    default_min_elims = 60
    default_beam = 6
    default_steps = 18

min_elims = st.sidebar.number_input("Min eliminations to call it ‘Large’", 1, 99999, value=default_min_elims)
beam_width = st.sidebar.number_input("Beam width (search breadth)", 1, 50, value=default_beam)
max_steps = st.sidebar.number_input("Max steps (search depth)", 1, 50, value=default_steps)
exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=True)

# -----------------------
# Seed context (optional)
# -----------------------
with st.expander("Seed context (optional)", expanded=False):
    col_s1, col_s2, col_s3 = st.columns(3)
    seed = col_s1.text_input("Seed (prev draw)")
    prev_seed = col_s2.text_input("Prev Seed (2-back)")
    prev_prev_seed = col_s3.text_input("Prev Prev Seed (3-back)")

# -----------------------
# Combo Pool
# -----------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma, space, or newline separated):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name (optional hint)", value="Result")

pool: List[str] = []
if pool_text.strip():
    pool = parse_combos(pool_text)
elif pool_file:
    try:
        df_pool = pd.read_csv(pool_file)
        col = pool_col_hint if pool_col_hint in df_pool.columns else "Result"
        if col not in df_pool.columns:
            st.error("Pool CSV must include a 'Result' column (or set the correct column name).")
            st.stop()
        pool = df_pool[col].astype(str).tolist()
    except Exception as e:
        st.error(f"Failed to read pool CSV: {e}")
        st.stop()
else:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

st.caption(f"Pool size: **{len(pool)}**")

# -----------------------
# Winners History (path)
# -----------------------
st.subheader("Winners History (optional)")
history_path = st.text_input(
    "Path to winners history CSV (leave as default if you like)",
    value=DEFAULT_HISTORY_CSV,
)
# We keep this value available; not used directly below unless you wire history-derived features.

# -----------------------
# Hot / Cold / Due (optional)
# -----------------------
with st.expander("Hot / Cold / Due (optional)", expanded=False):
    hot_text = st.text_input("Hot digits (comma/space sep)")
    cold_text = st.text_input("Cold digits (comma/space sep)")
    due_text = st.text_input("Due digits (comma/space sep)")
hot_digits = parse_int_list(hot_text)
cold_digits = parse_int_list(cold_text)
due_digits = parse_int_list(due_text)

# -----------------------
# Filters input (IDs OR CSV)
# -----------------------
st.subheader("Filters")

filter_ids_text = st.text_area(
    "Paste applicable Filter IDs here (optional, comma/space/newline separated). If provided, they will be looked up in the selected filters CSV.",
    height=120,
    help="Example: N0643F056, NO202F053, 81F251 ..."
)

filters_csv_upload = st.file_uploader(
    "Or upload Filters CSV (defaults to lottery_filters_batch_10.csv if omitted)",
    type=["csv"]
)

use_default_filters_csv = not filters_csv_upload
filters_csv_path_ui = DEFAULT_FILTERS_CSV

# Allow full CSV paste too (with an expression column)
with st.expander("Paste full Filters CSV content (optional)", expanded=False):
    filters_csv_text = st.text_area(
        "If you paste here a full CSV (with 'expression' column), it will be used instead of file.",
        height=140
    )

# Load filters DF
filters_df: Optional[pd.DataFrame] = None
loaded_from = ""

try:
    if filters_csv_text.strip():
        filters_df = pd.read_csv(io.StringIO(filters_csv_text))
        loaded_from = "pasted CSV"
    elif filters_csv_upload is not None:
        filters_df = pd.read_csv(filters_csv_upload)
        loaded_from = "uploaded CSV"
    else:
        # default
        if not Path(filters_csv_path_ui).exists():
            st.warning(f"Default filters CSV not found at '{filters_csv_path_ui}'. Upload a filters CSV or paste full CSV.")
            filters_df = None
        else:
            filters_df = pd.read_csv(filters_csv_path_ui)
            loaded_from = f"default CSV ({filters_csv_path_ui})"
except Exception as e:
    st.error(f"Failed to read filters CSV: {e}")
    st.stop()

if filters_df is None or filters_df.empty:
    st.info("No filters loaded yet.")
    st.stop()

# If IDs were provided, subset by id/fid
ids_list: List[str] = []
if filter_ids_text.strip():
    ids_list = [s.strip() for s in re.split(r"[,\s]+", filter_ids_text.strip()) if s.strip()]
    cols_for_match = [c for c in ["id", "fid", "ID", "Id", "filter_id"] if c in filters_df.columns]
    if not cols_for_match:
        st.error("Filters CSV must contain an 'id' or 'fid' (or similar) column to match pasted IDs.")
        st.stop()
    mask = False
    for c in cols_for_match:
        mask = (filters_df[c].astype(str).isin(ids_list)) | mask
    subset_df = filters_df[mask].copy()
    if subset_df.empty:
        st.error("None of the pasted Filter IDs were found in the selected filters CSV.")
        st.stop()
    filters_df = subset_df
    loaded_from = f"{loaded_from} (filtered by pasted IDs)"

# Expression column check
expr_col = None
for cand in ["expression", "expr", "rule", "filter"]:
    if cand in filters_df.columns:
        expr_col = cand
        break
if expr_col is None:
    st.error("Filters CSV must include an 'expression' column (or a column named 'expr'/'rule'/'filter').")
    st.stop()

st.caption(f"Filters loaded from: **{loaded_from}**")
st.caption(f"Filters loaded: **{len(filters_df)}**")

# -----------------------
# Evaluate filters
# -----------------------
st.subheader("Evaluating filters on current pool…")
if len(filters_df) == 0:
    st.info("No filters to evaluate.")
    st.stop()

# Make sure we have some ID/name columns to display
id_col = None
for cand in ["id", "fid", "ID", "Id", "filter_id"]:
    if cand in filters_df.columns:
        id_col = cand
        break

name_col = None
for cand in ["name", "title", "label", "desc", "description"]:
    if cand in filters_df.columns:
        name_col = cand
        break

rows = []
pool_size = len(pool)

for _, row in filters_df.iterrows():
    expression = str(row[expr_col])
    try:
        eliminated, kept = apply_filter_expression_to_pool(
            expression,
            pool,
            seed=seed, prev_seed=prev_seed, prev_prev_seed=prev_prev_seed,
            hot_digits=hot_digits, cold_digits=cold_digits, due_digits=due_digits
        )
        elim_ct = len(eliminated)
        keep_ct = pool_size - elim_ct
        safety_pct = (elim_ct / pool_size * 100.0) if pool_size > 0 else 0.0
    except Exception as e:
        elim_ct = 0
        keep_ct = pool_size
        safety_pct = 0.0

    rows.append({
        "id": str(row[id_col]) if id_col else "",
        "name": str(row[name_col]) if name_col else "",
        "expression": expression,
        "elim_count_on_pool": elim_ct,
        "kept_count_on_pool": keep_ct,
        "safety_pct": round(safety_pct, 2)
    })

results_df = pd.DataFrame(rows)

# Optional parity-wiper exclusion if CSV already has it
if exclude_parity and "parity_wiper" in filters_df.columns:
    # re-attach that info if available
    pw_map = filters_df[[id_col, "parity_wiper"]] if id_col and "parity_wiper" in filters_df.columns else None
    if pw_map is not None:
        pw_map = pw_map.rename(columns={id_col: "id"})
        results_df = results_df.merge(pw_map, on="id", how="left")
        results_df = results_df[~results_df["parity_wiper"].fillna(False)]

# Large (by eliminations) selection
large_df = results_df[results_df["elim_count_on_pool"] >= int(min_elims)].copy()
large_df = large_df.sort_values(["elim_count_on_pool", "safety_pct"], ascending=[False, False])

st.markdown(f"**{len(large_df)}** filters qualify as 'Large' (≥ {min_elims} eliminated).")
st.dataframe(large_df, use_container_width=True)

# -----------------------
# Downloads
# -----------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "Download qualifying filters (CSV)",
        data=df_to_csv_bytes(large_df),
        file_name="qualifying_filters.csv",
        mime="text/csv",
        disabled=large_df.empty
    )
with col_d2:
    # Aggregate unique survivors after chaining all large filters is non-trivial without your planner.
    # Here we simply export the IDs/expressions so you can apply them downstream.
    txt = "\n".join(f"{r['id']} :: {r['expression']}" for _, r in large_df.iterrows())
    st.download_button(
        "Download qualifying filters (TXT)",
        data=txt.encode("utf-8"),
        file_name="qualifying_filters.txt",
        mime="text/plain",
        disabled=large_df.empty
    )

st.caption("Note: This page evaluates each filter against your pasted/uploaded pool and ranks by eliminations (and %). "
           "Your downstream workflow can now use the recommended set to reduce your playlist. "
           "Mode presets only change defaults; all inputs remain intact when you switch.")
