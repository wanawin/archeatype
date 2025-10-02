
# pages/1_Large_Filters_Planner.py
from __future__ import annotations

import io
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from collections import Counter

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

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s).strip() if ch.isdigit()]

def parse_digit_list(s: str) -> List[int]:
    """Parse user entry like '9,8,3' or '9 8 3' or '983' into unique digit list [9,8,3]."""
    s = (s or '').strip()
    if not s:
        return []
    # Split on commas or whitespace; also support bare string like '983'
    if (',' in s) or (' ' in s):
        parts = re.split(r'[\s,]+', s.strip())
        digs = [int(p) for p in parts if p != '']
    else:
        digs = [int(ch) for ch in s if ch.isdigit()]
    # keep order but unique
    seen = set(); out = []
    for d in digs:
        if d not in seen and 0 <= d <= 9:
            seen.add(d); out.append(d)
    return out

def parse_pool_text(text: str) -> List[str]:
    """Accept combos separated by comma, space, or newline. Return unique strings preserving order."""
    raw = (text or '').replace('\n', ',').replace('\r', ',')
    raw = re.sub(r'\s+', ',', raw.strip())
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    # normalize to 5 digits with leading zeros if necessary
    out = []
    seen = set()
    for p in parts:
        if not p.isdigit():
            continue
        norm = p.zfill(5)
        if norm not in seen:
            seen.add(norm); out.append(norm)
    return out

def build_eval_env(seed: str, prev_seed: str, prev_prev_seed: str,
                   hot: List[int], cold: List[int], due: List[int]) -> Dict:
    sd = digits_of(seed) if seed else []
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev_seed) if prev_prev_seed else []

    new_digits = set(sd) - set(pdigs)
    seed_counts = Counter(sd)
    seed_vtracs = set(VTRAC[d] for d in sd) if sd else set()
    prev_sum_cat = sum_category(sum(sd)) if sd else ""
    prev_pattern = []
    for digs in (ppdigs, pdigs, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    env = {
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': new_digits,
        'prev_pattern': tuple(prev_pattern),
        'hot_digits': sorted(set(hot)),
        'cold_digits': sorted(set(cold)),
        'due_digits':  sorted(set(due)),
        'seed_counts': seed_counts,
        'seed_sum': sum(sd) if sd else 0,
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted,
        # DC-5 compat:
        'seed_value': int(seed) if (seed and seed.isdigit()) else 0,
        'nan': float('nan'),
        'winner_structure': classify_structure(sd) if sd else "",
        'combo_structure': "",
        'combo_sum_category': "",
    }
    return env

def evaluate_expression(expr: str, env: Dict) -> bool:
    """Safe-ish eval: only allow name lookups from env and Python builtins we injected."""
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def load_filters_from_csv(file_or_buf) -> pd.DataFrame:
    df = pd.read_csv(file_or_buf)
    # normalize columns:
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id') or cols.get('fid') or 'id'
    if id_col not in df.columns:
        df[id_col] = range(1, len(df)+1)
    if 'name' not in df.columns:
        df['name'] = df[id_col].astype(str)
    if 'expression' not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    # optional columns
    if 'enabled' not in df.columns:
        df['enabled'] = True
    if 'applicable_if' not in df.columns:
        df['applicable_if'] = ""
    # reorder a bit
    return df[[id_col, 'name', 'enabled', 'applicable_if', 'expression']].rename(columns={id_col: 'fid'})

def evaluate_filters_on_pool(filters_df: pd.DataFrame, pool: List[str],
                             base_env: Dict, show_progress: bool=True) -> pd.DataFrame:
    rows = []
    progress = st.progress(0) if show_progress else None
    total = max(len(filters_df), 1)
    for idx, row in filters_df.iterrows():
        if not bool(row.get('enabled', True)):
            continue
        fid = row['fid']
        name = str(row.get('name', fid))
        applicable_if = str(row.get('applicable_if', '') or '').strip()
        expr = str(row.get('expression', '') or '').strip()
        if not expr:
            continue

        # Check applicability
        env = dict(base_env)
        applicable = True
        if applicable_if:
            applicable = evaluate_expression(applicable_if, env)
        if not applicable:
            # do not include
            if progress: progress.progress(min((idx+1)/total, 1.0))
            continue

        elim = 0
        errors = 0
        for combo in pool:
            cd = digits_of(combo)
            env['combo'] = combo
            env['combo_digits'] = sorted(cd)
            env['combo_digits_list'] = sorted(cd)
            env['combo_sum'] = sum(cd)
            env['combo_sum_cat'] = sum_category(sum(cd))
            env['combo_sum_category'] = env['combo_sum_cat']
            env['combo_vtracs'] = set(VTRAC[d] for d in cd)
            env['combo_structure'] = classify_structure(cd)
            try:
                if evaluate_expression(expr, env):
                    elim += 1
            except Exception:
                errors += 1

        kept = len(pool) - elim
        rows.append({
            'fid': fid,
            'name': name,
            'applicable_if': applicable_if,
            'expression': expr,
            'elim_count_on_pool': elim,
            'kept_count_on_pool': kept,
            'errors': errors,
        })
        if progress:
            progress.progress(min((idx+1)/total, 1.0))
    if progress: progress.empty()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['elim_count_on_pool','kept_count_on_pool'], ascending=[False, True])
    return out

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

with st.sidebar:
    st.subheader("Mode")
    mode = st.radio(
        "Select mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1,
        help="Playlist Reducer = original idea with strict threshold.\nSafe Filter Explorer = lower threshold to surface more suggestions."
    )
    if mode == "Playlist Reducer":
        default_min_elims = 200
        default_beam = 5
        default_steps = 15
    else:
        default_min_elims = 60
        default_beam = 6
        default_steps = 18

    min_elims = st.number_input("Min eliminations to call it ‘Large’", min_value=1, max_value=99999, value=default_min_elims)
    beam_width = st.number_input("Beam width (search breadth)", min_value=1, max_value=50, value=default_beam)
    max_steps = st.number_input("Max steps (search depth)", min_value=1, max_value=50, value=default_steps)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

st.header("Seed context (optional)")
c1, c2, c3 = st.columns(3)
seed = c1.text_input("Seed (prev draw)")
prev2 = c2.text_input("Prev Seed (2-back)")
prev3 = c3.text_input("Prev Prev Seed (3-back)")

st.header("Combo Pool")
pool_text = st.text_area("Paste combos (comma, space, or newline separated):", height=140)
c4, c5 = st.columns([2,1])
pool_file = c4.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_col_hint = c5.text_input("Pool column name (optional hint)", value="Result")

# Winner history (optional, for hot/cold/due if not manually provided)
st.header("Winner history (optional)")
hist_file = st.file_uploader("Upload winner history CSV (optional)", type=["csv"])
hist_default_path = st.text_input("Or use repo path (optional; defaults to DC5_Midday_Full_Cleaned_Expanded.csv if left blank)",
                                  value="")

st.header("Manual hot / cold / due (optional)")
hc1, hc2, hc3 = st.columns(3)
hot_in = hc1.text_input("Hot digits (e.g., 9,8,3)", value="")
cold_in = hc2.text_input("Cold digits (e.g., 6,7,0,5)", value="")
due_in = hc3.text_input("Due digits (e.g., 1,2,4)", value="")

# Filters
st.header("Filters")
filters_text = st.text_area("Paste filter CSV content here (optional):", height=160, help="Same columns your batch CSV uses. Must contain an 'expression' column.")
filters_file = st.file_uploader("Or upload filters CSV (defaults to lottery_filters_batch_10.csv if omitted)", type=["csv"])

# ---- Build pool ----
pool: List[str] = []
if pool_text.strip():
    pool = parse_pool_text(pool_text)
elif pool_file is not None:
    try:
        df_pool = pd.read_csv(pool_file)
        col = pool_col_hint if pool_col_hint in df_pool.columns else None
        if col is None:
            for try_col in ['Result','result','combo','Combo']:
                if try_col in df_pool.columns: col = try_col; break
        if col is None:
            st.error("Could not find a pool column. Provide the column name in the hint box.")
        else:
            pool = [str(x).zfill(5) for x in df_pool[col].astype(str).tolist() if str(x).strip()]
    except Exception as e:
        st.error(f"Failed to read pool CSV: {e}")

if not pool:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

# ---- Winner history & H/C/D ----
hot = parse_digit_list(hot_in)
cold = parse_digit_list(cold_in)
due = parse_digit_list(due_in)

if not (hot or cold or due):
    # try to compute from history if provided
    hist_df = None
    if hist_file is not None:
        try:
            hist_df = pd.read_csv(hist_file)
        except Exception as e:
            st.warning(f"Could not read uploaded history CSV: {e}")
    elif hist_default_path.strip():
        try:
            hist_df = pd.read_csv(hist_default_path.strip())
        except Exception as e:
            st.warning(f"Could not read history at path '{hist_default_path}': {e}")
    else:
        # try default file in repo if present
        try:
            hist_df = pd.read_csv("DC5_Midday_Full_Cleaned_Expanded.csv")
        except Exception:
            hist_df = None

    if hist_df is not None:
        # assume a 'Result' column with winners
        col = None
        for try_col in ['Result','result','winner','Winner']:
            if try_col in hist_df.columns: col = try_col; break
        if col is not None:
            winners_digits = [[int(ch) for ch in str(x).zfill(5)] for x in hist_df[col].astype(str).tolist() if str(x).strip().isdigit()]
            # compute rough hot/cold/due over last 10
            hist = winners_digits[-10:] if len(winners_digits)>=10 else winners_digits
            flat = [d for row in hist for d in row]
            cnt = Counter(flat)
            most = cnt.most_common()
            if most:
                topk = min(6, len(most))
                thresh = most[topk-1][1]
                hot = [d for d,c in most if c >= thresh]
                least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
                coldk = min(4, len(least))
                cth = least[coldk-1][1]
                cold = [d for d,c in least if c <= cth]
                last2 = set(d for row in winners_digits[-2:] for d in row) if winners_digits else set()
                due = [d for d in range(10) if d not in last2]
        # else leave empty lists

# ---- Base evaluation environment ----
base_env = build_eval_env(seed, prev2, prev3, hot, cold, due)

# ---- Load filters ----
filters_df = None
if filters_text.strip():
    try:
        filters_df = load_filters_from_csv(io.StringIO(filters_text))
    except Exception as e:
        st.error(f"Failed to parse pasted filters CSV: {e}")
elif filters_file is not None:
    try:
        filters_df = load_filters_from_csv(filters_file)
    except Exception as e:
        st.error(f"Failed to read uploaded filters CSV: {e}")
else:
    # try default CSV path
    default_path = "lottery_filters_batch_10.csv"
    try:
        filters_df = load_filters_from_csv(default_path)
        st.caption(f"Using default filters file: {default_path}")
    except Exception as e:
        st.error("No filters provided and default 'lottery_filters_batch_10.csv' not found or invalid.")
        st.stop()

st.subheader("Evaluating filters on current pool…")
results_df = evaluate_filters_on_pool(filters_df, pool, base_env, show_progress=True)

if results_df.empty:
    st.warning("No filters produced eliminations on the current pool.")
    st.stop()

# mark parity-wipers heuristically if possible (optional)
if 'parity_wiper' not in results_df.columns:
    results_df['parity_wiper'] = False  # placeholder unless your CSV provides it

# apply large threshold + parity exclusion
large_df = results_df[results_df['elim_count_on_pool'] >= int(min_elims)].copy()
if exclude_parity and 'parity_wiper' in large_df.columns:
    large_df = large_df[~large_df['parity_wiper']]

st.subheader("Candidate Large Filters")
st.write(f"{len(large_df)} filters qualify as 'Large' with current settings.")
st.dataframe(large_df, use_container_width=True)

# Downloads
st.download_button(
    "Download candidates (CSV)",
    data=large_df.to_csv(index=False).encode('utf-8'),
    file_name="large_filters_candidates.csv",
    mime="text/csv",
)

# TXT: one filter per line "fid: name"
txt_lines = [f"{r.fid}: {r.name} — elim={r.elim_count_on_pool}, kept={r.kept_count_on_pool}"
             for r in large_df.itertuples(index=False)]
st.download_button(
    "Download candidates (TXT)",
    data="\n".join(txt_lines).encode('utf-8'),
    file_name="large_filters_candidates.txt",
    mime="text/plain",
)
