
import streamlit as st
import pandas as pd
from io import StringIO
from collections import Counter
from pathlib import Path
import re
from datetime import datetime

st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

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

def classify_structure(digs):
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def digits_of(s: str):
    return [int(ch) for ch in str(s) if str(s).strip() != ""]

def hot_cold_due(winners_digits, k=10):
    if not winners_digits:
        return set(), set(), set(range(10))
    hist = winners_digits[-k:] if len(winners_digits) >= k else winners_digits
    flat = [d for row in hist for d in row]
    cnt = Counter(flat)
    if not cnt:
        return set(), set(), set(range(10))
    most = cnt.most_common()
    topk = 6
    thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
    hot = {d for d,c in most if c >= thresh}

    least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
    coldk = 4
    cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
    cold = {d for d,c in least if c <= cth}

    last2 = set(d for row in winners_digits[-2:] for d in row)
    due  = set(range(10)) - last2
    return hot, cold, due

def build_day_env(winners_list, i):
    seed = winners_list[i-1]
    winner = winners_list[i]
    sd = digits_of(seed)
    cd = sorted(digits_of(winner))
    hist_digits = [digits_of(x) for x in winners_list[:i]]
    hot, cold, due = hot_cold_due(hist_digits, k=10)

    prev_seed = winners_list[i-2] if i-2 >= 0 else ""
    prev_prev = winners_list[i-3] if i-3 >= 0 else ""
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev) if prev_prev else []

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
        'new_seed_digits': set(sd) - set(pdigs),
        'seed_counts': Counter(sd),
        'seed_sum': sum(sd),
        'prev_sum_cat': sum_category(sum(sd)),
        'prev_pattern': tuple(prev_pattern),

        'combo': winner,
        'combo_digits': cd,
        'combo_digits_list': cd,
        'combo_sum': sum(cd),
        'combo_sum_cat': sum_category(sum(cd)),
        'combo_sum_category': sum_category(sum(cd)),

        'seed_vtracs': set(VTRAC[d] for d in sd),
        'combo_vtracs': set(VTRAC[d] for d in cd),
        'mirror': MIRROR,

        'hot_digits': sorted(hot),
        'cold_digits': sorted(cold),
        'due_digits': sorted(due),

        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted, 'Counter': Counter
    }
    env['seed_value'] = int(seed) if str(seed).strip() != "" else None
    env['nan'] = float('nan')
    env['winner_structure'] = classify_structure(sd)
    env['combo_structure'] = classify_structure(cd)
    env['combo_sum_category'] = env['combo_sum_cat']
    return env

def build_ctx_for_pool(seed, prev_seed, prev_prev):
    sd = digits_of(seed) if str(seed).strip() != "" else []
    pdigs = digits_of(prev_seed) if str(prev_seed).strip() != "" else []
    ppdigs = digits_of(prev_prev) if str(prev_prev).strip() != "" else []

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

    base = {
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': new_digits,
        'prev_pattern': tuple(prev_pattern),
        'hot_digits': [],
        'cold_digits': [],
        'due_digits': [d for d in range(10) if d not in pdigs and d not in ppdigs],
        'seed_counts': seed_counts,
        'seed_sum': sum(sd) if sd else 0,
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted
    }
    base['seed_value'] = int("".join(str(d) for d in sd)) if sd else None
    base['prev_seed_sum'] = sum(pdigs) if pdigs else None
    base['prev_prev_seed_sum'] = sum(ppdigs) if ppdigs else None
    base['seed_digits_1'] = pdigs
    base['seed_digits_2'] = ppdigs
    base['nan'] = float('nan')
    base['winner_structure'] = classify_structure(sd) if sd else ""
    base['combo_structure'] = classify_structure(sd) if sd else ""
    return base

SAFE_GLOBALS = {
    "__builtins__": {},
    "any": any, "all": all, "len": len, "sum": sum, "max": max, "min": min, "set": set, "sorted": sorted,
    "Counter": Counter,
    "VTRAC": VTRAC, "MIRROR": MIRROR,
    "sum_category": sum_category, "spread_band": spread_band, "classify_structure": classify_structure
}

def safe_eval(expr: str, env: dict) -> bool:
    if expr is None or str(expr).strip() == "":
        return False
    try:
        return bool(eval(expr, SAFE_GLOBALS, env))
    except Exception:
        return False

# ---- Mode selection ----
mode = st.sidebar.radio(
    "Mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    help="Playlist Reducer = original loser-list workflow (strict big-cuts).  Safe Filter Explorer = lower elimination threshold to surface more candidates."
)
if mode == "Playlist Reducer":
    default_min_elims = 200
    default_beam = 5
    default_steps = 15
else:
    default_min_elims = 60
    default_beam = 6
    default_steps = 18

min_elims = st.sidebar.number_input("Min eliminations to call it ‘Large’", min_value=1, max_value=99999, value=default_min_elims, step=1)
beam_width = st.sidebar.number_input("Beam width (search breadth)", min_value=1, max_value=50, value=default_beam, step=1)
max_steps = st.sidebar.number_input("Max steps (search depth)", min_value=1, max_value=50, value=default_steps, step=1)
exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=True)

st.markdown("---")

# ---- Seed context (optional) ----
st.subheader("Seed context (optional)")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    seed = st.text_input("Seed (prev draw)", value="")
with col_s2:
    prev_seed = st.text_input("Prev Seed (2-back)", value="")
with col_s3:
    prev_prev = st.text_input("Prev Prev Seed (3-back)", value="")

# ---- Combo Pool Input ----
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma, space, or newline separated):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name (optional hint)", value="Result")

pool = []
if pool_text.strip():
    raw = re.split(r'[\s,]+', pool_text.strip())
    pool = [x.strip() for x in raw if x.strip()]
elif pool_file is not None:
    df_pool = pd.read_csv(pool_file)
    pool_col = None
    cand_cols = [pool_col_hint] if pool_col_hint else []
    cand_cols += ["Result", "result", "combo", "Combo"]
    for c in cand_cols:
        if c and c in df_pool.columns:
            pool_col = c
            break
    if not pool_col:
        st.error("Pool CSV must contain a 'Result' (or result/combo) column, or provide the column name in the hint.")
        st.stop()
    pool = df_pool[pool_col].astype(str).tolist()

if not pool:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

pool_size = len(pool)
st.write(f"Pool size: **{pool_size}**")

# ---- Filters Input ----
st.subheader("Filters")
filters_text = st.text_area("Paste Filters CSV content (optional):", height=160)
filters_file = st.file_uploader("Or upload Filters CSV", type=["csv"])

if filters_text.strip():
    filters_df = pd.read_csv(StringIO(filters_text))
elif filters_file is not None:
    filters_df = pd.read_csv(filters_file)
else:
    st.info("Paste Filters CSV content or upload a Filters CSV to continue.")
    st.stop()

# Ensure expected columns
if "expression" not in filters_df.columns:
    st.error("Filters CSV must have an 'expression' column.")
    st.stop()
if "name" not in filters_df.columns:
    filters_df["name"] = filters_df["expression"].astype(str).str.slice(0, 60)
if "applicable_if" not in filters_df.columns:
    filters_df["applicable_if"] = ""

# ---- History CSV ----
st.subheader("History CSV")
history_path = st.text_input("Winner history CSV path (optional)", value="dc5_midday_full.csv")
history_df = None
winners_list = None
if history_path.strip():
    try:
        hdf = pd.read_csv(history_path)
        col = None
        for c in ["Result", "result", "Winning Numbers", "combo", "Combo"]:
            if c in hdf.columns:
                col = c
                break
        if col:
            winners_list = hdf[col].astype(str).tolist()
            history_df = hdf
        else:
            st.warning("History CSV loaded but couldn't find a winners column (Result/result/Winning Numbers/combo).")
    except Exception as e:
        st.warning(f"Could not read history CSV at {history_path}: {e}")

# ---- Evaluate filters on current pool ----
st.markdown("---")
st.subheader("Evaluating filters on current pool…")

base_env = build_ctx_for_pool(seed, prev_seed, prev_prev)

elim_counts = []
for _, row in filters_df.iterrows():
    expr = str(row["expression"]) if pd.notna(row["expression"]) else ""
    app_if = str(row.get("applicable_if", "") or "")
    eliminated = 0
    for combo in pool:
        cd = sorted(digits_of(combo))
        env = dict(base_env)
        env.update({
            "combo": combo,
            "combo_digits": cd,
            "combo_digits_list": cd,
            "combo_sum": sum(cd),
            "combo_sum_cat": sum_category(sum(cd)),
            "combo_sum_category": sum_category(sum(cd)),
            "combo_vtracs": set(VTRAC[d] for d in cd),
            "combo_structure": classify_structure(cd),
        })
        if app_if.strip():
            if not safe_eval(app_if, env):
                continue
        if safe_eval(expr, env):
            eliminated += 1
    elim_counts.append(eliminated)

filters_df["elim_count_on_pool"] = elim_counts
filters_df["kept_count_on_pool"] = pool_size - filters_df["elim_count_on_pool"]

# ---- Historical safety (optional) ----
hist_kept_rate = []
hist_blocked_rate = []
hist_applicable_days = []

if winners_list and len(winners_list) >= 2:
    for _, row in filters_df.iterrows():
        expr = str(row["expression"]) if pd.notna(row["expression"]) else ""
        app_if = str(row.get("applicable_if", "") or "")
        kept = 0
        blocked = 0
        applicable = 0
        for i in range(1, len(winners_list)):
            env = build_day_env(winners_list, i)
            if app_if.strip():
                if not safe_eval(app_if, env):
                    continue
            applicable += 1
            if safe_eval(expr, env):
                blocked += 1
            else:
                kept += 1
        hist_applicable_days.append(applicable)
        if applicable > 0:
            hist_kept_rate.append(kept / applicable)
            hist_blocked_rate.append(blocked / applicable)
        else:
            hist_kept_rate.append(None)
            hist_blocked_rate.append(None)
else:
    hist_kept_rate = [None] * len(filters_df)
    hist_blocked_rate = [None] * len(filters_df)
    hist_applicable_days = [0] * len(filters_df)

filters_df["hist_applicable_days"] = hist_applicable_days
filters_df["hist_kept_rate"] = hist_kept_rate
filters_df["hist_blocked_rate"] = hist_blocked_rate

# ---- Candidate Large Filters & outputs ----
if "parity_wiper" not in filters_df.columns:
    filters_df["parity_wiper"] = False

candidates = filters_df[filters_df["elim_count_on_pool"] >= int(min_elims)].copy()
if exclude_parity and "parity_wiper" in candidates.columns:
    candidates = candidates[~candidates["parity_wiper"]]

candidates = candidates.sort_values(
    by=["elim_count_on_pool", "hist_kept_rate"],
    ascending=[False, False],
    na_position="last"
)

st.subheader("Candidate Large Filters")
st.write(f"{len(candidates)} filters qualify as 'Large' (elim ≥ {int(min_elims)}).")
st.dataframe(candidates, use_container_width=True)

csv_bytes = candidates.to_csv(index=False).encode("utf-8")
names_txt = "\n".join(candidates["name"].astype(str).tolist()).encode("utf-8")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button("Download candidates CSV", data=csv_bytes, file_name=f"large_filters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
with col_d2:
    st.download_button("Download filter names (TXT)", data=names_txt, file_name=f"large_filter_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
