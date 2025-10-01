import streamlit as st
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

# -----------------------
# Utilities
# -----------------------

VTRAC = {0: 1, 5: 1, 1: 2, 6: 2, 2: 3, 7: 3, 3: 4, 8: 4, 4: 5, 9: 5}
MIRROR = {0: 5, 5: 0, 1: 6, 6: 1, 2: 7, 7: 2, 3: 8, 8: 3, 4: 9, 9: 4}

def digits_of(x: str) -> List[int]:
    return [int(c) for c in str(x) if c.isdigit()]

def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    return 'High'

# -----------------------
# Build evaluation context
# -----------------------

def build_ctx_for_pool(seed: str, prev_seed: str, prev_prev: str) -> Dict:
    sd = digits_of(seed)
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev) if prev_prev else []

    new_digits = set(sd) - set(pdigs)
    seed_counts = Counter(sd)
    seed_vtracs = set(VTRAC[d] for d in sd)
    prev_sum_cat = sum_category(sum(sd))

    prev_pattern = []
    for digs in (ppdigs, pdigs, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    return {
        'seed_value': int(seed) if seed else None,
        'seed_sum': sum(sd),
        'prev_seed_sum': sum(pdigs) if pdigs else None,
        'prev_prev_seed_sum': sum(ppdigs) if ppdigs else None,
        'seed_digits_1': pdigs,
        'seed_digits_2': ppdigs,
        'nan': float('nan'),
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': new_digits,
        'prev_pattern': tuple(prev_pattern),
        'hot_digits': [],
        'cold_digits': [],
        'due_digits': [d for d in range(10) if d not in pdigs and d not in ppdigs],
        'seed_counts': seed_counts,
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted
    }

# -----------------------
# History safety computation
# -----------------------

def hist_safety(app_code, expr_code, winners_list: List[str]) -> Tuple[int, int, float, float]:
    """Return (applicable_days, blocked_days, kept_rate, blocked_rate) over history."""
    if not winners_list or len(winners_list) < 2:
        return 0, 0, None, None
    applicable = 0
    blocked = 0
    for i in range(1, len(winners_list)):
        ctx = build_ctx_for_pool(winners_list[i-1], winners_list[i-2] if i >= 2 else "", winners_list[i-3] if i >= 3 else "")
        try:
            if eval(app_code, ctx, ctx):
                applicable += 1
                if eval(expr_code, ctx, ctx):
                    blocked += 1
        except Exception:
            pass
    kept = applicable - blocked
    kept_rate = (kept / applicable) if applicable else None
    blocked_rate = (blocked / applicable) if applicable else None
    return applicable, blocked, kept_rate, blocked_rate

# -----------------------
# UI
# -----------------------

st.title("Large Filters Planner")

mode = st.sidebar.radio(
    "Mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    help=(
        "Playlist Reducer = original winner-preserving logic with stricter defaults.\n"
        "Safe Filter Explorer = lower elimination threshold and deeper search to surface more high-safety filters."
    ),
)

if mode == "Playlist Reducer":
    default_min_elims = 120
    default_beam = 5
    default_steps = 15
else:
    default_min_elims = 60
    default_beam = 6
    default_steps = 18

with st.sidebar:
    st.header("Inputs")
    seed = st.text_input("Seed (1-back, 5 digits) *", value="", max_chars=5).strip()
    prev_seed = st.text_input("Prev seed (2-back, optional)", value="", max_chars=5).strip()
    prev_prev = st.text_input("Prev-prev seed (3-back, optional)", value="", max_chars=5).strip()

    st.markdown("---")
    filters_path_str = st.text_input("Filters CSV path", value="lottery_filters_batch10.csv")
    winners_path_str = st.text_input("Winners CSV (for history)", value="DC5_Midday_Full_Cleaned_Expanded.csv")

    st.markdown("---")
    st.subheader("Paste/Upload Pools")
    pool_text = st.text_area("Paste combos here (one per line)", height=140)
    pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])

    st.markdown("---")
    st.subheader("Large Filter Rules")
    min_elims = st.number_input("Min eliminations to call it ‘Large’", min_value=1, max_value=99999, value=default_min_elims)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# Validate seed
if not (seed.isdigit() and len(seed) == 5):
    st.error("Seed must be exactly 5 digits.")
    st.stop()

# Now your app logic continues exactly as before...

# Validate seed
if not (seed.isdigit() and len(seed) == 5):
    st.error("Seed must be exactly 5 digits.")
    st.stop()

prof = seed_profile(seed, prev_seed, prev_prev)
st.caption(f"Seed signature: {prof['signature']}")

# -----------------------
# Load winners history (path input)
# -----------------------
if not winners_path_str.strip():
    winners_path_str = "DC5_Midday_Full_Cleaned_Expanded.csv"

try:
    winners_df = pd.read_csv(winners_path_str)
    # Accept either columns [Date, Number] or a single column with 5-digit strings
    if 'Number' in winners_df.columns:
        winners_list = winners_df['Number'].astype(str).tolist()
    else:
        # best-effort: take first column
        first_col = winners_df.columns[0]
        winners_list = winners_df[first_col].astype(str).tolist()
except Exception as e:
    winners_list = []
    st.warning(f"Could not load winners history from '{winners_path_str}': {e}")

# -----------------------
# Load pool (paste OR upload OR path)
# -----------------------
pool_digits: List[str] = []
if pool_text.strip():
    pool_digits = [ln.strip() for ln in pool_text.splitlines() if ln.strip()]
elif pool_file is not None:
    try:
        df_pool = pd.read_csv(pool_file)
        if 'Result' in df_pool.columns:
            pool_digits = df_pool['Result'].astype(str).tolist()
        else:
            # fallback: first column
            pool_digits = df_pool.iloc[:,0].astype(str).tolist()
    except Exception as e:
        st.error(f"Failed to read uploaded pool CSV: {e}")
        st.stop()
else:
    # optional: try a default path if you have one; otherwise require paste/upload
    st.info("Paste your combo pool or upload a pool CSV to continue.")
    st.stop()

pool_series = pd.Series(pool_digits)

# -----------------------
# Load filters (paste OR upload OR path)
# -----------------------
if filters_text.strip():
    try:
        from io import StringIO
        df_filters = pd.read_csv(StringIO(filters_text))
    except Exception as e:
        st.error(f"Failed to parse pasted filters CSV: {e}")
        st.stop()
elif filters_file is not None:
    try:
        df_filters = pd.read_csv(filters_file)
    except Exception as e:
        st.error(f"Failed to read uploaded filters CSV: {e}")
        st.stop()
else:
    try:
        df_filters = pd.read_csv(filters_path_str)
    except Exception as e:
        st.error(f"Could not load filters from '{filters_path_str}': {e}")
        st.stop()

# Normalize/clean
try:
    df_filters = clean_filter_df(df_filters)
except Exception as e:
    st.error(f"Filters CSV error: {e}")
    st.stop()

# Compile filters
compiled: Dict[str, Tuple[object, object]] = {}
for _, row in df_filters.iterrows():
    app_code, expr_code = compile_filter(row)
    fid = str(row["id"]).strip()
    compiled[fid] = (app_code, expr_code)

# -----------------------
# Elimination counts on YOUR pool
# -----------------------
seed_ctx_base = build_ctx_for_pool(seed, prev_seed, prev_prev)

records = []
for _, r in df_filters.iterrows():
    fid = str(r['id']).strip()
    name = r.get('name','')
    app_code, expr_code = compiled[fid]

    elim_count = 0
    elim_even = 0
    elim_odd = 0
    parity_wiper = False

    # Evaluate against current pool
    for combo in pool_digits:
        cd = digits_of(combo)
        ctx = seed_ctx_base.copy()
        ctx.update({
            'combo': combo,
            'combo_digits': cd,
            'combo_digits_list': cd,
            'combo_sum': sum(cd),
            'combo_sum_cat': sum_category(sum(cd)),
            'combo_sum_category': sum_category(sum(cd)),
            'combo_vtracs': set(VTRAC[d] for d in cd),
        })
        try:
            if eval(app_code, ctx, ctx) and eval(expr_code, ctx, ctx):
                elim_count += 1
                if sum(cd) % 2 == 0:
                    elim_even += 1
                else:
                    elim_odd += 1
        except Exception:
            # errors just don't eliminate that combo
            pass

    if elim_count > 0 and (elim_even == 0 or elim_odd == 0):
        parity_wiper = True

    # History-based safety
    app_days = blocked = None
    kept_rate = blocked_rate = None
    if winners_list and len(winners_list) >= 2:
        app_days, blocked, kept_rate, blocked_rate = hist_safety(app_code, expr_code, winners_list)

    # Seed-specific trigger heuristic (simple text scan)
    text_blob = f"{r.get('applicable_if','')} || {r.get('expression','')}".lower()
    seed_specific = ("seed" in text_blob) or ("winner" in text_blob)

    records.append({
        "filter_id": fid,
        "name": name,
        "elim_count_on_pool": elim_count,
        "elim_even": elim_even,
        "elim_odd": elim_odd,
        "parity_wiper": parity_wiper,
        "seed_specific_trigger": seed_specific,
        "hist_applicable_days": app_days,
        "hist_kept_rate": (None if kept_rate is None else round(100*kept_rate, 2)),
        "hist_blocked_rate": (None if blocked_rate is None else round(100*blocked_rate, 2)),
    })

df = pd.DataFrame(records)

# -----------------------
# Large & Trigger sets
# -----------------------

def bucket(c):
    c = int(c)
    if c >= 701: return "701+"
    if c >= 501: return "501–700"
    if c >= 301: return "301–500"
    if c >= 101: return "101–300"
    if c >=  61: return "61–100"
    if c >=   1: return "1–60"
    return "0"

large_df = df[(df["elim_count_on_pool"] >= int(min_elims))].copy()
if exclude_parity:
    large_df = large_df[~large_df["parity_wiper"]].copy()

large_df["aggression_group"] = large_df["elim_count_on_pool"].map(bucket)
_group_order = ["701+","501–700","301–500","101–300","61–100","1–60","0"]
rank_map = {g:i for i,g in enumerate(_group_order)}
large_df["__g"] = large_df["aggression_group"].map(rank_map).fillna(999).astype(int)

large_df = large_df.sort_values(
    by=["__g","hist_kept_rate","hist_applicable_days","elim_count_on_pool","filter_id"],
    ascending=[True, False, False, False, True]
).drop(columns="__g")

trig_df = df[df["seed_specific_trigger"]].copy().sort_values(by=["filter_id"]) if not df.empty else df

# -----------------------
# Show + Downloads (Large / Triggers)
# -----------------------
st.subheader("Large Filters (bucketed by eliminations on *your* pool)")
if large_df.empty:
    st.info("No large filters matched your rules among the pasted/uploaded filters for this pool.")
else:
    for g in _group_order:
        sub = large_df[large_df["aggression_group"] == g]
        if sub.empty:
            continue
        st.markdown(f"### Group **{g}**")
        st.dataframe(
            sub[[
                "aggression_group","filter_id","name","elim_count_on_pool","elim_even","elim_odd",
                "hist_applicable_days","hist_kept_rate","hist_blocked_rate"
            ]],
            use_container_width=True, hide_index=True, height=min(400, 60 + 28*len(sub))
        )

    # CSV download
    large_csv = "large_filters_detected.csv"
    large_df.to_csv(large_csv, index=False)
    st.download_button("Download ALL large filters (CSV)", data=Path(large_csv).read_bytes(), file_name=large_csv, mime="text/csv")

# Triggers
st.subheader("Seed-Specific Trigger Filters (by ID)")
if trig_df.empty:
    st.info("No seed-specific trigger filters detected among the pasted/uploaded filters.")
else:
    st.dataframe(
        trig_df[[
            "filter_id","name","elim_count_on_pool","elim_even","elim_odd",
            "hist_applicable_days","hist_kept_rate","hist_blocked_rate","parity_wiper"
        ]],
        use_container_width=True, hide_index=True, height=min(360, 60 + 28*len(trig_df))
    )
    trig_csv  = "trigger_filters_detected.csv"
    trig_df.to_csv(trig_csv, index=False)
    st.download_button("Download trigger filters (CSV)", data=Path(trig_csv).read_bytes(), file_name=trig_csv, mime="text/csv")

st.markdown("---")

st.caption("Note: Planning/sequencing logic is unchanged — this page focuses on surfacing more candidates with paste-or-upload inputs and a history path override.")
