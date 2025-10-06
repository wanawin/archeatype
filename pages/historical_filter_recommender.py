
import streamlit as st
import pandas as pd
import numpy as np
import ast
from collections import Counter
from datetime import datetime

# -----------------------------
# CONSTANTS / MAPPINGS (no placeholders)
# -----------------------------
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR_PAIRS = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

# -----------------------------
# HELPERS (pure functions, deterministic)
# -----------------------------
def digits_from_str(s: str):
    s = ''.join(ch for ch in str(s) if ch.isdigit())
    return [int(ch) for ch in s]

def sum_of_digits(digs):
    return sum(digs)

def parity_counts(digs):
    ev = sum(1 for d in digs if d % 2 == 0)
    od = len(digs) - ev
    return ev, od

def high_low_counts(digs, low_set={0,1,2,3,4}, high_set={5,6,7,8,9}):
    lo = sum(1 for d in digs if d in low_set)
    hi = sum(1 for d in digs if d in high_set)
    return hi, lo

def vtrac_groups(digs):
    return [V_TRAC_GROUPS[d] for d in digs]

def mirror_of(d):
    return MIRROR_PAIRS[d]

def mirrors(digs):
    return [MIRROR_PAIRS[d] for d in digs]

def carry_over_count(seed_digits, winner_digits):
    # number of digits appearing in both lists (multiset-style intersection)
    cseed = Counter(seed_digits)
    cwin = Counter(winner_digits)
    inter = sum(min(cseed[k], cwin[k]) for k in cseed.keys() & cwin.keys())
    return inter

def spread(digs):
    return max(digs) - min(digs) if digs else 0

def stringify_digits(digs):
    return ''.join(str(d) for d in digs)

def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    return 'High'

def seed_profile(digs):
    s = sum_of_digits(digs)
    ev, od = parity_counts(digs)
    hi, lo = high_low_counts(digs)
    vt = vtrac_groups(digs)
    spr = spread(digs)
    return {
        "seed_str": stringify_digits(digs),
        "sum": s,
        "sum_category": sum_category(s),
        "even": ev,
        "odd": od,
        "high": hi,
        "low": lo,
        "vtrac": vt,
        "spread": spr,
    }

def similarity_score(p_now: dict, p_hist: dict, weights=None):
    """Weighted distance-like score (higher is more similar)."""
    if weights is None:
        weights = {
            "sum": 1.0,
            "sum_category": 1.5,
            "even": 0.6,
            "odd": 0.6,
            "high": 0.8,
            "low": 0.8,
            "spread": 0.8,
            "vtrac": 1.2,  # proportion overlap
        }
    score = 0.0
    # numeric deltas normalized
    score += weights["sum"] * (1.0 - min(abs(p_now["sum"] - p_hist["sum"]) / 45.0, 1.0))
    score += weights["spread"] * (1.0 - min(abs(p_now["spread"] - p_hist["spread"]) / 9.0, 1.0))
    score += weights["even"] * (1.0 - min(abs(p_now["even"] - p_hist["even"]) / 5.0, 1.0))
    score += weights["odd"] * (1.0 - min(abs(p_now["odd"] - p_hist["odd"]) / 5.0, 1.0))
    score += weights["high"] * (1.0 - min(abs(p_now["high"] - p_hist["high"]) / 5.0, 1.0))
    score += weights["low"] * (1.0 - min(abs(p_now["low"] - p_hist["low"]) / 5.0, 1.0))
    # categorical
    score += weights["sum_category"] * (1.0 if p_now["sum_category"] == p_hist["sum_category"] else 0.0)
    # vtrac overlap
    now_v = Counter(p_now["vtrac"])
    hist_v = Counter(p_hist["vtrac"])
    overlap = sum(min(now_v[k], hist_v[k]) for k in now_v.keys() & hist_v.keys())
    score += weights["vtrac"] * (overlap / 5.0)
    return score

def safe_eval_expression(expr: str, context: dict) -> bool:
    """
    Evaluate a Python boolean expression safely with a restricted namespace.
    The expression should return True if the filter WOULD eliminate the winner.
    """
    # Allowed names are strictly from context
    allowed_names = {k: context[k] for k in context.keys()}
    # Disallow builtins by providing empty dict; eval only the expression
    return bool(eval(expr, {"__builtins__": {}}, allowed_names))

def try_parse_date(val):
    if pd.isna(val):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(val), fmt)
        except Exception:
            pass
    return None

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Historical Safety Filter Recommender (Additive Layer)", layout="wide")
st.title("Historical Safety Filter Recommender (Additive Layer)")
st.caption("This app **adds** a historical-safety recommendation layer without removing anything from your existing app.")

with st.expander("Upload Data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1) Seed/Winner History")
        hist_file = st.file_uploader(
            "Upload seed/winner history CSV or TXT",
            type=["csv", "txt"],
            help="Order matters: keep the chronological order as-is. If reverse chronological, leave it that way."
        )
        history_reverse_checkbox = st.checkbox(
            "History file is in reverse chronological order (newest first)",
            value=True,
            help="If checked, we will preserve the order but mark it so we can correctly reference 'previous' vs 'next' when needed."
        )
        st.info("The app uses the history **exactly** as provided (no reordering).")

    with col2:
        st.subheader("2) Filters Catalog")
        filt_file = st.file_uploader(
            "Upload Filters CSV",
            type=["csv"],
            help="Must include at least: id, name (or layman_explanation), expression"
        )
        st.caption("Expression semantics: **True means eliminate winner.**")

st.markdown("---")

with st.expander("Current Seed & Similarity Settings", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        seed_input = st.text_input("Enter current seed (digits only, e.g., 27500)", value="27500")
        seed_digits_now = digits_from_str(seed_input)
        if len(seed_digits_now) != 5:
            st.error("Seed must have exactly 5 digits (0–9).")
        seed_prof_now = seed_profile(seed_digits_now) if len(seed_digits_now) == 5 else None
        if seed_prof_now:
            st.write("Seed profile:", seed_prof_now)

    with colB:
        st.write("Similarity Weights")
        w_sum = st.slider("Weight: Sum", 0.0, 3.0, 1.0, 0.1)
        w_sumcat = st.slider("Weight: Sum Category", 0.0, 3.0, 1.5, 0.1)
        w_even = st.slider("Weight: Even Count", 0.0, 3.0, 0.6, 0.1)
        w_odd = st.slider("Weight: Odd Count", 0.0, 3.0, 0.6, 0.1)
        w_high = st.slider("Weight: High Count", 0.0, 3.0, 0.8, 0.1)
        w_low = st.slider("Weight: Low Count", 0.0, 3.0, 0.8, 0.1)
        w_spread = st.slider("Weight: Spread", 0.0, 3.0, 0.8, 0.1)
        w_vtrac = st.slider("Weight: V-Trac Overlap", 0.0, 3.0, 1.2, 0.1)
        weights = {"sum": w_sum, "sum_category": w_sumcat, "even": w_even, "odd": w_odd,
                   "high": w_high, "low": w_low, "spread": w_spread, "vtrac": w_vtrac}

    with colC:
        st.write("Neighbor Selection")
        k_neighbors = st.slider("Top-K similar seeds to use", 10, 300, 100, 5)
        min_similarity = st.slider("Minimum similarity score threshold", 0.0, 8.0, 4.0, 0.1)
        st.caption("Filters are evaluated on historical winners whose seeds are most similar to the current seed.")

st.markdown("---")

run_btn = st.button("Run Historical Safety Recommendation")

# -----------------------------
# CORE LOGIC
# -----------------------------
if run_btn:
    if not hist_file:
        st.error("Upload a seed/winner history file first.")
        st.stop()
    if not filt_file:
        st.error("Upload a filters CSV first.")
        st.stop()
    if seed_prof_now is None:
        st.error("Enter a valid 5-digit seed.")
        st.stop()

    # --- Load History ---
    # We accept flexible column names by trying common variants.
    try:
        if hist_file.name.lower().endswith(".txt"):
            # Attempt robust parsing for TXT with delims: comma or whitespace
            raw = hist_file.read().decode("utf-8", errors="ignore").strip().splitlines()
            # Expect 2 columns minimal: seed,winner (optionally date)
            rows = []
            for line in raw:
                parts = [p.strip() for p in line.replace("\t", " ").replace("  ", " ").split(",")]
                if len(parts) == 1:
                    parts = [p.strip() for p in line.split()]
                if len(parts) >= 2:
                    rows.append(parts[:3])  # date optional
            df_hist = pd.DataFrame(rows, columns=["seed", "winner", "date"][:len(rows[0])])
        else:
            df_hist = pd.read_csv(hist_file)
    except Exception as e:
        st.error(f"Could not read history file: {e}")
        st.stop()

    # Normalize expected columns: try to find seed and winner columns by heuristics
    cols_lower = {c.lower(): c for c in df_hist.columns}
    seed_col = None
    winner_col = None
    for key in ["seed", "prev", "previous", "prev_seed", "seed_value"]:
        if key in cols_lower:
            seed_col = cols_lower[key]
            break
    for key in ["winner", "result", "current", "draw", "win_value", "result_value"]:
        if key in cols_lower:
            winner_col = cols_lower[key]
            break

    if seed_col is None or winner_col is None:
        st.error(f"History file must contain seed and winner columns. Found columns: {list(df_hist.columns)}")
        st.stop()

    # Preserve order exactly as provided (no reordering). Mark whether it is reverse chronological.
    df_hist = df_hist[[seed_col, winner_col] + [c for c in df_hist.columns if c not in (seed_col, winner_col)]].copy()
    df_hist.rename(columns={seed_col: "seed", winner_col: "winner"}, inplace=True)
    df_hist["seed_digits"] = df_hist["seed"].apply(digits_from_str)
    df_hist["winner_digits"] = df_hist["winner"].apply(digits_from_str)

    # Drop any malformed rows safely
    df_hist = df_hist[df_hist["seed_digits"].apply(len) == 5]
    df_hist = df_hist[df_hist["winner_digits"].apply(len) == 5].reset_index(drop=True)

    # Precompute profiles
    prof_cols = ["seed_sum","seed_sum_category","seed_even","seed_odd","seed_high","seed_low","seed_vtrac","seed_spread"]
    for col in prof_cols:
        df_hist[col] = None
    prof_list = []
    for i, row in df_hist.iterrows():
        prof = seed_profile(row["seed_digits"])
        df_hist.at[i, "seed_sum"] = prof["sum"]
        df_hist.at[i, "seed_sum_category"] = prof["sum_category"]
        df_hist.at[i, "seed_even"] = prof["even"]
        df_hist.at[i, "seed_odd"] = prof["odd"]
        df_hist.at[i, "seed_high"] = prof["high"]
        df_hist.at[i, "seed_low"] = prof["low"]
        df_hist.at[i, "seed_vtrac"] = prof["vtrac"]
        df_hist.at[i, "seed_spread"] = prof["spread"]

    # Compute similarity
    sims = []
    for i, row in df_hist.iterrows():
        ph = {
            "sum": row["seed_sum"],
            "sum_category": row["seed_sum_category"],
            "even": row["seed_even"],
            "odd": row["seed_odd"],
            "high": row["seed_high"],
            "low": row["seed_low"],
            "vtrac": row["seed_vtrac"],
            "spread": row["seed_spread"],
        }
        sims.append(similarity_score(seed_prof_now, ph, weights))
    df_hist["similarity"] = sims

    # Select neighborhood
    df_nbrs = df_hist[df_hist["similarity"] >= min_similarity].copy()
    df_nbrs = df_nbrs.sort_values("similarity", ascending=False).head(k_neighbors)
    if df_nbrs.empty:
        st.warning("No similar historical seeds met the threshold. Try lowering the threshold or increasing K.")
        st.stop()

    st.success(f"Using {len(df_nbrs)} historical cases of similar seeds for safety evaluation.")

    # --- Load Filters CSV ---
    try:
        df_filters = pd.read_csv(filt_file)
    except Exception as e:
        st.error(f"Could not read filters CSV: {e}")
        st.stop()

    # Normalize expected columns
    cols_lower = {c.lower(): c for c in df_filters.columns}
    id_col = None
    name_col = None
    expr_col = None

    for key in ["id","filter_id","fid"]:
        if key in cols_lower:
            id_col = cols_lower[key]
            break
    for key in ["name","layman_explanation","layman","description","title"]:
        if key in cols_lower:
            name_col = cols_lower[key]
            break
    for key in ["expression","expr","rule"]:
        if key in cols_lower:
            expr_col = cols_lower[key]
            break

    if id_col is None or expr_col is None:
        st.error("Filters CSV must include at least 'id' and 'expression' columns (name strongly recommended).")
        st.stop()

    if name_col is None:
        name_col = id_col

    df_filters = df_filters[[id_col, name_col, expr_col] + [c for c in df_filters.columns if c not in (id_col, name_col, expr_col)]].copy()
    df_filters.rename(columns={id_col: "id", name_col: "name", expr_col: "expression"}, inplace=True)

    # Evaluate historical SAFETY for each filter over neighborhood
    results = []
    for _, filt in df_filters.iterrows():
        fid = str(filt["id"])
        fname = str(filt["name"])
        expr = str(filt["expression"]).strip()
        if not expr:
            continue

        eliminated_count = 0
        total_cases = 0

        for _, row in df_nbrs.iterrows():
            seed_d = row["seed_digits"]
            win_d = row["winner_digits"]
            context = {
                # canonical variables available to expressions
                "seed_digits": seed_d,
                "winner_digits": win_d,
                "seed_sum": sum_of_digits(seed_d),
                "winner_sum": sum_of_digits(win_d),
                "seed_even": parity_counts(seed_d)[0],
                "seed_odd": parity_counts(seed_d)[1],
                "winner_even": parity_counts(win_d)[0],
                "winner_odd": parity_counts(win_d)[1],
                "seed_high": high_low_counts(seed_d)[0],
                "seed_low": high_low_counts(seed_d)[1],
                "winner_high": high_low_counts(win_d)[0],
                "winner_low": high_low_counts(win_d)[1],
                "seed_vtrac": vtrac_groups(seed_d),
                "winner_vtrac": vtrac_groups(win_d),
                "seed_spread": spread(seed_d),
                "winner_spread": spread(win_d),
                "carry_overs": carry_over_count(seed_d, win_d),
                "mirror": MIRROR_PAIRS,
                "vtrac_map": V_TRAC_GROUPS,
                # convenience aliases seen in prior CSVs
                "combo_digits": win_d,
                "combo_sum": sum_of_digits(win_d),
                "combo_spread": spread(win_d),
            }
            try:
                would_eliminate = safe_eval_expression(expr, context)
            except Exception as e:
                would_eliminate = False  # expression error => treat as not safe to eliminate winner
            total_cases += 1
            if would_eliminate:
                eliminated_count += 1

        safety = 1.0 - (eliminated_count / total_cases if total_cases > 0 else 0.0)
        results.append({
            "id": fid,
            "name": fname,
            "expression": expr,
            "similar_cases": total_cases,
            "eliminated_winner_in_similar": eliminated_count,
            "historical_safety": round(safety, 6),
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        st.error("No evaluable filters found.")
        st.stop()

    # Rank: prioritize historical safety (desc), then similar_cases (desc)
    df_res = df_res.sort_values(["historical_safety", "similar_cases"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("Recommended Filters (Historical Safety First)")
    st.caption("Higher safety = less likely to eliminate the actual winner in similar seed contexts.")
    st.dataframe(df_res, hide_index=True, use_container_width=True)

    # Provide downloadable CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"historical_recommendations_{ts}.csv"
    st.download_button("Download Recommendations CSV", df_res.to_csv(index=False).encode("utf-8"), file_name=out_name, mime="text/csv")

    # Summary
    st.markdown("### Summary")
    top5 = df_res.head(5)[["id","name","historical_safety","similar_cases","eliminated_winner_in_similar"]]
    st.write(top5)

    st.success("Historical safety layer complete. This module **adds** recommendations based on past seeds similar to your current seed, giving precedence to historical safety without removing any existing logic.")
