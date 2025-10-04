# pages/1_Large_Filters_Planner.py
from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from collections import Counter
import streamlit as st

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
    """Split on commas, spaces, semicolons, or newlines. Keeps order of tokens encountered."""
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
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    return parts

# ---------------------------------------
# Seed / context env builders for eval()
# ---------------------------------------
def make_base_env(
    seed: str,
    prev_seed: str,
    prev_prev_seed: str,
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int],
) -> Dict:
    sd  = digits_of(seed) if seed else []
    sd2 = digits_of(prev_seed) if prev_seed else []
    sd3 = digits_of(prev_prev_seed) if prev_prev_seed else []

    base = {
        # seed digits & relatives
        "seed_digits": sd,
        "seed_digits_1": sd,                 # convenience alias
        "prev_seed_digits": sd2,
        "seed_digits_2": sd2,                # convenience alias
        "prev_prev_seed_digits": sd3,
        "seed_digits_3": sd3,                # convenience alias
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd) if sd else 0,
        "prev_seed_sum": sum(sd2) if sd2 else 0,
        "prev_prev_seed_sum": sum(sd3) if sd3 else 0,
        "prev_sum_cat": sum_category(sum(sd) if sd else 0),

        # vtrac/mirror
        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),
        "mirror": MIRROR,
        "VTRAC": VTRAC,

        # user-provided H/C/D (or empty)
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),

        # utility
        "Counter": Counter,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted,

        # sometimes referenced by older filters
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
        # convenience for some filters
        "last_digit": cd[-1] if cd else None,
        "spread": (max(cd) - min(cd)) if cd else 0,
        "seed_spread": (max(base_env["seed_digits"]) - min(base_env["seed_digits"]))
                        if base_env["seed_digits"] else 0,
    })
    return env

# ---------------------------------------
# CSV loaders (pool, winners, filters)
# ---------------------------------------
def load_pool_from_csv(f, col_name_hint: str) -> List[str]:
    df = pd.read_csv(f)
    cols_lower = {c.lower(): c for c in df.columns}
    # Try hint first
    if col_name_hint and col_name_hint in df.columns:
        s = df[col_name_hint]
    elif "result" in cols_lower:
        s = df[cols_lower["result"]]
    elif "combo" in cols_lower:
        s = df[cols_lower["combo"]]
    else:
        raise ValueError("Pool CSV must contain a 'Result' column (or set a valid column hint).")
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_winners_csv(path: str, uploaded_file) -> List[str]:
    # If uploader used, prefer it
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            return []
    else:
        if not path:
            return []
        try:
            df = pd.read_csv(path)
        except Exception:
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
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    # normalize id/fid
    if "id" not in df.columns and "fid" in df.columns:
        df["id"] = df["fid"]
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)
    # require expression
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)
    # optional parity_wiper column; fill if missing
    if "parity_wiper" not in df.columns:
        df["parity_wiper"] = False
    # enabled column optional
    if "enabled" not in df.columns:
        df["enabled"] = True
    return df

# ---------------------------------------
# Filter evaluation
# ---------------------------------------
def eval_filter_on_pool(
    row: pd.Series,
    pool: List[str],
    base_env: Dict,
) -> Tuple[Set[str], int]:
    """
    Return (eliminated_set, count). Expression returns True to eliminate combo.
    """
    expr = str(row["expression"])
    try:
        code = compile(expr, "<filter_expr>", "eval")
    except Exception:
        # invalid expression => eliminates nothing
        return set(), 0

    eliminated = set()
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # If expression errors on this combo, treat as not eliminating that combo
            pass
    return eliminated, len(eliminated)

def greedy_plan(
    candidates: pd.DataFrame,
    pool: List[str],
    base_env: Dict,
    beam_width: int,
    max_steps: int,
) -> Tuple[List[Dict], List[str]]:
    """
    Simple, fast greedy planner.
    At each step: re-calc elimination on remaining; try top <beam_width>; pick best.
    """
    remaining = set(pool)
    chosen: List[Dict] = []

    for step in range(max_steps):
        if not remaining:
            break

        # Re-score on current remaining pool
        scored = []
        for _, r in candidates.iterrows():
            elim, cnt = eval_filter_on_pool(r, list(remaining), base_env)
            if cnt > 0:
                scored.append((cnt, elim, r))
        if not scored:
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        # try top beam_width and pick best
        best_cnt, best_elim, best_row = max(scored[:beam_width], key=lambda x: x[0])

        # apply
        remaining -= best_elim
        chosen.append({
            "id": best_row["id"],
            "name": best_row.get("name", ""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining),
        })

        # stop if no progress
        if best_cnt == 0:
            break

    return chosen, sorted(list(remaining))

# ---------------------------------------
# Historical safety scoring
# ---------------------------------------
def seed_fingerprint(digs: List[int]) -> Tuple[str, str, int]:
    """Structure, even/odd label, and sum category to match similar seeds."""
    structure = classify_structure(digs)
    parity = "Even" if (sum(digs) % 2 == 0) else "Odd"
    sum_cat = sum_category(sum(digs))
    return structure, parity, sum_cat

def safety_score_for_filter_on_history(
    row: pd.Series,
    winners: List[str],
    seed_value: str,
    prev_seed: str,
    prev_prev_seed: str,
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int],
    similarity_window: int = 2500,   # use up to this many back days (if available)
) -> Tuple[float, int, int]:
    """
    For similar past seeds, measure how often the filter would have killed the actual winner.
    Returns (safety_score in [0..1], similar_count, killed_count).
    Higher is safer (killing fewer winners).
    """
    if not winners or not seed_value:
        return (float("nan"), 0, 0)

    # Build "today" fingerprint
    sd = digits_of(seed_value)
    if len(sd) != 5:
        return (float("nan"), 0, 0)
    fp_target = seed_fingerprint(sd)

    # build last K days list of seeds (previous draws act as seed for next day)
    N = len(winners)
    if N < 2:
        return (float("nan"), 0, 0)
    K = min(similarity_window, N - 1)

    # compile once
    expr = str(row["expression"])
    try:
        code = compile(expr, "<filter_expr>", "eval")
    except Exception:
        return (float("nan"), 0, 0)

    similar = 0
    killed = 0

    # we consider day i as "today", so seed is winners[i-1], winner is winners[i]
    # loop i from 1..N-1
    for i in range(1, K + 1):
        seed_i = winners[i-1]
        win_i  = winners[i]

        sd_i = digits_of(seed_i)
        if len(sd_i) != 5:  # require 5-digit seed
            continue
        if seed_fingerprint(sd_i) != fp_target:
            continue

        prev_i = winners[i-2] if i-2 >= 0 else ""
        prevprev_i = winners[i-3] if i-3 >= 0 else ""

        base_env_i = make_base_env(
            seed=seed_i,
            prev_seed=prev_i,
            prev_prev_seed=prevprev_i,
            hot_digits=hot_digits,
            cold_digits=cold_digits,
            due_digits=due_digits,
        )
        # evaluate expression on the actual winning combo that day
        env_win = combo_env(base_env_i, win_i)
        similar += 1
        try:
            eliminated = bool(eval(code, {"__builtins__": {}}, env_win))
            if eliminated:
                killed += 1
        except Exception:
            # treat errors as not eliminating the winner
            pass

    if similar == 0:
        return (float("nan"), 0, 0)

    # safety is fraction of similar days where the filter did NOT kill the winner
    safety = (similar - killed) / similar
    return (safety, similar, killed)

# -------------------------------------------------
# SIDEBAR: modes & knobs
# -------------------------------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1,
        help="Playlist Reducer = larger eliminations (kept logic).  Safe Filter Explorer = lower threshold & deeper search."
    )
    if mode == "Playlist Reducer":
        default_min_elims = 120
        default_beam = 5
        default_steps = 15
    else:
        default_min_elims = 60
        default_beam = 6
        default_steps = 18

    min_elims = st.number_input("Min eliminations to call it 'Large'", 1, 99999, value=default_min_elims, step=1)
    beam_width = st.number_input("Beam width (search breadth)", 1, 50, value=default_beam, step=1)
    max_steps = st.number_input("Max steps (search depth)", 1, 50, value=default_steps, step=1)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# -------------------------------------------------
# SEED context & H/C/D
# -------------------------------------------------
st.subheader("Seed Context")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (prev draw)", value="")
prev_seed = c2.text_input("Prev Seed (2-back)", value="")
prev_prev = c3.text_input("Prev Prev Seed (3-back)", value="")

st.subheader("Hot / Cold / Due digits (optional)")
c4, c5, c6 = st.columns(3)
hot_txt  = c4.text_input("Hot digits (comma-separated)", value="")
cold_txt = c5.text_input("Cold digits (comma-separated)", value="")
due_txt  = c6.text_input("Due digits (comma-separated)", value="")
hot_digits  = parse_list(hot_txt, to_int=True)
cold_digits = parse_list(cold_txt, to_int=True)
due_digits  = parse_list(due_txt, to_int=True)

# -------------------------------------------------
# Combo Pool: paste OR upload
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV with 'Result' col OR one-per-line / comma / space separated):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")

pool: List[str] = []
if pool_text.strip():
    # accept either a CSV blob with 'Result' column, OR free-form tokens
    if "Result" in pool_text.splitlines()[0] and "," in pool_text:
        try:
            df_pool = pd.read_csv(io.StringIO(pool_text))
            if pool_col_hint and pool_col_hint in df_pool.columns:
                pool = df_pool[pool_col_hint].astype(str).dropna().tolist()
            elif 'Result' in df_pool.columns:
                pool = df_pool['Result'].astype(str).dropna().tolist()
            else:
                st.error("Pasted CSV does not include a 'Result' column.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to parse pasted CSV for pool ➜ {e}")
            st.stop()
    else:
        pool = parse_list(pool_text, to_int=False)
elif pool_file is not None:
    try:
        pool = load_pool_from_csv(pool_file, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to load pool CSV ➜ {e}")
        st.stop()
else:
    st.info("Paste CSV combos or upload a pool CSV to continue.")
    st.stop()

pool = [p for p in pool if p]  # clean
st.write(f"**Pool size**: {len(pool)}")

# -------------------------------------------------
# Winners History CSV (default path OR upload)
# -------------------------------------------------
st.subheader("Winners History")
hcol1, hcol2 = st.columns([2,1])
history_path = hcol1.text_input(
    "Path to winners history CSV",
    value="DC5_Midday_Full_Cleaned_Expanded.csv",
    help="Default stays as-is. You can override the path or upload a file instead."
)
history_upload = hcol2.file_uploader("Or upload history CSV (Result/Combo column)", type=["csv"])
winners_list = load_winners_csv(history_path, history_upload)
if not winners_list:
    st.warning("History CSV not found or empty. Historical safety scoring will be limited to your manual H/C/D and seed input.")

# -------------------------------------------------
# Filters: default CSV OR uploaded CSV + optional pasted IDs
# -------------------------------------------------
st.subheader("Filters")
ids_text = st.text_area(
    "Paste applicable Filter IDs (optional, CSV OR one-per-line / comma / space / newline). "
    "If provided, they will be matched in the selected filters CSV.",
    height=90,
)
filters_file_up = st.file_uploader(
    "Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)",
    type=["csv"]
)

# choose CSV source
filters_csv_path_default = "lottery_filters_batch_10.csv"
filters_source = filters_file_up if filters_file_up is not None else filters_csv_path_default

try:
    filters_df_full = load_filters_csv(filters_source)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    st.stop()

# If IDs pasted, subset
applicable_ids_raw = set(parse_list(ids_text, to_int=False))
if applicable_ids_raw:
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(applicable_ids_raw)].copy()
    if filters_df.empty:
        st.error("None of the pasted IDs matched rows in the selected Filters CSV.")
        st.stop()
else:
    filters_df = filters_df_full.copy()

# Optionally exclude parity wipers if column exists
if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]].copy()

# Keep only enabled
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True].copy()

st.write(f"**Filters loaded:** {len(filters_df)} (after ID/policy filtering)")

# -------------------------------------------------
# Build base environment from current context
# -------------------------------------------------
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

# -------------------------------------------------
# RUN BUTTON — stops auto recomputation noise
# -------------------------------------------------
run_clicked = st.button("▶ Run Planner")
if not run_clicked:
    st.stop()

# -------------------------------------------------
# Score filters on current pool
# -------------------------------------------------
st.subheader("Evaluating filters on current pool…")
scored_rows = []
for _, r in filters_df.iterrows():
    elim_set, cnt = eval_filter_on_pool(r, pool, base_env)
    # safety score against similar seeds in history
    safety, similar_n, killed_n = safety_score_for_filter_on_history(
        row=r,
        winners=winners_list,
        seed_value=seed,
        prev_seed=prev_seed,
        prev_prev_seed=prev_prev,
        hot_digits=hot_digits,
        cold_digits=cold_digits,
        due_digits=due_digits,
    )
    scored_rows.append({
        "id": r["id"],
        "name": r.get("name", ""),
        "expression": r["expression"],
        "elim_count_on_pool": cnt,
        "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
        "parity_wiper": bool(r.get("parity_wiper", False)),
        "safety_score": safety,
        "similar_seed_count": similar_n,
        "killed_winner_count": killed_n,
        "_elim_set": elim_set,   # internal use for planning
    })

scored_df = pd.DataFrame(scored_rows)
if scored_df.empty:
    st.warning("No filters evaluated (empty).")
    st.stop()

# Display scored list
disp_cols = ["id", "name", "elim_count_on_pool", "elim_pct_on_pool",
             "safety_score", "similar_seed_count", "killed_winner_count"]
st.dataframe(scored_df[disp_cols].sort_values(
    by=["elim_count_on_pool","safety_score"], ascending=[False, False]
), use_container_width=True)

# -------------------------------------------------
# Candidate “Large” filters by threshold
# -------------------------------------------------
st.subheader(f"Candidate Large Filters (≥ {int(min_elims)} eliminated on current pool)")
if "elim_count_on_pool" not in scored_df.columns:
    st.error("Internal error: elimination counts missing.")
    st.stop()

large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
large_df = large_df.sort_values(by=["elim_count_on_pool", "safety_score"], ascending=[False, False])
st.write(f"**Large filters:** {len(large_df)}")
if len(large_df):
    st.dataframe(
        large_df[["id","name","elim_count_on_pool","elim_pct_on_pool","safety_score","similar_seed_count","killed_winner_count"]],
        use_container_width=True
    )

# -------------------------------------------------
# Greedy planning (fast; uses beam_width & max_steps)
# -------------------------------------------------
st.subheader("Planner (greedy)")
if large_df.empty:
    st.info("No candidates meet the 'Large' threshold; nothing to plan.")
    kept_after = pool
    plan = []
else:
    plan_df = large_df[["id", "name", "expression"]].copy()

    plan, kept_after = greedy_plan(
        candidates=plan_df,
        pool=pool,
        base_env=base_env,
        beam_width=int(beam_width),
        max_steps=int(max_steps),
    )

st.write(f"**Kept combos after plan:** {len(kept_after)} / {len(pool)}")
if plan:
    st.write("**Chosen sequence (in order):**")
    st.dataframe(pd.DataFrame(plan), use_container_width=True)

# -------------------------------------------------
# Downloads
# -------------------------------------------------
st.subheader("Downloads")
kept_df = pd.DataFrame({"Result": kept_after})
removed = sorted(set(pool) - set(kept_after))
removed_df = pd.DataFrame({"Result": removed})

cA, cB = st.columns(2)
cA.download_button("Download KEPT combos (CSV)", kept_df.to_csv(index=False), file_name="kept_combos.csv", mime="text/csv")
cA.download_button("Download KEPT combos (TXT)", "\n".join(kept_after), file_name="kept_combos.txt", mime="text/plain")
cB.download_button("Download REMOVED combos (CSV)", removed_df.to_csv(index=False), file_name="removed_combos.csv", mime="text/csv")
cB.download_button("Download REMOVED combos (TXT)", "\n".join(removed), file_name="removed_combos.txt", mime="text/plain")
