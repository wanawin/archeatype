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
# Page config
# -----------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

# -----------------------
# Core helpers & signals
# -----------------------
VTRAC: Dict[int, int] = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s).strip() if ch.isdigit()]

def parse_list(text: str, to_int: bool=False) -> List:
    if not text:
        return []
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

def parity_name(digs: List[int]) -> str:
    return "Even" if (sum(digs) % 2 == 0) else "Odd"

# -----------------------
# Seed env assembly
# -----------------------
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
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd) if sd else 0,
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

        # for older filters
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

# -----------------------
# CSV loaders
# -----------------------
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
        raise ValueError("Pool CSV must contain a 'Result' column (or set a valid column hint).")
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_winners_csv(source) -> List[str]:
    try:
        if hasattr(source, "read"):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)
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
    if hasattr(source, "read"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    # normalize ids
    if "id" not in df.columns and "fid" in df.columns:
        df["id"] = df["fid"]
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)

    # basic columns
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "enabled" not in df.columns:
        df["enabled"] = True
    if "applicable_if" not in df.columns:
        df["applicable_if"] = "True"
    if "parity_wiper" not in df.columns:
        df["parity_wiper"] = False
    return df

# -----------------------
# Evaluation helpers
# -----------------------
def filter_applies(row: pd.Series, env: Dict) -> bool:
    """Check 'applicable_if' guard before evaluating elimination expression."""
    guard = str(row.get("applicable_if", "True"))
    try:
        code = compile(guard, "<guard>", "eval")
        return bool(eval(code, {"__builtins__": {}}, env))
    except Exception:
        # If guard is broken, treat as not applicable to be safe
        return False

def eval_filter_on_pool_row(row: pd.Series, pool: List[str], base_env: Dict) -> Tuple[Set[str], int]:
    """Return (eliminated_set, count) on current pool."""
    expr = str(row["expression"])
    try:
        code = compile(expr, "<filter_expr>", "eval")
    except Exception:
        return set(), 0

    eliminated = set()
    for c in pool:
        env = combo_env(base_env, c)
        if not filter_applies(row, env):
            continue
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # robust: treat error as not eliminating that combo
            pass
    return eliminated, len(eliminated)

def is_similar_seed(curr_sd: List[int], hist_sd: List[int]) -> bool:
    if not curr_sd or not hist_sd:
        return False
    return (
        classify_structure(curr_sd) == classify_structure(hist_sd) and
        parity_name(curr_sd) == parity_name(hist_sd) and
        sum_category(sum(curr_sd)) == sum_category(sum(hist_sd))
    )

def historical_safety(
    row: pd.Series,
    winners: List[str],
    hot: List[int],
    cold: List[int],
    due: List[int],
    curr_seed: str,
) -> Tuple[int, int, float]:
    """
    Compute safety over history on days with seeds similar to the current seed.
    Safety = 1 - (#times filter would have eliminated the actual winner on similar-seed days / #similar-seed days)
    Returns: (unsafe_hits, total_similar, safety_float)
    """
    if len(winners) < 4 or not curr_seed:
        return 0, 0, 1.0  # not enough data

    curr_sd = digits_of(curr_seed)
    expr_str = str(row["expression"])
    try:
        expr = compile(expr_str, "<hist_expr>", "eval")
    except Exception:
        return 0, 0, 1.0

    unsafe = 0
    total = 0
    # day i uses winner[i-1] as seed; i from 1..N-1
    for i in range(1, len(winners)):
        seed = winners[i-1]
        win = winners[i]
        sd = digits_of(seed)
        if not is_similar_seed(curr_sd, sd):
            continue
        total += 1
        base = make_base_env(seed, winners[i-2] if i-2 >= 0 else "", winners[i-3] if i-3 >= 0 else "", hot, cold, due)
        env = combo_env(base, win)
        if not filter_applies(row, env):
            continue
        try:
            if bool(eval(expr, {"__builtins__": {}}, env)):
                unsafe += 1  # would have eliminated the true winner -> unsafe
        except Exception:
            # evaluation error => treat as unsafe to be conservative
            unsafe += 1

    safety = 1.0
    if total > 0:
        safety = 1.0 - (unsafe / total)
    return unsafe, total, safety

def greedy_plan(
    candidates_df: pd.DataFrame,
    pool: List[str],
    base_env: Dict,
    beam_width: int,
    max_steps: int,
) -> Tuple[List[Dict], List[str]]:
    """Greedy chooser: at each step, recompute elim on remaining, try top beam, pick best."""
    remaining = set(pool)
    chosen: List[Dict] = []

    # Convert rows we need
    cand_rows = candidates_df[["id", "name", "expression"]].to_dict("records")

    for step in range(int(max_steps)):
        if not remaining:
            break

        scored = []
        for r in cand_rows:
            # build a series-like object
            s = pd.Series(r)
            elim, cnt = eval_filter_on_pool_row(s, list(remaining), base_env)
            if cnt > 0:
                scored.append((cnt, elim, r))
        if not scored:
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        best_cnt, best_elim, best_row = max(scored[: int(beam_width)], key=lambda x: x[0])

        remaining -= best_elim
        chosen.append({
            "id": best_row["id"],
            "name": best_row.get("name", ""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining),
        })
        if best_cnt == 0:
            break

    return chosen, sorted(list(remaining))

# -----------------------
# Sidebar: modes & knobs
# -----------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1,
        help="Reducer = bigger cuts (kept logic). Explorer = more candidates (lower threshold)."
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

# -----------------------
# Seed context & H/C/D
# -----------------------
st.subheader("Seed Context")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (prev)", value="")
prev_seed = c2.text_input("Prev Seed (2-back)", value="")
prev_prev = c3.text_input("Prev Prev Seed (3-back)", value="")

st.subheader("Hot/Cold/Due digits (optional)")
c4, c5, c6 = st.columns(3)
hot_digits  = parse_list(c4.text_input("Hot digits", value=""), to_int=True)
cold_digits = parse_list(c5.text_input("Cold digits", value=""), to_int=True)
due_digits  = parse_list(c6.text_input("Due digits", value=""), to_int=True)

# -----------------------
# Combo Pool
# -----------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma/space/newline separated):", height=140)
pool_file = st.file_uploader("Or upload pool CSV", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")

pool: List[str] = []
if pool_text.strip():
    pool = parse_list(pool_text, to_int=False)
elif pool_file is not None:
    try:
        pool = load_pool_from_csv(pool_file, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to load pool CSV ➜ {e}")
        st.stop()
else:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

pool = [p for p in pool if p]
st.write(f"**Pool size:** {len(pool)}")

# -----------------------
# Winners history
# -----------------------
st.subheader("Winners History")
hist_col1, hist_col2 = st.columns([3,2])
history_path = hist_col1.text_input(
    "Path to winners history CSV (leave default or override)",
    value="DC5_Midday_Full_Cleaned_Expanded.csv"
)
history_upload = hist_col2.file_uploader("...or upload a history CSV", type=["csv"])

if history_upload is not None:
    winners_list = load_winners_csv(history_upload)
else:
    winners_list = load_winners_csv(history_path)

if not winners_list:
    st.warning("History not found/empty. Safety ratings will default to safe=1.0.")
else:
    st.write(f"Loaded **{len(winners_list)}** historical winners.")

# -----------------------
# Filters: IDs paste + CSV (default or uploaded)
# -----------------------
st.subheader("Filters")
ids_text = st.text_area(
    "Paste applicable Filter IDs (CSV or one-per-line or comma/space separated) — optional.",
    height=100,
    help="If provided, we'll subset the filter CSV by these IDs."
)
filters_file_up = st.file_uploader("Upload Filters CSV (omit to use default lottery_filters_batch_10.csv)", type=["csv"])
filters_csv_path_default = "lottery_filters_batch_10.csv"
filters_source = filters_file_up if filters_file_up is not None else filters_csv_path_default

try:
    filters_df_full = load_filters_csv(filters_source)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    st.stop()

# If IDs pasted, subset
applicable_ids = set([s.strip() for s in parse_list(ids_text, to_int=False)])
if applicable_ids:
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(applicable_ids)].copy()
    if filters_df.empty:
        st.error("None of the pasted IDs matched rows in the selected Filters CSV.")
        st.stop()
else:
    filters_df = filters_df_full.copy()

# Optionally exclude parity wipers
if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]].copy()

# Only enabled
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True].copy()

st.write(f"**Filters loaded:** {len(filters_df)}")

# -----------------------
# Run button — full pipeline
# -----------------------
run = st.button("Run Planner", type="primary")

if run:
    if not seed or len(digits_of(seed)) != 5:
        st.error("Seed must be exactly 5 digits.")
        st.stop()

    base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

    # 1) Score filters on current pool
    st.subheader("Evaluating filters on current pool…")
    scored_rows = []
    for _, r in filters_df.iterrows():
        elim_set, cnt = eval_filter_on_pool_row(r, pool, base_env)
        scored_rows.append({
            "id": r["id"],
            "name": r.get("name", ""),
            "expression": r["expression"],
            "elim_count_on_pool": cnt,
            "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
            "parity_wiper": bool(r.get("parity_wiper", False)),
            "_elim_set": elim_set,
        })
    scored_df = pd.DataFrame(scored_rows)
    if scored_df.empty:
        st.warning("No filters evaluated.")
        st.stop()

    # 2) Historical safety on similar seeds
    st.subheader("Computing historical safety on similar seeds…")
    safety_list = []
    for _, r in filters_df.iterrows():
        unsafe, total, safe = historical_safety(
            r, winners_list, hot_digits, cold_digits, due_digits, seed
        )
        safety_list.append({"id": r["id"], "unsafe_hits": unsafe, "total_similar": total, "safety": safe})
    safety_df = pd.DataFrame(safety_list)

    # 3) Merge scores + safety
    merged = scored_df.merge(safety_df, on="id", how="left")
    merged["safety"] = merged["safety"].fillna(1.0)
    merged["total_similar"] = merged["total_similar"].fillna(0).astype(int)
    merged["unsafe_hits"] = merged["unsafe_hits"].fillna(0).astype(int)

    # 4) Candidates threshold by eliminations
    st.subheader("Candidate 'Large' filters")
    large_df = merged[merged["elim_count_on_pool"] >= int(min_elims)].copy()
    large_df = large_df.sort_values(
        by=["safety", "elim_count_on_pool", "elim_pct_on_pool"],
        ascending=[False, False, False]
    )
    st.write(f"**Large filters (≥ {min_elims} eliminated):** {len(large_df)}")
    st.dataframe(
        large_df[["id", "name", "elim_count_on_pool", "elim_pct_on_pool", "safety", "total_similar", "unsafe_hits"]],
        use_container_width=True
    )

    # 5) Greedy plan over candidates (uses expressions)
    st.subheader("Planner (greedy)")
    if large_df.empty:
        st.info("No candidates meet the 'Large' threshold; nothing to plan.")
        kept_after = pool
        plan = []
    else:
        plan, kept_after = greedy_plan(
            candidates_df=large_df[["id", "name", "expression"]],
            pool=pool,
            base_env=base_env,
            beam_width=int(beam_width),
            max_steps=int(max_steps),
        )
    st.write(f"**Kept combos after plan:** {len(kept_after)} / {len(pool)}")
    if plan:
        st.write("**Chosen sequence (in order):**")
        st.dataframe(pd.DataFrame(plan), use_container_width=True)

    # 6) Downloads
    st.subheader("Downloads")
    kept_df = pd.DataFrame({"Result": kept_after})
    removed = sorted(set(pool) - set(kept_after))
    removed_df = pd.DataFrame({"Result": removed})
    cA, cB = st.columns(2)
    cA.download_button("Download KEPT combos (CSV)", kept_df.to_csv(index=False), file_name="kept_combos.csv", mime="text/csv")
    cA.download_button("Download KEPT combos (TXT)", "\n".join(kept_after), file_name="kept_combos.txt", mime="text/plain")
    cB.download_button("Download REMOVED combos (CSV)", removed_df.to_csv(index=False), file_name="removed_combos.csv", mime="text/csv")
    cB.download_button("Download REMOVED combos (TXT)", "\n".join(removed), file_name="removed_combos.txt", mime="text/plain")
