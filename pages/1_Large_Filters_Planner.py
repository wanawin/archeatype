# pages/1_Large_Filters_Planner.py
from __future__ import annotations

# -------------------------
# Standard imports
# -------------------------
import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from collections import Counter

# -------------------------
# Page & layout
# -------------------------
st.set_page_config(page_title="Archetype Helper — Large Filters, Triggers & Plans", layout="wide")
st.title("Archetype Helper — Large Filters, Triggers & Plans")

# -------------------------------------------------
# Core helpers & signals (same definitions as DC-5)
# -------------------------------------------------
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

def parse_list(text: str, to_int: bool=False) -> List:
    """
    Split user-pasted lists on commas, whitespace, semicolons, or newlines.
    Keeps order-less given your use (IDs and combos).
    """
    if not text:
        return []
    raw = (
        text.replace("\n", ",")
            .replace("\t", ",")
            .replace(";", ",")
            .replace("  ", " ")
            .replace(" ", ",")
    )
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if to_int:
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                # Skip non-int token
                pass
        return out
    return parts

# -------------------------------------------------
# History (optional) — read winner list
# -------------------------------------------------
def load_winners_csv(path_or_file) -> List[str]:
    if not path_or_file:
        return []
    try:
        if hasattr(path_or_file, "read"):  # Uploaded file-like
            df = pd.read_csv(path_or_file)
        else:
            df = pd.read_csv(path_or_file)
    except Exception:
        return []
    low = {c.lower(): c for c in df.columns}
    col = low.get("result") or low.get("combo") or None
    if not col:
        return []
    return [str(x).strip() for x in df[col].dropna().astype(str)]

# -------------------------------------------------
# Pool loading
# -------------------------------------------------
def load_pool_from_csv(f, col_name_hint: str="Result") -> List[str]:
    df = pd.read_csv(f)
    low = {c.lower(): c for c in df.columns}
    if col_name_hint and col_name_hint in df.columns:
        col = col_name_hint
    else:
        col = low.get("result") or low.get("combo")
        if not col:
            raise ValueError("Pool CSV must contain a 'Result' (or 'combo') column, or set the correct column hint.")
    return [str(x).strip() for x in df[col].dropna().astype(str)]

# -------------------------------------------------
# Filters CSV loading/normalization
# -------------------------------------------------
def coerce_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true","1","yes","y","t"}: return True
    if s in {"false","0","no","n","f"}: return False
    return False

def strip_quotes(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

def load_filters_csv(source) -> pd.DataFrame:
    # Accept path or uploaded file-like
    df = pd.read_csv(source)
    # Normalize expected columns
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # expression
    exp_col = lower.get("expression")
    if not exp_col:
        # Some old files had 'expr' or unnamed columns after merges
        exp_col = lower.get("expr") or lower.get("Unnamed: 14") or None
    if not exp_col:
        raise ValueError("Filters CSV must include an 'expression' column.")

    # id / fid
    id_col = lower.get("id") or lower.get("fid")
    if id_col is None:
        df["id"] = range(1, len(df) + 1)
    elif id_col != "id":
        df["id"] = df[id_col]

    # name (optional)
    name_col = lower.get("name")
    if name_col and name_col != "name":
        df["name"] = df[name_col]
    elif "name" not in df.columns:
        df["name"] = df["id"].astype(str)

    # applicable_if (optional)
    app_col = lower.get("applicable_if")
    if app_col and app_col != "applicable_if":
        df["applicable_if"] = df[app_col]
    elif "applicable_if" not in df.columns:
        df["applicable_if"] = "True"

    # enabled (optional)
    en_col = lower.get("enabled")
    if en_col and en_col != "enabled":
        df["enabled"] = df[en_col].map(coerce_bool)
    elif "enabled" not in df.columns:
        df["enabled"] = True
    else:
        df["enabled"] = df["enabled"].map(coerce_bool)

    # parity_wiper (optional)
    pw_col = lower.get("parity_wiper")
    if pw_col and pw_col != "parity_wiper":
        df["parity_wiper"] = df[pw_col].map(coerce_bool)
    elif "parity_wiper" not in df.columns:
        df["parity_wiper"] = False
    else:
        df["parity_wiper"] = df["parity_wiper"].map(coerce_bool)

    # Clean string columns (quotes, stray spaces)
    df["expression"] = df[exp_col].astype(str).map(strip_quotes)
    df["applicable_if"] = df["applicable_if"].astype(str).map(strip_quotes)
    df["name"] = df["name"].astype(str).map(strip_quotes)
    df["id"] = df["id"].astype(str).map(strip_quotes)

    # Keep canonical order
    keep = ["id","name","enabled","applicable_if","expression","parity_wiper"]
    extra = [c for c in df.columns if c not in keep]
    df = df[keep + extra]
    return df

# -------------------------------------------------
# Build environment (seed + context) for eval()
# -------------------------------------------------
def make_base_env(
    seed: str,
    prev_seed: str,
    prev_prev_seed: str,
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int],
) -> Dict:
    sd  = digits_of(seed)
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

        # some filters expect these keys
        "seed_value": int("".join(map(str, sd))) if sd else None,
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
        "d_in_common_to_both": set(base_env["seed_digits"]).intersection(cd) if base_env["seed_digits"] else set(),
    })
    return env

# -------------------------------------------------
# Filter evaluation
# -------------------------------------------------
def expr_true(expr: str, env: Dict) -> bool:
    # Evaluate a Python expression safely (no builtins)
    try:
        code = compile(expr, "<expr>", "eval")
        return bool(eval(code, {"__builtins__": {}}, env))
    except Exception:
        return False

def eval_filter_on_pool(
    row: pd.Series,
    pool: List[str],
    base_env: Dict,
    respect_applicable_if: bool,
) -> Tuple[Set[str], int]:
    expr = str(row["expression"])
    app_if = str(row.get("applicable_if", "True")).strip() or "True"

    eliminated: Set[str] = set()
    # Precompute compiled codes to speed
    try:
        code_expr = compile(expr, "<filter_expr>", "eval")
    except Exception:
        return set(), 0

    if respect_applicable_if:
        try:
            code_app = compile(app_if, "<applicable_if>", "eval")
        except Exception:
            # if app_if is broken, treat as False (filter not applicable)
            return set(), 0
    else:
        code_app = None  # ignored

    for c in pool:
        env = combo_env(base_env, c)
        # Check applicable_if first (if enforced)
        if code_app is not None:
            try:
                if not bool(eval(code_app, {"__builtins__": {}}, env)):
                    continue
            except Exception:
                continue
        # Now the elimination expression
        try:
            if bool(eval(code_expr, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # treat failure as "doesn't eliminate"
            pass

    return eliminated, len(eliminated)

# -------------------------------------------------
# Planner (greedy with beam width / max steps)
# -------------------------------------------------
def greedy_plan(
    candidates: pd.DataFrame,
    pool: List[str],
    base_env: Dict,
    beam_width: int,
    max_steps: int,
    respect_applicable_if: bool,
) -> Tuple[List[Dict], List[str]]:
    remaining = set(pool)
    seq: List[Dict] = []

    for step in range(int(max_steps)):
        if not remaining:
            break

        scored = []
        for _, r in candidates.iterrows():
            elim_set, cnt = eval_filter_on_pool(r, list(remaining), base_env, respect_applicable_if)
            if cnt > 0:
                scored.append((cnt, elim_set, r))

        if not scored:
            break

        # Try the top K by eliminations
        scored.sort(key=lambda x: x[0], reverse=True)
        best_cnt, best_set, best_row = max(scored[: int(beam_width)], key=lambda x: x[0])

        remaining -= best_set
        seq.append({
            "id": best_row["id"],
            "name": best_row.get("name", ""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining),
        })

        if best_cnt == 0:
            break

    return seq, sorted(list(remaining))

# -------------------------------------------------
# Sidebar controls (modes & knobs)
# -------------------------------------------------
with st.sidebar:
    st.header("Planner Mode")
    mode = st.radio(
        "Select Mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1,
        help="Playlist Reducer = original winner-preserving logic; Safe Filter Explorer = more candidates (lower threshold)."
    )
    if mode == "Playlist Reducer":
        default_min_elims = 120
        default_beam = 5
        default_steps = 15
    else:  # Safe Filter Explorer
        default_min_elims = 60
        default_beam = 6
        default_steps = 18

    min_elims = st.number_input("Min eliminations to call it ‘Large’", min_value=1, max_value=99999, value=default_min_elims, step=1)
    beam_width = st.number_input("Beam width (search breadth)", min_value=1, max_value=50, value=default_beam, step=1)
    max_steps = st.number_input("Max steps (search depth)", min_value=1, max_value=50, value=default_steps, step=1)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)
    respect_applicable_if = st.checkbox("Respect 'applicable_if' from Filters CSV", value=True,
        help="ON = obey applicability conditions; OFF = ignore and test every filter against the pool.")

# -------------------------------------------------
# Seed context (REQUIRED seed) + H/C/D
# -------------------------------------------------
st.subheader("Seed context (required)")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (1-back, 5 digits) *", value="")
prev_seed = c2.text_input("Prev seed (2-back, optional)", value="")
prev_prev = c3.text_input("Prev-prev seed (3-back, optional)", value="")

if not seed or len(digits_of(seed)) != 5:
    st.error("Seed must be exactly 5 digits.")
    st.stop()

st.subheader("Hot/Cold/Due digits (optional)")
c4, c5, c6 = st.columns(3)
hot_digits  = parse_list(c4.text_input("Hot digits", value=""), to_int=True)
cold_digits = parse_list(c5.text_input("Cold digits", value=""), to_int=True)
due_digits  = parse_list(c6.text_input("Due digits", value=""), to_int=True)

# -------------------------------------------------
# Combo Pool
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (comma/space/newline separated):", height=130, help="Paste straight from DevTools or any list; we'll split safely.")
pool_file = st.file_uploader("Or upload pool CSV", type=["csv"])
pool_col_hint = st.text_input("Pool column name (optional hint)", value="Result")

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

pool = [p for p in pool if p]  # clean blanks away
st.write(f"**Pool size:** {len(pool)}")

# -------------------------------------------------
# Winners history (optional, default path + override)
# -------------------------------------------------
st.subheader("Winners History (optional)")
hc1, hc2 = st.columns([2,1])
history_path = hc1.text_input(
    "Path to winners history CSV",
    value="DC5_Midday_Full_Cleaned_Expanded.csv",
    help="Leave as default or point to another (evening) history file."
)
history_file_up = hc2.file_uploader("Or upload history CSV", type=["csv"])
if history_file_up is not None:
    winners_list = load_winners_csv(history_file_up)
else:
    winners_list = load_winners_csv(history_path)

# We aren't auto-deriving H/C/D here; you can paste them above from DC-5 if you want.
# History is still relevant for your own cross-checks or future scoring.

# -------------------------------------------------
# Filters: default CSV OR uploaded CSV + optional pasted IDs
# -------------------------------------------------
st.subheader("Filters")
ids_text = st.text_area(
    "Paste applicable Filter IDs (optional)",
    value="",
    help="Comma/space/newline separated IDs; they will be matched against the selected Filters CSV."
)
filters_file_up = st.file_uploader("Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)", type=["csv"])
filters_csv_path_default = "lottery_filters_batch_10.csv"

source = filters_file_up if filters_file_up is not None else filters_csv_path_default
try:
    filters_df_full = load_filters_csv(source)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    st.stop()

# Apply optional ID subsetting
applicable_ids = set(parse_list(ids_text, to_int=False))
if applicable_ids:
    # compare as strings
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(applicable_ids)].copy()
else:
    filters_df = filters_df_full.copy()

# Policy exclusions/enabled
if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]].copy()
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True].copy()

st.write(f"**Filters loaded:** {len(filters_df)}")

# -------------------------------------------------
# Build base environment for evaluation
# -------------------------------------------------
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

# -------------------------------------------------
# RUN BUTTON
# -------------------------------------------------
run = st.button("▶ Run Planner", type="primary")

# -------------------------------------------------
# Evaluate filters against current pool
# -------------------------------------------------
st.subheader("Evaluating filters on current pool…")

if not run:
    st.info("No filters evaluated yet. Click **Run Planner** to compute.")
    st.stop()

scored_rows = []
for _, r in filters_df.iterrows():
    elim_set, cnt = eval_filter_on_pool(r, pool, base_env, respect_applicable_if)
    scored_rows.append({
        "id": r["id"],
        "name": r.get("name", ""),
        "enabled": bool(r.get("enabled", True)),
        "applicable_if": r.get("applicable_if", "True"),
        "expression": r["expression"],
        "parity_wiper": bool(r.get("parity_wiper", False)),
        "_elim_set": elim_set,  # internal
        "elim_count_on_pool": int(cnt),
        "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
    })

scored_df = pd.DataFrame(scored_rows)
if scored_df.empty:
    st.warning("No filters evaluated.")
    st.stop()

# -------------------------------------------------
# Candidate “Large” filters by threshold + table
# -------------------------------------------------
large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
large_df = large_df.sort_values(by=["elim_count_on_pool","elim_pct_on_pool"], ascending=False)

st.write(f"**{mode}** running on **{len(filters_df)}** filters")
st.write(f"**Large filters (≥ {min_elims} eliminated):** {len(large_df)}")

# Display a compact table of candidates (avoid showing the internal set column)
view_cols = ["id","name","enabled","applicable_if","expression","elim_count_on_pool","elim_pct_on_pool","parity_wiper"]
st.dataframe(large_df[view_cols], use_container_width=True)

# -------------------------------------------------
# Planner (greedy)
# -------------------------------------------------
st.subheader("Planner (greedy)")
if large_df.empty:
    st.info("No candidates meet the 'Large' threshold; nothing to plan.")
    kept_after = pool
    plan = []
else:
    plan_df = large_df[["id","name","expression","applicable_if"]].copy()
    plan, kept_after = greedy_plan(
        candidates=plan_df,
        pool=pool,
        base_env=base_env,
        beam_width=int(beam_width),
        max_steps=int(max_steps),
        respect_applicable_if=respect_applicable_if,
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
