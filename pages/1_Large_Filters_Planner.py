# pages/1_Large_Filters_Planner.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")

st.title("Archetype Helper — Large Filters, Triggers & Plans")

# -------------------------------------------------
# Core helpers & signals
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

def parse_token_list(text: str) -> List[str]:
    """
    Split ANY pasted text into tokens by comma/space/newline/semicolon/tab.
    Returns strings exactly as written (no eval, no int cast) to avoid
    'invalid decimal literal' or similar issues.
    """
    if not text:
        return []
    raw = (
        text.replace("\n", ",")
            .replace("\t", ",")
            .replace(";", ",")
            .replace(" ", ",")
    )
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

# -------------------------------------------------
# Env builders (seed, history-aware context)
# -------------------------------------------------
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

    prev_pattern = []
    for digs in (sd3, sd2, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    base = {
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd) if sd else 0,
        "prev_sum_cat": sum_category(sum(sd) if sd else 0),
        "prev_pattern": tuple(prev_pattern),

        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),
        "mirror": MIRROR,
        "VTRAC": VTRAC,

        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),

        "Counter": Counter,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted,

        "seed_value": int(seed) if seed and seed.isdigit() else None,
        "nan": float("nan"),
        "winner_structure": classify_structure(sd) if sd else "",
        "combo_structure": "",   # set per combo
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

# -------------------------------------------------
# CSV loaders
# -------------------------------------------------
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

def load_winners_csv(path: str) -> List[str]:
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
    # normalize id / fid
    if "id" not in df.columns and "fid" in df.columns:
        df["id"] = df["fid"]
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)
    # require expression
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)
    if "parity_wiper" not in df.columns:
        df["parity_wiper"] = False
    if "enabled" not in df.columns:
        df["enabled"] = True
    # Ensure text types where appropriate
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["expression"] = df["expression"].astype(str)
    return df

# -------------------------------------------------
# Filter evaluation
# -------------------------------------------------
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
        return set(), 0

    eliminated = set()
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # Treat per-combo errors as non-eliminations
            pass
    return eliminated, len(eliminated)

def greedy_plan(
    candidates: pd.DataFrame,
    pool: List[str],
    base_env: Dict,
    beam_width: int,
    max_steps: int,
) -> Tuple[List[Dict], List[str]]:
    remaining = set(pool)
    chosen: List[Dict] = []

    for _ in range(max_steps):
        if not remaining:
            break

        scored = []
        for _, r in candidates.iterrows():
            elim, cnt = eval_filter_on_pool(r, list(remaining), base_env)
            if cnt > 0:
                scored.append((cnt, elim, r))
        if not scored:
            break

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

        if best_cnt == 0:
            break

    return chosen, sorted(list(remaining))

# -------------------------------------------------
# Sidebar: Mode + knobs
# -------------------------------------------------
with st.sidebar:
    st.header("Planner Mode")
    mode = st.radio(
        "Select Mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1
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
    beam_width = st.number_input("Beam width", 1, 50, value=default_beam, step=1)
    max_steps = st.number_input("Max steps", 1, 50, value=default_steps, step=1)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# -------------------------------------------------
# Seed context
# -------------------------------------------------
st.subheader("Seed / context")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (prev draw)", value="")
prev_seed = c2.text_input("Prev seed (2-back, optional)", value="")
prev_prev = c3.text_input("Prev-prev seed (3-back, optional)", value="")

st.subheader("Hot / Cold / Due digits")
c4, c5, c6 = st.columns(3)
hot_txt  = c4.text_input("Hot digits (comma/space)", value="")
cold_txt = c5.text_input("Cold digits (comma/space)", value="")
due_txt  = c6.text_input("Due digits (comma/space)", value="")
hot_digits  = [int(x) for x in parse_token_list(hot_txt) if x.isdigit()]
cold_digits = [int(x) for x in parse_token_list(cold_txt) if x.isdigit()]
due_digits  = [int(x) for x in parse_token_list(due_txt) if x.isdigit()]

# -------------------------------------------------
# Combo pool (paste OR upload)
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos as CSV text (must have a 'Result' column) OR as plain list (comma/space/newline).", height=120)
pool_file = st.file_uploader("Or upload pool CSV", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (optional, default 'Result')", value="Result")

pool: List[str] = []
if pool_text.strip():
    # Accept raw list, OR CSV text with a 'Result' column
    try:
        # Try read as CSV with 'Result'
        csv_buf = io.StringIO(pool_text)
        test_df = pd.read_csv(csv_buf)
        cols_lower = {c.lower(): c for c in test_df.columns}
        if "result" in cols_lower or pool_col_hint in test_df.columns:
            col = pool_col_hint if pool_col_hint in test_df.columns else cols_lower["result"]
            pool = [str(x).strip() for x in test_df[col].dropna().astype(str)]
        else:
            # Fallback: treat as delimited list
            pool = parse_token_list(pool_text)
    except Exception:
        pool = parse_token_list(pool_text)
elif pool_file is not None:
    try:
        pool = load_pool_from_csv(pool_file, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to load pool CSV ➜ {e}")
        st.stop()
else:
    st.info("Paste CSV combos or upload a CSV to continue.")
    st.stop()

pool = [p for p in pool if p]  # clean empties
st.caption(f"Pool size: **{len(pool)}**")

# -------------------------------------------------
# Winners history (path or leave default)
# -------------------------------------------------
st.subheader("Winners History (optional)")
history_path = st.text_input(
    "Path to winners history CSV (leave default if you like)",
    value="DC5_Midday_Full_Cleaned_Expanded.csv"
)
winners_list = load_winners_csv(history_path)  # not used to auto-HCD here, but preserved for compatibility

# -------------------------------------------------
# Filters: paste IDs and/or upload CSV (defaults to path)
# -------------------------------------------------
st.subheader("Filters")
ids_text = st.text_area(
    "Paste applicable Filter IDs (optional, comma/space/newline; e.g., 1F475, NO202F053, ...)",
    height=100
)
filters_file_up = st.file_uploader(
    "Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)",
    type=["csv"]
)
filters_csv_path_default = "lottery_filters_batch_10.csv"
filters_source = filters_file_up if filters_file_up is not None else filters_csv_path_default

try:
    filters_df_full = load_filters_csv(filters_source)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    st.stop()

applicable_ids = parse_token_list(ids_text)  # strings only, no eval
if applicable_ids:
    id_series = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_series.isin(applicable_ids)].copy()
else:
    filters_df = filters_df_full.copy()

if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]].copy()

if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True].copy()

st.caption(f"Filters loaded: **{len(filters_df)}**")

# -------------------------------------------------
# Build base env
# -------------------------------------------------
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

# -------------------------------------------------
# Run planner action
# -------------------------------------------------
run_clicked = st.button("▶ Run Planner", type="primary")

if run_clicked:
    st.subheader(f"{mode} running on {len(filters_df)} filters")

    # 1) Score filters on current pool
    scored_rows = []
    for _, r in filters_df.iterrows():
        elim_set, cnt = eval_filter_on_pool(r, pool, base_env)
        scored_rows.append({
            "id": r["id"],
            "name": r.get("name", ""),
            "expression": r["expression"],
            "elim_count_on_pool": cnt,
            "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
            "parity_wiper": bool(r.get("parity_wiper", False)),
        })
    scored_df = pd.DataFrame(scored_rows)

    if scored_df.empty:
        st.warning("No filters evaluated.")
        st.stop()

    # 2) Candidate large filters (by count threshold)
    if "elim_count_on_pool" not in scored_df.columns:
        st.error("Scoring failed: 'elim_count_on_pool' not found.")
        st.stop()

    large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
    large_df = large_df.sort_values(by=["elim_count_on_pool", "elim_pct_on_pool"], ascending=False)

    st.write(f"**Large filters (≥ {min_elims} eliminated):** {len(large_df)}")
    if not large_df.empty:
        st.dataframe(
            large_df[["id", "name", "elim_count_on_pool", "elim_pct_on_pool"]],
            use_container_width=True,
            height=300
        )
    else:
        st.info("No candidates meet the 'Large' threshold at current settings.")

    # 3) Greedy plan
    st.subheader("Planner (greedy)")
    if large_df.empty:
        plan = []
        kept_after = pool
        st.info("Nothing to plan — try lowering the 'Min eliminations' or increasing beam/depth.")
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

    # 4) Downloads
    st.subheader("Downloads")
    kept_df = pd.DataFrame({"Result": kept_after})
    removed = sorted(set(pool) - set(kept_after))
    removed_df = pd.DataFrame({"Result": removed})

    cA, cB = st.columns(2)
    cA.download_button("Download KEPT combos (CSV)", kept_df.to_csv(index=False), file_name="kept_combos.csv", mime="text/csv")
    cA.download_button("Download KEPT combos (TXT)", "\n".join(kept_after), file_name="kept_combos.txt", mime="text/plain")
    cB.download_button("Download REMOVED combos (CSV)", removed_df.to_csv(index=False), file_name="removed_combos.csv", mime="text/csv")
    cB.download_button("Download REMOVED combos (TXT)", "\n".join(removed), file_name="removed_combos.txt", mime="text/plain")
else:
    st.info("Set your inputs and click **Run Planner** to evaluate.")
