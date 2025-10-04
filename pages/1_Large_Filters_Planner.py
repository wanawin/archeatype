# pages/1_Large_Filters_Planner.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from collections import Counter

import streamlit as st

# -------------------------------------------------
# Page
# -------------------------------------------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

# -------------------------------------------------
# Core helpers & signals
# -------------------------------------------------
VTRAC: Dict[int, int] = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def digits_of(s: str) -> List[int]:
    s = str(s).strip()
    return [int(ch) for ch in s if ch.isdigit()]

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

def parse_list_any(text: str) -> List[str]:
    """Split string on commas / spaces / newlines / semicolons into tokens."""
    if not text:
        return []
    raw = (
        text.replace("\t", ",")
            .replace("\n", ",")
            .replace(";", ",")
            .replace(" ", ",")
    )
    return [p.strip() for p in raw.split(",") if p.strip()]

def seed_features(digs: List[int]) -> Tuple[str, str, str]:
    """Return (sum_category, parity, structure) for similarity."""
    if not digs:
        return "", "", ""
    s = sum(digs)
    parity = "Even" if s % 2 == 0 else "Odd"
    return sum_category(s), parity, classify_structure(digs)

def hot_cold_due(winners_digits: List[List[int]], k: int = 10) -> Tuple[Set[int], Set[int], Set[int]]:
    if not winners_digits:
        return set(), set(), set(range(10))
    hist = winners_digits[-k:] if len(winners_digits) >= k else winners_digits
    flat = [d for row in hist for d in row]
    cnt = Counter(flat)
    if not cnt:
        return set(), set(), set(range(10))
    # hot = top 6 by frequency (ties included)
    most = cnt.most_common()
    topk = 6
    thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
    hot = {d for d, c in most if c >= thresh}
    # cold = bottom 4 by frequency (ties included)
    least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
    coldk = 4
    cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
    cold = {d for d, c in least if c <= cth}
    # due = not seen in last 2
    last2 = set(d for row in winners_digits[-2:] for d in row)
    due = set(range(10)) - last2
    return hot, cold, due

# -------------------------------------------------
# Environment builders
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

    base = {
        # seed digits & relatives
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd),
        "seed_sum": sum(sd) if sd else 0,
        "prev_sum_cat": sum_category(sum(sd)) if sd else "",
        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),

        # vtrac/mirror maps
        "VTRAC": VTRAC,
        "mirror": MIRROR,

        # user-provided H/C/D (or empty)
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),

        # utilities
        "Counter": Counter,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted,

        # legacy convenience
        "seed_value": int(seed) if seed and seed.isdigit() else None,
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

def build_day_env(winners_list: List[str], i: int) -> Dict:
    """
    Environment for historical day i (i>=1):
    seed = winners[i-1], winner/combo = winners[i]
    """
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

    env = {
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': list(set(sd) - set(pdigs)),
        'seed_counts': Counter(sd),
        'seed_sum': sum(sd),
        'prev_sum_cat': sum_category(sum(sd)),

        'combo': winner,
        'combo_digits': cd,
        'combo_digits_list': cd,
        'combo_sum': sum(cd),
        'combo_sum_cat': sum_category(sum(cd)),
        'combo_sum_category': sum_category(sum(cd)),

        'seed_vtracs': set(VTRAC[d] for d in sd),
        'combo_vtracs': set(VTRAC[d] for d in cd),
        'mirror': MIRROR,
        'VTRAC': VTRAC,

        'hot_digits': sorted(hot),
        'cold_digits': sorted(cold),
        'due_digits': sorted(due),

        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted,
        'Counter': Counter,

        'seed_value': int(seed) if seed.isdigit() else None,
        'nan': float('nan'),
        'winner_structure': classify_structure(sd),
        'combo_structure': classify_structure(cd),
    }
    return env

# -------------------------------------------------
# CSV loaders (pool, winners, filters)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_winners_csv_from_path(path: str) -> List[str]:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "result" in cols_lower:
        s = df[cols_lower["result"]]
    elif "combo" in cols_lower:
        s = df[cols_lower["combo"]]
    else:
        return []
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_pool_from_text_or_csv(text: str, col_hint: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Try CSV first if it looks like CSV
    looks_csv = ("," in text and "\n" in text) or text.lower().startswith("result")
    if looks_csv:
        try:
            df = pd.read_csv(io.StringIO(text))
            cols_lower = {c.lower(): c for c in df.columns}
            if col_hint in df.columns:
                s = df[col_hint]
            elif "result" in cols_lower:
                s = df[cols_lower["result"]]
            elif "combo" in cols_lower:
                s = df[cols_lower["combo"]]
            else:
                # Fallback to first column
                s = df[df.columns[0]]
            return [str(x).strip() for x in s.dropna().astype(str)]
        except Exception:
            pass
    # Not CSV or CSV failed: treat as tokens
    return parse_list_any(text)

def load_pool_from_file(f, col_hint: str) -> List[str]:
    df = pd.read_csv(f)
    cols_lower = {c.lower(): c for c in df.columns}
    if col_hint and col_hint in df.columns:
        s = df[col_hint]
    elif "result" in cols_lower:
        s = df[cols_lower["result"]]
    elif "combo" in cols_lower:
        s = df[cols_lower["combo"]]
    else:
        s = df[df.columns[0]]
    return [str(x).strip() for x in s.dropna().astype(str)]

def normalize_filters_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # normalize id/fid
    if "id" not in out.columns and "fid" in out.columns:
        out["id"] = out["fid"]
    if "id" not in out.columns:
        out["id"] = range(1, len(out) + 1)
    # require expression
    if "expression" not in out.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "name" not in out.columns:
        out["name"] = out["id"].astype(str)
    if "parity_wiper" not in out.columns:
        out["parity_wiper"] = False
    if "enabled" not in out.columns:
        out["enabled"] = True
    # optional applicable_if
    if "applicable_if" not in out.columns:
        out["applicable_if"] = "True"
    return out

def load_filters_from_source(
    pasted_csv_text: str,
    uploaded_csv_file,
    csv_path: str,
) -> pd.DataFrame:
    # 1) pasted CSV text wins
    if pasted_csv_text and pasted_csv_text.strip():
        df = pd.read_csv(io.StringIO(pasted_csv_text))
        return normalize_filters_df(df)
    # 2) uploaded file
    if uploaded_csv_file is not None:
        df = pd.read_csv(uploaded_csv_file)
        return normalize_filters_df(df)
    # 3) path (default path allowed)
    df = pd.read_csv(csv_path)
    return normalize_filters_df(df)

# -------------------------------------------------
# Filter evaluation
# -------------------------------------------------
def eval_applicable(applicable_if: str, base_env: Dict) -> bool:
    try:
        code = compile(str(applicable_if), "<applicable_if>", "eval")
        return bool(eval(code, {"__builtins__": {}}, base_env))
    except Exception:
        # if it errors, consider applicable to avoid false negatives
        return True

def eval_filter_on_pool(
    row: pd.Series,
    pool: List[str],
    base_env: Dict,
) -> Tuple[Set[str], int]:
    """
    Return (eliminated_set, count). Expression returns True to eliminate combo.
    Honours optional 'applicable_if': if false for the base env, it still evaluates
    per combo env (legacy behavior), but you can wrap your logic there too.
    """
    expr = str(row["expression"])
    try:
        code = compile(expr, "<filter_expr>", "eval")
    except Exception:
        return set(), 0

    eliminated: Set[str] = set()
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # on per-combo error, treat as not eliminating that combo
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

    for step in range(int(max_steps)):
        if not remaining:
            break

        # re-score on current remaining pool
        scored = []
        for _, r in candidates.iterrows():
            elim, cnt = eval_filter_on_pool(r, list(remaining), base_env)
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

# -------------------------------------------------
# Sidebar knobs (modes)
# -------------------------------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode",
        ["Playlist Reducer", "Safe Filter Explorer"],
        index=1,
        help="Playlist Reducer = larger eliminations / original logic.\nSafe Filter Explorer = lower threshold, deeper search."
    )
    if mode == "Playlist Reducer":
        default_min_elims = 120
        default_beam = 5
        default_steps = 15
    else:
        default_min_elims = 60
        default_beam = 6
        default_steps = 18

    min_elims = st.number_input("Min eliminations to call it ‘Large’", 1, 99999, value=default_min_elims, step=1)
    beam_width = st.number_input("Beam width (search breadth)", 1, 50, value=default_beam, step=1)
    max_steps = st.number_input("Max steps (search depth)", 1, 50, value=default_steps, step=1)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

# -------------------------------------------------
# Seed context & Hot/Cold/Due
# -------------------------------------------------
st.subheader("Seed Context")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (prev draw)")
prev_seed = c2.text_input("Prev Seed (2-back)")
prev_prev = c3.text_input("Prev Prev Seed (3-back)")

st.subheader("Hot / Cold / Due digits (optional)")
d1, d2, d3 = st.columns(3)
hot_digits  = [int(x) for x in parse_list_any(d1.text_input("Hot digits (comma-separated)")) if x.isdigit()]
cold_digits = [int(x) for x in parse_list_any(d2.text_input("Cold digits (comma-separated)")) if x.isdigit()]
due_digits  = [int(x) for x in parse_list_any(d3.text_input("Due digits (comma-separated)")) if x.isdigit()]

# -------------------------------------------------
# Combo Pool
# -------------------------------------------------
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV with 'Result' column OR one-per-line / comma / space separated):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")

pool: List[str] = []
if pool_text.strip():
    try:
        pool = load_pool_from_text_or_csv(pool_text, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to parse pasted pool ➜ {e}")
        st.stop()
elif pool_file is not None:
    try:
        pool = load_pool_from_file(pool_file, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to load pool CSV ➜ {e}")
        st.stop()
else:
    st.info("Paste combos or upload a pool CSV to continue.")
    st.stop()

pool = [p for p in pool if p]  # clean blanks
st.write(f"**Pool size:** {len(pool)}")

# -------------------------------------------------
# Winners history CSV
# -------------------------------------------------
st.subheader("Winners History")
hc1, hc2 = st.columns([2, 1])
history_path = hc1.text_input("Path to winners history CSV (default kept)", value="DC5_Midday_Full_Cleaned_Expanded.csv")
history_upload = hc2.file_uploader("Or upload history CSV", type=["csv"], key="hist_up")

winners_list: List[str] = []
if history_upload is not None:
    try:
        winners_list = load_winners_csv_from_path(history_upload)
    except Exception as e:
        st.warning(f"Uploaded history CSV failed to read: {e}. Will try path.")
if not winners_list:
    try:
        winners_list = load_winners_csv_from_path(history_path)
    except Exception as e:
        st.warning(f"History path failed: {e}. Continuing without history safety.")

# -------------------------------------------------
# Filters: IDs + CSV (pasted / upload / path)
# -------------------------------------------------
st.subheader("Filters")
fids_text = st.text_area("Paste applicable Filter IDs (optional; comma / space / newline separated):", height=90)
filters_pasted_csv = st.text_area("Paste Filters CSV content (optional):", height=150, help="If provided, this CSV is used.")
filters_file_up = st.file_uploader("Or upload Filters CSV (used if pasted CSV is empty)", type=["csv"])
filters_csv_path = st.text_input("Or path to Filters CSV (used if pasted/upload empty)", value="lottery_filters_batch_10.csv")

try:
    filters_df_full = load_filters_from_source(filters_pasted_csv, filters_file_up, filters_csv_path)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    st.stop()

# subset by IDs (if provided)
applicable_ids = set(parse_list_any(fids_text))
if applicable_ids:
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(applicable_ids)].copy()
    if filters_df.empty:
        st.error("None of the pasted Filter IDs matched rows in the selected Filters CSV.")
        st.stop()
else:
    filters_df = filters_df_full.copy()

# optionally exclude parity wipers
if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]].copy()
# keep only enabled
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == True].copy()

st.write(f"**Filters loaded:** {len(filters_df)}")

# -------------------------------------------------
# RUN BUTTON
# -------------------------------------------------
run = st.button("▶ Run Planner", type="primary", use_container_width=False)
if not run:
    st.stop()

# -------------------------------------------------
# Build base env & evaluate
# -------------------------------------------------
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

st.subheader("Evaluating filters on current pool…")
scored_rows = []
for _, r in filters_df.iterrows():
    # skip if globally not applicable (best-effort)
    if not eval_applicable(r.get("applicable_if", "True"), base_env):
        continue

    elim_set, cnt = eval_filter_on_pool(r, pool, base_env)
    scored_rows.append({
        "id": r["id"],
        "name": r.get("name", ""),
        "expression": r["expression"],
        "elim_count_on_pool": cnt,
        "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
        "parity_wiper": bool(r.get("parity_wiper", False)),
        "_elim_set": elim_set,   # keep internally for planning
    })

scored_df = pd.DataFrame(scored_rows)
if scored_df.empty:
    st.warning("No filters evaluated (empty).")
    st.stop()

# -------------------------------------------------
# Historic safety on similar seeds (based on winners history)
# -------------------------------------------------
def similar_seed(sd_hist: List[int], sd_now: List[int]) -> bool:
    if not sd_now or not sd_hist:
        return True  # if no current seed, don't filter by similarity
    f_hist = seed_features(sd_hist)
    f_now  = seed_features(sd_now)
    match_count = sum(1 for a, b in zip(f_hist, f_now) if a == b and a != "")
    return match_count >= 2  # at least 2 of 3 features match

def safety_on_history(expr: str, winners_list: List[str], sd_now: List[int]) -> Tuple[Optional[float], int, int]:
    if not winners_list or len(winners_list) < 2:
        return None, 0, 0
    try:
        code = compile(str(expr), "<hist_expr>", "eval")
    except Exception:
        return None, 0, 0
    total_sim, bad_hits = 0, 0
    for i in range(1, len(winners_list)):
        env = build_day_env(winners_list, i)
        if not similar_seed(env["seed_digits"], sd_now):
            continue
        total_sim += 1
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):  # True means eliminate
                bad_hits += 1
        except Exception:
            # treat errors as no elimination for safety
            pass
    if total_sim == 0:
        return None, 0, 0
    safety = 100.0 * (1.0 - bad_hits / total_sim)
    return safety, total_sim, bad_hits

sd_now = digits_of(seed)
safety_cols = {"historic_safety_pct": [], "similar_days": [], "bad_hits": []}
for i, row in scored_df.iterrows():
    s_pct, sims, bad = safety_on_history(row["expression"], winners_list, sd_now)
    safety_cols["historic_safety_pct"].append(None if s_pct is None else round(s_pct, 2))
    safety_cols["similar_days"].append(sims)
    safety_cols["bad_hits"].append(bad)

scored_df = pd.concat([scored_df, pd.DataFrame(safety_cols)], axis=1)

# -------------------------------------------------
# Candidate “Large” filters by threshold
# -------------------------------------------------
large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
large_df = large_df.sort_values(
    by=["elim_count_on_pool", "historic_safety_pct", "elim_pct_on_pool"],
    ascending=[False, False, False]
)

st.write(f"**Large filters (≥ {min_elims} eliminated):** {len(large_df)}")
st.dataframe(
    large_df[["id", "name", "elim_count_on_pool", "elim_pct_on_pool", "historic_safety_pct", "similar_days", "bad_hits"]],
    use_container_width=True
)

# -------------------------------------------------
# Greedy planning (fast)
# -------------------------------------------------
st.subheader("Planner (greedy)")
if large_df.empty:
    st.info("No candidates meet the 'Large' threshold; nothing to plan.")
    kept_after = pool
    plan = []
else:
    candidates = large_df[["id", "name", "expression"]].copy()
    plan, kept_after = greedy_plan(
        candidates=candidates,
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
