
# archetype_large_filter_scan.py
from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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

def digits_of(s: str):
    return [int(ch) for ch in str(s)]

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
    env['combo_sum_category'] = env['combo_sum_cat']
    return env

def build_ctx_for_pool(seed, prev_seed, prev_prev):
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

    env = {
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
        'seed_counts': seed_counts,
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,

        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted, 'Counter': Counter
    }
    env['combo_sum_category'] = env.get('combo_sum_cat')
    return env


# -----------------------
# Filter CSV loading/compilation
# -----------------------
def load_filters_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "id" not in df.columns:
        if "fid" in df.columns:
            df["id"] = df["fid"]
        else:
            raise ValueError("Filters CSV must contain an 'id' or 'fid' column.")
    for col in ("name","applicable_if","expression"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.replace('"""','"', regex=False)
                .str.replace("'''","'", regex=False)
            )
            df[col] = df[col].apply(
                lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s
            )
    if "enabled" not in df.columns:
        df["enabled"] = True
    return df

def compile_filter(row):
    app = (row.get("applicable_if") or "True")
    expr = (row.get("expression") or "False")
    try:
        app_code = compile(app, "<app_if>", "eval")
    except Exception:
        app_code = compile("False", "<app_if>", "eval")
    try:
        expr_code = compile(expr, "<expr>", "eval")
    except Exception:
        expr_code = compile("False", "<expr>", "eval")
    return app_code, expr_code

def is_seed_specific(text: str) -> bool:
    if not text: return False
    pattern = r"\b(seed_digits|prev_seed_digits|prev_prev_seed_digits|seed_vtracs|seed_counts|prev_pattern|prev_sum_cat|new_seed_digits)\b"
    return bool(re.search(pattern, text))

# -----------------------
# Evaluation helpers
# -----------------------
def eval_app_expr(app_code, expr_code, env):
    try:
        applicable = bool(eval(app_code, {"__builtins__": {}}, env))
    except Exception:
        applicable = False
    if not applicable:
        return False, False
    try:
        eliminate = bool(eval(expr_code, {"__builtins__": {}}, env))
    except Exception:
        eliminate = False
    return applicable, eliminate

def elim_set_on_pool(app_code, expr_code, base_env, pool_digits):
    """Return the set of pool indices eliminated by this filter (respecting applicability)."""
    eliminated_idx = set()
    elim_even = elim_odd = 0
    for idx, cd in enumerate(pool_digits):
        env = dict(base_env)
        env.update({
            "combo": "".join(map(str, cd)),
            "combo_digits": sorted(cd),
            "combo_digits_list": sorted(cd),
            "combo_sum": sum(cd),
            "combo_sum_cat": sum_category(sum(cd)),
            "combo_sum_category": sum_category(sum(cd)),
            "combo_vtracs": set(VTRAC[d] for d in cd),
        })
        applicable, eliminate = eval_app_expr(app_code, expr_code, env)
        if applicable and eliminate:
            eliminated_idx.add(idx)
            if (sum(cd) % 2) == 0: elim_even += 1
            else: elim_odd += 1
    return eliminated_idx, elim_even, elim_odd

def hist_safety(app_code, expr_code, winners):
    """Over history: when applicable, did it keep the true winner?"""
    app_days = 0; blocked_days = 0
    if not winners or len(winners) < 2:
        return 0, 0, None, None
    for i in range(1, len(winners)):
        env = build_day_env(winners, i)
        applicable, eliminate = eval_app_expr(app_code, expr_code, env)
        if applicable:
            app_days += 1
            if eliminate:
                blocked_days += 1
    if app_days == 0:
        return 0, 0, None, None
    blocked_rate = blocked_days / app_days
    kept_rate = 1.0 - blocked_rate
    return app_days, blocked_days, kept_rate, blocked_rate

# -----------------------
# Planners
# -----------------------
def winner_preserving_plan(
    large_df, E_map, names_map, pool_len, winner_idx,
    target_max=45, beam_width=3, max_steps=12
):
    """
    Returns dict with keys:
      'steps': list of dicts {filter_id, name, eliminated_now, remaining_after}
      'final_pool_idx': set of indices kept
    or None if no progress possible.
    """
    P0 = set(range(pool_len))
    if winner_idx is not None and winner_idx not in P0:
        return None

    large_index = large_df.set_index("filter_id")

    best = {"pool": P0, "steps": []}
    seen = {}

    def score_candidate(fid, elim_now):
        kept = large_index.loc[fid]["hist_kept_rate"]
        days = large_index.loc[fid]["hist_applicable_days"]
        kept01 = (float(kept)/100.0) if pd.notna(kept) else 0.5
        days = float(days) if pd.notna(days) else 0.0
        return elim_now * (0.5 + 0.5*kept01) * math.log1p(days)

    def key(P, used):
        return (len(P), tuple(sorted(used))[:8])

    def dfs(P, used, log, depth):
        nonlocal best
        if winner_idx is not None and winner_idx not in P:
            return
        if len(P) <= target_max:
            if len(P) < len(best["pool"]) or (len(P) == len(best["pool"]) and len(log) < len(best["steps"])):
                best = {"pool": set(P), "steps": list(log)}
            return
        if depth >= max_steps:
            if len(P) < len(best["pool"]):
                best = {"pool": set(P), "steps": list(log)}
            return

        k = key(P, used)
        if k in seen and seen[k] <= len(P):
            return
        seen[k] = len(P)

        cands = []
        for fid, E in E_map.items():
            if fid in used: continue
            elim_now = len(P & E)
            if elim_now <= 0: continue
            if winner_idx is not None and winner_idx in E: continue
            cands.append((fid, elim_now, score_candidate(fid, elim_now)))

        if not cands:
            if len(P) < len(best["pool"]):
                best = {"pool": set(P), "steps": list(log)}
            return

        cands.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for fid, elim_now, _sc in cands[:beam_width]:
            newP = P - E_map[fid]
            step = {
                "filter_id": fid,
                "name": names_map.get(fid, ""),
                "eliminated_now": elim_now,
                "remaining_after": len(newP)
            }
            dfs(newP, used | {fid}, log + [step], depth+1)

    dfs(P0, set(), [], 0)
    if best["steps"] and len(best["pool"]) < len(P0):
        return {"steps": best["steps"], "final_pool_idx": best["pool"]}
    return None

def best_case_plan_no_winner(
    large_df, E_map, names_map, pool_len, target_max,
    exp_safety_map: Dict[str, float]
):
    """
    Greedy plan (no known winner). At each step choose the unused Large filter
    that maximizes score = eliminations_now * (0.25 + 0.75 * expected_safety).
    expected_safety is blended from hist kept-rate and archetype lift for the
    current seed profile (if available).
    """
    P = set(range(pool_len))
    used: Set[str] = set()
    steps = []
    while len(P) > target_max:
        best = None
        best_score = -1.0
        for fid, E in E_map.items():
            if fid in used: 
                continue
            elim_now = len(P & E)
            if elim_now <= 0:
                continue
            w = float(exp_safety_map.get(fid, 0.5))
            # favor safety, but still care about eliminations
            score = elim_now * (0.25 + 0.75 * w)
            if score > best_score:
                best_score = score
                best = (fid, elim_now, w)
        if best is None:
            break
        fid, elim_now, w = best
        P = P - E_map[fid]
        used.add(fid)
        steps.append({
            "filter_id": fid,
            "name": names_map.get(fid, ""),
            "eliminated_now": elim_now,
            "remaining_after": len(P),
            "expected_safety_%": round(100.0 * w, 2)
        })
    if steps:
        return {"steps": steps, "final_pool_idx": P}
    return None

# -----------------------
# Archetype lift loader
# -----------------------
def _first_col(df: pd.DataFrame, cand: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for x in cand:
        if x.lower() in lc:
            return lc[x.lower()]
    return None

def load_archetype_dimension_lifts(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # try to normalize column names we’ll use
        fid = _first_col(df, ["filter_id","fid","id"])
        dim = _first_col(df, ["dimension","dim","feature","trait_name"])
        val = _first_col(df, ["value","trait","bucket","bin"])
        lift = None
        for c in df.columns:
            if "lift" in c.lower():
                lift = c
                break
        kept = _first_col(df, ["kept_rate","kept%","kept_pct"])
        base = _first_col(df, ["baseline_kept_rate","baseline_kept%","baseline_pct"])
        supp = _first_col(df, ["applicable_days","support","n","days","app_days"])

        if not (fid and dim and val):
            return None
        if not lift:
            # synth lift if possible
            if kept and base:
                df["__lift_tmp__"] = pd.to_numeric(df[kept], errors="coerce") / pd.to_numeric(df[base], errors="coerce")
                lift = "__lift_tmp__"
            else:
                return None

        use_cols = [fid, dim, val, lift]
        if supp: use_cols.append(supp)
        out = df[use_cols].dropna().copy()
        out.columns = ["filter_id","dimension","value","lift"] + (["support"] if supp else [])
        # sanitize
        out["filter_id"] = out["filter_id"].astype(str).str.strip()
        out["dimension"] = out["dimension"].astype(str).str.strip().str.lower()
        out["value"] = out["value"].astype(str).str.strip()
        out["lift"] = pd.to_numeric(out["lift"], errors="coerce")
        out = out.dropna(subset=["lift"])
        return out
    except Exception:
        return None

def current_traits_for_match(prof: Dict[str, object]) -> Dict[str, List[str]]:
    # Provide multiple synonyms to increase match chance with analyzer outputs
    traits = {
        "sum_cat": [str(prof["sum_cat"])],
        "sum_category": [str(prof["sum_cat"])],
        "parity": [str(prof["parity"])],                     # e.g., "3E/2O"
        "parity_major": ["even>=3" if prof["parity"] and prof["parity"][0].isdigit() and int(prof["parity"][0])>=3 else "even<=2"],
        "structure": [str(prof["structure"])],
        "spread_band": [str(prof["spread_band"])],
        "spread": [str(prof["spread_band"])],
        "hi_lo": [str(prof["hi_lo"])],
        "has_dupe": ["True" if prof["has_dupe"] else "False", "Yes" if prof["has_dupe"] else "No"],
    }
    return traits

def compute_expected_safety_map(
    large_df: pd.DataFrame,
    arch_df: Optional[pd.DataFrame],
    prof: Dict[str, object]
) -> Dict[str, float]:
    """
    Returns fid -> expected safety in [0,1].
    expected = clamp( base_hist_kept * geom_mean(clamped_lifts for matching traits) )
    """
    # base kept-rate from history
    base_map = {}
    for _, r in large_df.iterrows():
        fid = str(r["filter_id"])
        kept = r.get("hist_kept_rate")
        base_map[fid] = float(kept)/100.0 if pd.notna(kept) else 0.5

    if arch_df is None or arch_df.empty:
        return {fid: max(0.05, min(0.99, base_map.get(fid, 0.5))) for fid in base_map}

    trait_dict = current_traits_for_match(prof)  # dim -> list of acceptable values
    # pre-index archetype table by filter
    by_f = {fid: sub for fid, sub in arch_df.groupby(arch_df["filter_id"].astype(str))}
    out = {}
    for fid, base in base_map.items():
        sub = by_f.get(str(fid))
        if sub is None or sub.empty:
            out[fid] = max(0.05, min(0.99, base))
            continue
        lifts = []
        for _, row in sub.iterrows():
            dim = str(row["dimension"]).lower()
            val = str(row["value"])
            if dim in trait_dict and any(val == v for v in trait_dict[dim]):
                L = float(row["lift"])
                # clamp lift to temper extremes
                L = max(0.7, min(1.3, L))
                lifts.append(L)
        if lifts:
            gm = math.exp(sum(math.log(x) for x in lifts) / len(lifts))
            expected = base * gm
        else:
            expected = base
        out[fid] = max(0.05, min(0.99, expected))
    return out

# -----------------------
# UI (Sidebar + Results)
# -----------------------
st.set_page_config(page_title="Archetype — Large Filters Planner (Winner-Preserving)", layout="wide")
st.title("Archetype Helper — Large Filters, Triggers & Plans")

with st.sidebar:
    st.header("Inputs")
    seed = st.text_input("Seed (1-back, 5 digits) *", value="", max_chars=5).strip()
    prev_seed = st.text_input("Prev seed (2-back, optional)", value="", max_chars=5).strip()
    prev_prev = st.text_input("Prev-prev seed (3-back, optional)", value="", max_chars=5).strip()

    st.markdown("---")
    filters_path_str = st.text_input("Filters CSV path", value="lottery_filters_batch_10.csv")
    winners_path_str = st.text_input("Winners CSV (optional for history)", value="DC5_Midday_Full_Cleaned_Expanded.csv")
    pool_path_str = st.text_input("Pool CSV path", value="today_pool.csv")
    pool_col_hint = st.text_input("Pool column (blank → auto)", value="")

    

# Optional paste inputs (do not clear on mode change)
st.markdown("---")
st.subheader("Optional Paste Inputs")
pool_text = st.text_area(
    "Paste combo pool here (CSV or space/comma separated; DevTools paste OK)",
    value="",
    height=120,
    help="If provided, overrides Pool CSV path. Example: 35579,35666 35667\nOr one per line."
)
filters_csv_text = st.text_area(
    "Paste Filters CSV content (optional)",
    value="",
    height=120,
    help="If provided, overrides Filters CSV path."
)
st.markdown("---")
    ids_text = st.text_area("Paste Applicable Filter IDs", height=140,
                            help="Comma/space/newline separated — the app will consider ONLY these.")

    st.markdown("---")

# ---- Mode selection (pre-fills defaults; does not clear inputs) ----
mode = st.sidebar.radio(
    "Mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    help=(
        "Playlist Reducer = original loser-list workflow (strict big-cuts, but wider/deeper search).\n"
        "Safe Filter Explorer = lower elimination threshold to surface more candidates."
    )
)
if mode == "Playlist Reducer":
    default_min_elims = 200
    default_beam = 5
    default_steps = 15
else:  # Safe Filter Explorer
    default_min_elims = 60
    default_beam = 6
    default_steps = 18
    st.subheader("Large Filter Rules")
    \1default_min_elims\2
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True,
                                 help="Parity-wiper = wipes all evens or all odds in your pool")

    st.markdown("---")
    st.subheader("Best-case Planner (no known winner)")
    use_archetype_lifts = st.checkbox("Use archetype-lift CSV if present", value=True)
    target_max_bc = st.number_input("Target kept combos (best-case)", min_value=5, max_value=200, value=45, step=1)

    st.markdown("---")
    st.subheader("Winner-preserving Planner (for backtests)")
    known_winner = st.text_input("Known winner (5 digits)", value="", max_chars=5).strip()
    target_max_wp = st.number_input("Target kept combos (winner-preserving)", min_value=5, max_value=200, value=45, step=1)
    \1default_beam\2
    \1default_steps\2

    st.markdown("---")
    run_btn = st.button("Run analysis & plans", type="primary", use_container_width=True)

# --------- Validate seed & show seed profile panel (TOP) ---------
if not (seed.isdigit() and len(seed) == 5):
    st.error("Seed must be exactly 5 digits.")
    st.stop()

prof = seed_profile(seed, prev_seed, prev_prev)

st.markdown("### Seed Profile • Archetype")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Signature", prof["signature"])
    st.caption("structure • parity • sum band • spread band • hi/lo")
with c2:
    st.metric("Sum", f"{prof['sum']} ({prof['sum_cat']})")
    st.metric("Parity", prof["parity"])
with c3:
    st.metric("Spread", f"{prof['spread']} ({prof['spread_band']})")
    st.metric("VTRAC groups", f"{prof['vtrac_diversity']}")
with c4:
    st.metric("Duplicates", "Yes" if prof["has_dupe"] else "No")
    st.metric("Carry/New vs prev", f"{prof['carry_from_prev']}/{prof['new_vs_prev']}")

st.markdown("---")

# -------------- Guard: require Run for heavy work --------------
if "results_cache" not in st.session_state:
    st.session_state.results_cache = None

if not run_btn and st.session_state.results_cache is None:
    st.info("Use the **sidebar** to set inputs, then click **Run analysis & plans**.")
    st.stop()

# ---------- Pipeline ----------
def compute_everything():
    # Load filters CSV (path OR pasted CSV)
if filters_csv_text.strip():
    from io import StringIO
    df_filters = load_filters_csv(StringIO(filters_csv_text))
else:
    filters_path = Path(filters_path_str)
    if not filters_path.exists():
        return {"error": f"Filters CSV not found: {filters_path}"}
    df_filters = load_filters_csv(filters_path)


    # Limit to pasted IDs
    ids = [x.strip() for x in re.split(r"[,\s]+", ids_text) if x.strip()]
    if not ids:
        return {"error": "Paste at least one applicable filter ID in the sidebar."}
    df_filters = df_filters[df_filters["id"].astype(str).isin(set(ids))].copy()
    if df_filters.empty:
        return {"error": "None of the pasted IDs were found in the filters CSV."}

    # Compile filters
    compiled = {}
    names_map = {}
    for _, r in df_filters.iterrows():
        fid = str(r["id"]).strip()
        names_map[fid] = str(r.get("name","")).strip()
        compiled[fid] = compile_filter(r)

    # Load pool (path OR pasted combos)
if pool_text.strip():
    raw = pool_text.replace("
", ",").replace(" ", ",")
    combos = [x.strip() for x in raw.split(",") if x.strip()]
    df_pool = pd.DataFrame({"Result": combos})
    pool_col = "Result"
else:
    pool_path = Path(pool_path_str)
    if not pool_path.exists():
        return {"error": f"Pool CSV not found: {pool_path}"}
    df_pool = pd.read_csv(pool_path)
    pool_col = None
    for cand in ([pool_col_hint] if pool_col_hint else []) + ["combo","Combo","result","Result"]:
        if cand and cand in df_pool.columns:
            pool_col = cand
            break
    if not pool_col:
        return {"error": "Could not find pool column. Provide 'Pool column' hint or ensure CSV has one of: combo, Combo, result, Result"}


    pool_series = df_pool[pool_col].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
    pool_series = pool_series[pool_series.str.fullmatch(r"\d{5}")]
    pool_digits = [digits_of(s) for s in pool_series.tolist()]
    if len(pool_digits) == 0:
        return {"error": "Pool has no valid 5-digit combos after cleaning."}

    # Even/odd counts for parity check
    total_even = sum(1 for cd in pool_digits if (sum(cd) % 2) == 0)
    total_odd  = len(pool_digits) - total_even

    # Build base env for pool evaluation
    base_env = build_ctx_for_pool(seed, prev_seed, prev_prev)

    # Optional: winners (for historical safety)
    winners_list = None
    if winners_path_str:
        wp = Path(winners_path_str)
        if wp.exists():
            wdf = pd.read_csv(wp)
            wcol = None
            if "Result" in wdf.columns: wcol = "Result"
            else:
                for c in wdf.columns:
                    tmp = wdf[c].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
                    if tmp.str.fullmatch(r"\d{5}").all():
                        wcol = c; break
            if wcol:
                winners_list = wdf[wcol].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
                winners_list = winners_list[winners_list.str.fullmatch(r"\d{5}")].tolist()

    # -----------------------
    # Scan the pasted filters
    # -----------------------
    records = []
    E_map = {}  # filter_id -> set(indices eliminated on initial pool)
    for _, r in df_filters.iterrows():
        fid = str(r["id"]).strip()
        name = names_map.get(fid, "")
        app_code, expr_code = compiled[fid]

        E, elim_even, elim_odd = elim_set_on_pool(app_code, expr_code, base_env, pool_digits)
        elim_count = len(E)
        parity_wiper = ((elim_even == total_even and total_even > 0) or (elim_odd == total_odd and total_odd > 0))
        E_map[fid] = E

        text_blob = f"{r.get('applicable_if','')} || {r.get('expression','')}"
        seed_specific = is_seed_specific(text_blob)

        app_days = blocked = None
        kept_rate = blocked_rate = None
        if winners_list and len(winners_list) >= 2:
            app_days, blocked, kept_rate, blocked_rate = hist_safety(app_code, expr_code, winners_list)

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
    group_order = ["701+","501–700","301–500","101–300","61–100","1–60","0"]
    rank_map = {g:i for i,g in enumerate(group_order)}
    large_df["__g"] = large_df["aggression_group"].map(rank_map).fillna(999).astype(int)

    large_df = large_df.sort_values(
        by=["__g","hist_kept_rate","hist_applicable_days","elim_count_on_pool","filter_id"],
        ascending=[True, False, False, False, True]
    ).drop(columns="__g")

    trig_df = df[df["seed_specific_trigger"]].copy().sort_values(by=["filter_id"])

    # -----------------------
    # Expected-safety weights (archetype lifts)
    # -----------------------
    arch_df = None
    if use_archetype_lifts:
        arch_df = load_archetype_dimension_lifts(Path("archetype_filter_dimension_stats.csv"))

    exp_safety_map = compute_expected_safety_map(large_df, arch_df, prof)

    # -----------------------
    # Plans
    # -----------------------
    plan_best = None
    if not large_df.empty:
        E_sub = {fid:E_map[fid] for fid in large_df["filter_id"].astype(str) if fid in E_map}
        plan_best = best_case_plan_no_winner(
            large_df=large_df,
            E_map=E_sub,
            names_map={fid: df_filters.set_index("id").loc[int(fid) if fid.isdigit() else fid].get("name","") if fid in E_sub else "" for fid in E_sub},
            pool_len=len(pool_digits),
            target_max=int(target_max_bc),
            exp_safety_map=exp_safety_map
        )

    plan_wp = None
    winner_idx = None
    if known_winner and known_winner.isdigit() and len(known_winner) == 5 and not large_df.empty:
        try:
            winner_idx = pool_series.tolist().index(known_winner)
        except ValueError:
            winner_idx = None
        if winner_idx is not None:
            E_sub = {fid:E_map[fid] for fid in large_df["filter_id"].astype(str) if fid in E_map}
            names_map = {fid: df_filters.set_index("id").loc[int(fid) if fid.isdigit() else fid].get("name","") if fid in E_sub else "" for fid in E_sub}
            plan_wp = winner_preserving_plan(
                large_df=large_df,
                E_map=E_sub,
                names_map=names_map,
                pool_len=len(pool_digits),
                winner_idx=winner_idx,
                target_max=int(target_max_wp),
                beam_width=int(beam_width),
                max_steps=int(max_steps)
            )

    return {
        "df": df,
        "large_df": large_df,
        "trig_df": trig_df,
        "group_order": group_order,
        "pool_series": pool_series,
        "plan_best": plan_best,
        "plan_wp": plan_wp,
        "exp_safety_map": exp_safety_map,
        "error": None,
    }

# Compute or reuse cached results
if run_btn or st.session_state.results_cache is None:
    results = compute_everything()
    st.session_state.results_cache = results
else:
    results = st.session_state.results_cache

if results.get("error"):
    st.error(results["error"])
    st.stop()

df = results["df"]
large_df = results["large_df"]
trig_df = results["trig_df"]
group_order = results["group_order"]
pool_series = results["pool_series"]
plan_best = results["plan_best"]
plan_wp = results["plan_wp"]
exp_safety_map = results["exp_safety_map"]

# -----------------------
# Show + Downloads (Large / Triggers)
# -----------------------
st.subheader("Large Filters (bucketed by eliminations on *your* pool)")
if large_df.empty:
    st.info("No large filters matched your rules among the pasted IDs for this pool.")
else:
    for g in group_order:
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

    # CSV
    large_csv = "large_filters_detected.csv"
    large_df.to_csv(large_csv, index=False)
    st.download_button("Download ALL large filters (CSV)",
                       data=Path(large_csv).read_bytes(),
                       file_name=large_csv, mime="text/csv")

    # TXT with group delineations
    lines = []
    for g in group_order:
        sub = large_df[large_df["aggression_group"] == g]
        if sub.empty: continue
        lines.append(f"===== GROUP {g} =====")
        for _, rr in sub.iterrows():
            lines.append(f"{rr['filter_id']}  | kept%={rr.get('hist_kept_rate')}  | app_days={rr.get('hist_applicable_days')}  | elim={rr['elim_count_on_pool']}")
    st.download_button("Download group delineations (TXT)",
                       data=io.BytesIO(("\n".join(lines)).encode("utf-8")),
                       file_name="large_filter_groups.txt", mime="text/plain")

st.subheader("Seed-Specific Trigger Filters (by ID)")
if trig_df.empty:
    st.info("No seed-specific trigger filters detected among the pasted IDs.")
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
    st.download_button("Download trigger filters (CSV)", data=Path(trig_csv).read_bytes(),
                       file_name=trig_csv, mime="text/csv")

st.markdown("---")

# -----------------------
# Best-case planner (no known winner)
# -----------------------
st.subheader(f"Best-case plan — Large filters only (aim: ≤ {int(st.session_state.get('results_cache', {}).get('plan_best', {}) and st.session_state.results_cache['plan_best'] and 0) or int(st.session_state.get('target_max_bc', 45))} combos)")
if plan_best is None:
    st.info("Best-case plan could not reduce the pool to your target with available Large filters, or no Large filters were detected.")
else:
    steps = plan_best["steps"]
    bc_df = pd.DataFrame(steps)
    st.dataframe(bc_df, use_container_width=True, hide_index=True, height=min(320, 60 + 28*len(bc_df)))

    # Downloads: plan CSV + TXT
    bc_csv = "best_case_plan.csv"
    bc_txt = "best_case_plan.txt"
    bc_df.to_csv(bc_csv, index=False)
    with open(bc_txt, "w", encoding="utf-8") as fh:
        for i, row in enumerate(steps, 1):
            fh.write(f"Step {i}: {row['filter_id']}  {row['name']}  | eliminated={row['eliminated_now']}  | remaining={row['remaining_after']}  | expected_safety%={row['expected_safety_%']}\n")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download best-case plan (CSV)", data=Path(bc_csv).read_bytes(),
                           file_name=bc_csv, mime="text/csv")
    with c2:
        st.download_button("Download best-case plan (TXT)", data=Path(bc_txt).read_bytes(),
                           file_name=bc_txt, mime="text/plain")

    # Final kept pool (indices)
    kept_idx = sorted(list(plan_best["final_pool_idx"]))
    kept_combos = [pool_series.iloc[i] for i in kept_idx]
    kept_df = pd.DataFrame({"combo": kept_combos})
    st.caption(f"Best-case final kept pool size: {len(kept_combos)}")
    st.dataframe(kept_df.head(100), use_container_width=True, hide_index=True, height=260)
    kept_csv = "best_case_final_kept_pool.csv"
    kept_txt = "best_case_final_kept_pool.txt"
    kept_df.to_csv(kept_csv, index=False)
    with open(kept_txt, "w", encoding="utf-8") as fh:
        for s in kept_combos:
            fh.write(f"{s}\n")
    c3, c4 = st.columns(2)
    with c3:
        st.download_button("Download best-case kept pool (CSV)", data=Path(kept_csv).read_bytes(),
                           file_name=kept_csv, mime="text/csv")
    with c4:
        st.download_button("Download best-case kept pool (TXT)", data=Path(kept_txt).read_bytes(),
                           file_name=kept_txt, mime="text/plain")

st.markdown("---")

# -----------------------
# Winner-preserving planner (optional)
# -----------------------
st.subheader(f"Winner-preserving plan — Large filters only (aim: ≤ {int(target_max_wp)} combos)")
if not known_winner:
    st.info("Provide a 5-digit **Known winner** in the sidebar to compute a winner-preserving plan (use backtests).")
else:
    if plan_wp is None:
        st.info("No winner-preserving reduction possible with the available Large filters (or winner not in pool).")
    else:
        steps = plan_wp["steps"]
        plan_df = pd.DataFrame(steps)
        st.dataframe(plan_df, use_container_width=True, hide_index=True, height=min(320, 60 + 28*len(plan_df)))

        # Downloads: plan CSV + TXT
        plan_csv = "winner_preserving_plan.csv"
        plan_txt = "winner_preserving_plan.txt"
        plan_df.to_csv(plan_csv, index=False)
        with open(plan_txt, "w", encoding="utf-8") as fh:
            for i, row in enumerate(steps, 1):
                fh.write(f"Step {i}: {row['filter_id']}  {row['name']}  | eliminated={row['eliminated_now']}  | remaining={row['remaining_after']}\n")

        # Final kept pool (indices)
        kept_idx = sorted(list(plan_wp["final_pool_idx"]))
        kept_combos = [pool_series.iloc[i] for i in kept_idx]
        kept_df = pd.DataFrame({"combo": kept_combos})
        st.caption(f"Winner-preserving final kept pool size: {len(kept_combos)}")
        st.dataframe(kept_df.head(100), use_container_width=True, hide_index=True, height=260)
        kept_csv = "winner_preserving_final_kept_pool.csv"
        kept_txt = "winner_preserving_final_kept_pool.txt"
        kept_df.to_csv(kept_csv, index=False)
        with open(kept_txt, "w", encoding="utf-8") as fh:
            for s in kept_combos:
                fh.write(f"{s}\n")
        c3, c4 = st.columns(2)
        with c3:
            st.download_button("Download winner-preserving kept pool (CSV)", data=Path(kept_csv).read_bytes(),
                               file_name=kept_csv, mime="text/csv")
        with c4:
            st.download_button("Download winner-preserving kept pool (TXT)", data=Path(kept_txt).read_bytes(),
                               file_name=kept_txt, mime="text/plain")

# ---------------
# Column guide
# ---------------
st.markdown("---")
with st.expander("Column guide", expanded=False):
    st.markdown("""
- **Seed Profile / Archetype** – Snapshot: structure • parity • sum band • spread band • hi/lo mix; plus duplicates, carry/new vs prev, VTRAC diversity.
- **elim_count_on_pool** – Eliminations on the pool you provided (**your aggression metric**).
- **elim_even / elim_odd** – Split of eliminations; helps spot parity-wipers.
- **parity_wiper** – True if a filter wipes *all* evens or *all* odds in your pool (optionally excluded).
- **seed_specific_trigger** – Filter references seed/prev-seed signals (seed_digits, seed_vtracs, prev_pattern, etc.).
- **hist_applicable_days** – Across history, days the filter was applicable.
- **hist_kept_rate** – When applicable, % of days the filter **kept** the true winner (higher = safer historically).
- **aggression_group** – Bucket by eliminations on the pool: 701+, 501–700, 301–500, 101–300, 61–100, 1–60, 0.
- **Best-case plan** – Greedy order of **Large** filters to reach ≤ target using expected safety = hist kept-rate × archetype-lift (if CSV found).
- **Winner-preserving plan** – Beam-search order of **Large** filters that keeps a known winner (for backtests).
""")
