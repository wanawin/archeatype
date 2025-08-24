# archetype_lab_app.py — Large-filter safety by seed archetype (full copy/paste)
from __future__ import annotations

import re, math, itertools as it
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Defaults (you can change in UI)
# ==============================
DEFAULT_WINNERS = "DC5_Midday_Full_Cleaned_Expanded.csv"
DEFAULT_FILTERS = "lottery_filters_batch_10.csv"
DEFAULT_POOL    = "today_pool.csv"  # optional

# ===============
# Small utilities
# ===============
def _fmt_ts(p: Path) -> str:
    try:
        return pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"

def _read_csv_flex(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)

def _text_to_list(s: str) -> List[str]:
    if not s: return []
    s = s.replace("\n", ",").replace(" ", ",")
    return [t.strip() for t in s.split(",") if t.strip()]

def _clean_combo_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
        .where(lambda s: s.str.fullmatch(r"\d{5}"), np.nan)
        .dropna()
    )

# ======================
# Domain helpers / envs
# ======================
MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
VTRAC  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s)]

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def spread_band(spread: int) -> str:
    if spread <= 3: return "0-3"
    if spread <= 5: return "4-5"
    if spread <= 7: return "6-7"
    if spread <= 9: return "8-9"
    return "10+"

def classify_structure(digs: List[int]) -> str:
    from collections import Counter
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def parity_major_label(digs: List[int]) -> str:
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"

def build_env(seed: str, winner: str) -> Dict[str, object]:
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(winner))
    seed_sum   = sum(seed_list)
    combo_sum  = sum(combo_list)
    env = {
        "combo": winner,
        "combo_digits": set(combo_list),
        "combo_digits_list": combo_list,
        "combo_sum": combo_sum,
        "combo_sum_cat": sum_category(combo_sum),
        "combo_sum_category": sum_category(combo_sum),

        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": seed_sum,
        "seed_sum_category": sum_category(seed_sum),

        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),

        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),

        "mirror": MIRROR, "vtrac": VTRAC,
        "any": any, "all": all, "len": len, "sum": sum, "max": max, "min": min,
        "set": set, "sorted": sorted,
    }
    return env

# ====================
# Filter CSV handling
# ====================
@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

def load_filters(path: Path) -> Dict[str, FilterDef]:
    df = _read_csv_flex(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    req = ["id","name","enabled","applicable_if","expression"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV missing column: {r}")
    def to_bool(x):
        if isinstance(x, bool): return x
        if pd.isna(x): return False
        return str(x).strip().lower() in {"true","1","yes","y"}
    df["enabled"] = df["enabled"].map(to_bool)
    for col in ["applicable_if","expression"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        df[col] = df[col].apply(lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s)
    out = {}
    for _, r in df.iterrows():
        f = FilterDef(
            fid=str(r["id"]).strip(),
            name=str(r["name"]).strip(),
            enabled=bool(r["enabled"]),
            applicable_if=str(r["applicable_if"]).strip(),
            expression=str(r["expression"]).strip(),
        )
        out[f.fid] = f
    return out

# ===========
# Safe evals
# ===========
def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr: return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def compile_expr(expr: str, default_false=False):
    try:
        code = compile(expr if expr else ("False" if default_false else "True"), "<expr>", "eval")
    except Exception:
        code = compile("False" if default_false else "True", "<expr>", "eval")
    return code

# ======================
# Archetype definitions
# ======================
def seed_features(seed: str) -> Dict[str, object]:
    digs = digits_of(seed)
    evens = sum(d%2==0 for d in digs)
    highs = sum(d>=5 for d in digs)
    dup = (len(set(digs)) < 5)
    ssum = sum(digs)
    spread = max(digs) - min(digs)
    label = {
        "sum_cat":   sum_category(ssum),
        "spread":    spread_band(spread),
        "structure": classify_structure(digs),
        "parity":    parity_major_label(digs),
        "dup":       "dup" if dup else "no-dup",
        "hi_low":    "mixed" if (0<highs<5) else ("all-high" if highs==5 else "all-low"),
    }
    return label

def archetype_key(feats: Dict[str, object]) -> str:
    return "|".join([
        f"sum:{feats['sum_cat']}",
        f"sp:{feats['spread']}",
        f"str:{feats['structure']}",
        f"par:{feats['parity']}",
        f"dup:{feats['dup']}",
        f"hl:{feats['hi_low']}",
    ])

def archetype_name(feats: Dict[str, object]) -> str:
    return f"{feats['sum_cat']} • spread {feats['spread']} • {feats['structure']} • {feats['parity']} • {feats['dup']} • {feats['hi_low']}"

# ===========================
# Core evaluation (history)
# ===========================
def load_winners(path: Path) -> List[str]:
    df = _read_csv_flex(path)
    col = "Result" if "Result" in df.columns else None
    if col is None:
        for c in df.columns:
            vals = _clean_combo_series(df[c])
            if (vals.str.fullmatch(r"\d{5}")).all():
                col = c; break
    if col is None:
        raise ValueError("Winners CSV must have a 5-digit column (preferably named 'Result').")
    vals = _clean_combo_series(df[col])
    return vals.tolist()

def build_history_envs(winners: List[str]) -> List[Dict[str, object]]:
    """env[i] uses seed = winners[i-1], winner = winners[i] (i >= 1)."""
    envs = []
    for i in range(1, len(winners)):
        envs.append(build_env(winners[i-1], winners[i]))
    return envs

def filter_stats_for_archetype(
    filters_map: Dict[str, FilterDef],
    consider_fids: List[str],
    winners: List[str],
    feats_today: Dict[str, object]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    For each filter in consider_fids:
      - Count days with the SAME archetype (seed features match all)
      - Among those, when filter applicable_if==True, compute kept% of the winner (i.e., NOT blocked by expression)
      - Also compute overall kept% across all days as a reference
    Returns:
      main dataframe + per-feature attribution dict (each key is a single feature like 'spread', etc.)
    """
    envs = build_history_envs(winners)
    if not envs:
        return pd.DataFrame(), {}

    # Precompute seed-features and full-archetype key for each day
    seeds = [e["seed"] for e in envs]
    feat_rows = [seed_features(s) for s in seeds]
    keys = [archetype_key(f) for f in feat_rows]
    key_today = archetype_key(feats_today)

    # Which days match today's full archetype
    mask_full = np.array([k == key_today for k in keys], dtype=bool)

    # Single-feature masks for attribution
    dims = ["sum_cat","spread","structure","parity","dup","hi_low"]
    masks_feat = {
        d: np.array([fr[d] == feats_today[d] for fr in feat_rows], dtype=bool) for d in dims
    }

    # Compile filter expressions for speed
    comp = {}
    for fid in consider_fids:
        f = filters_map.get(fid)
        if not f: continue
        comp[fid] = {
            "app": compile_expr(f.applicable_if, default_false=False),
            "blk": compile_expr(f.expression, default_false=True),
            "name": f.name,
            "raw_app": f.applicable_if,
        }

    # Evaluate for full archetype
    rows = []
    per_feature = {d: [] for d in dims}

    for i, fid in enumerate(consider_fids):
        meta = comp.get(fid)
        if not meta: continue

        # FULL archetype stats
        app_n = blk_n = keep_n = 0
        for env, m in zip(envs, mask_full):
            if not m: continue
            try:
                a = bool(eval(meta["app"], {"__builtins__": {}}, env))
            except Exception:
                a = False
            if a:
                app_n += 1
                try:
                    b = bool(eval(meta["blk"], {"__builtins__": {}}, env))
                except Exception:
                    b = False
                if b:
                    blk_n += 1
                else:
                    keep_n += 1
        kept_pct_full = (keep_n / app_n * 100.0) if app_n > 0 else np.nan

        # OVERALL kept% reference
        app_all = blk_all = keep_all = 0
        for env in envs:
            try:
                a = bool(eval(meta["app"], {"__builtins__": {}}, env))
            except Exception:
                a = False
            if a:
                app_all += 1
                try:
                    b = bool(eval(meta["blk"], {"__builtins__": {}}, env))
                except Exception:
                    b = False
                if b:
                    blk_all += 1
                else:
                    keep_all += 1
        kept_pct_all = (keep_all / app_all * 100.0) if app_all > 0 else np.nan

        rows.append({
            "filter_id": fid,
            "name": meta["name"],
            "archetype_applicable_days": app_n,
            "archetype_kept_days": keep_n,
            "archetype_block_days": blk_n,
            "archetype_kept_pct": round(kept_pct_full, 3) if not np.isnan(kept_pct_full) else np.nan,
            "overall_applicable_days": app_all,
            "overall_kept_pct": round(kept_pct_all, 3) if not np.isnan(kept_pct_all) else np.nan,
            "trigger_flag": ("seed" in meta["raw_app"]) or ("spread_seed" in meta["raw_app"]) or ("seed_" in meta["raw_app"]),
        })

        # Single-feature attribution tables
        for d in dims:
            app_n_f = blk_n_f = keep_n_f = 0
            mask_d = masks_feat[d]
            for env, m in zip(envs, mask_d):
                if not m: continue
                try:
                    a = bool(eval(meta["app"], {"__builtins__": {}}, env))
                except Exception:
                    a = False
                if a:
                    app_n_f += 1
                    try:
                        b = bool(eval(meta["blk"], {"__builtins__": {}}, env))
                    except Exception:
                        b = False
                    if b:
                        blk_n_f += 1
                    else:
                        keep_n_f += 1
            kept_pct_f = (keep_n_f / app_n_f * 100.0) if app_n_f > 0 else np.nan
            per_feature[d].append({
                "filter_id": fid,
                f"{d}_applicable_days": app_n_f,
                f"{d}_kept_pct": round(kept_pct_f, 3) if not np.isnan(kept_pct_f) else np.nan,
            })

    main_df = pd.DataFrame(rows)
    feat_tables = {d: pd.DataFrame(per_feature[d]) for d in dims}
    return main_df, feat_tables

# ==============
# Pool handling
# ==============
def load_pool_from_csv(path: Path) -> List[str]:
    df = _read_csv_flex(path)
    col = "combo" if "combo" in df.columns else ("Result" if "Result" in df.columns else None)
    if col is None: raise ValueError("Pool CSV must contain 'combo' or 'Result' column.")
    return _clean_combo_series(df[col]).tolist()

def apply_filter_to_pool(f: FilterDef, env_today: Dict[str, object], pool: List[str]) -> Tuple[int, bool]:
    """Return elim_count and today_applicable flag."""
    app_code = compile_expr(f.applicable_if, default_false=False)
    blk_code = compile_expr(f.expression, default_false=True)

    # Evaluate applicability once on today's seed context (winner field is overwritten per combo)
    # but applicable_if may depend on combo too, so we re-check per combo for safety.
    eliminated = 0
    today_app = False
    for s in pool:
        clist = sorted(digits_of(s))
        env = dict(env_today)
        env.update({
            "combo": s,
            "combo_digits": set(clist),
            "combo_digits_list": clist,
            "combo_sum": sum(clist),
            "combo_sum_cat": sum_category(sum(clist)),
            "combo_sum_category": sum_category(sum(clist)),
            "spread_combo": max(clist) - min(clist),
            "combo_vtracs": set(VTRAC[d] for d in clist),
        })
        try:
            a = bool(eval(app_code, {"__builtins__": {}}, env))
        except Exception:
            a = False
        if a:
            today_app = True
            try:
                b = bool(eval(blk_code, {"__builtins__": {}}, env))
            except Exception:
                b = False
            if b:
                eliminated += 1
    return eliminated, today_app

# =================
# Streamlit UI/UX
# =================
st.set_page_config(page_title="Archetype Lab — Large Filter Safety", layout="wide")

st.title("Archetype Lab — Large Filter Safety (fact-based)")
st.caption("Paste applicable filter IDs + provide winners & pool. App computes today’s archetype, "
           "then evaluates historical kept% for each pasted filter inside that archetype, and shows "
           "today’s elimination strength on your pool.")

with st.expander("Inputs", expanded=True):
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        winners_path = st.text_input("Winners CSV path", DEFAULT_WINNERS)
        filters_path = st.text_input("Filters CSV path", DEFAULT_FILTERS)
        pool_path = st.text_input("Pool CSV path (optional)", DEFAULT_POOL)
    with c2:
        seed = st.text_input("Today Seed (5 digits, optional) – leave blank to auto-pick", "")
        prev1 = st.text_input("Prev 1 (optional)", "")
        prev2 = st.text_input("Prev 2 (optional)", "")
    with c3:
        applicable_ids_text = st.text_area("Paste applicable Filter IDs (from main app)", height=120,
                                           placeholder="e.g. 105f552, A12, FID_007 ...")
        exclude_ids_text = st.text_area("Exclude these IDs regardless (e.g., parity eliminators)", height=120)

    c4, c5, c6 = st.columns(3)
    with c4:
        pool_text = st.text_area("Or paste TODAY pool combos (one per line or comma/space separated)", height=120)
    with c5:
        large_min_elim = st.number_input("Large filter minimum eliminations (today)", min_value=0, max_value=100000, value=60, step=1)
        min_kept_pct   = st.number_input("Minimum kept% within archetype (safety)", min_value=0, max_value=100, value=75, step=1)
    with c6:
        min_app_days   = st.number_input("Minimum archetype applicable days (evidence size)", min_value=0, max_value=10000, value=20, step=1)
        show_feature_signals = st.checkbox("Show per-feature signals (why)", True)

    run_btn = st.button("Run analysis", type="primary")

if not run_btn:
    st.stop()

# ===============
# Load all inputs
# ===============
try:
    winners = load_winners(Path(winners_path))
except Exception as e:
    st.error(f"Failed to load winners: {e}")
    st.stop()

try:
    filters_map = load_filters(Path(filters_path))
except Exception as e:
    st.error(f"Failed to load filters: {e}")
    st.stop()

# Seed/prevs: default to the tail of winners if not provided
if not seed:
    if len(winners) < 2:
        st.error("Winners file must contain at least 2 rows to derive seed/winner pairs.")
        st.stop()
    seed = winners[-2]  # last seed = previous winner
if not prev1 and len(winners) >= 3:
    prev1 = winners[-3]
if not prev2 and len(winners) >= 4:
    prev2 = winners[-4]

# Build today's env scaffold (winner is dummy; combo fields overridden per pool item)
env_today = build_env(seed, seed)  # placeholder winner=seed, will be replaced per combo

# Pool
pool = []
if pool_text.strip():
    pool = _text_to_list(pool_text)
    pool = [p for p in _clean_combo_series(pd.Series(pool)).tolist()]
elif pool_path and Path(pool_path).exists():
    try:
        pool = load_pool_from_csv(Path(pool_path))
    except Exception as e:
        st.warning(f"Could not load pool from CSV: {e}")
        pool = []
else:
    st.info("No pool provided; today elimination counts will be 0 and grouping may not reflect size.")
    pool = []

# Applicable IDs
applicable_ids = _text_to_list(applicable_ids_text)
exclude_ids = set(_text_to_list(exclude_ids_text))

# Restrict to pasted IDs only, minus explicit excludes and disabled filters
consider_fids = [fid for fid in applicable_ids if fid in filters_map and filters_map[fid].enabled and fid not in exclude_ids]
if not consider_fids:
    st.error("No valid filters to consider. Check your pasted IDs and filters CSV.")
    st.stop()

# =========================
# Compute today’s archetype
# =========================
feats_today = seed_features(seed)
arch_name = archetype_name(feats_today)
st.success(f"Archetype: **{arch_name}**")
with st.expander("Archetype signature (all characteristics)", expanded=False):
    st.json(feats_today)

# ======================================
# Historical stats within this archetype
# ======================================
main_df, feat_tables = filter_stats_for_archetype(filters_map, consider_fids, winners, feats_today)
if main_df.empty:
    st.error("No history available for this archetype (or evaluation failed).")
    st.stop()

# ===============================
# Today elimination on the pool
# ===============================
elim_rows = []
for fid in consider_fids:
    f = filters_map[fid]
    elim, today_app = apply_filter_to_pool(f, env_today, pool) if pool else (0, safe_eval(f.applicable_if, env_today))
    elim_rows.append({"filter_id": fid, "elim_count_today": int(elim), "today_applicable": bool(today_app)})

elim_df = pd.DataFrame(elim_rows)

# ============================
# Merge & scoring / grouping
# ============================
df = main_df.merge(elim_df, on="filter_id", how="left")
df["elim_count_today"] = df["elim_count_today"].fillna(0).astype(int)
df["today_applicable"] = df["today_applicable"].fillna(False)

# Large filter grouping by today's elim count
def bucket(c):
    c = int(c)
    if c >= 701: return "701+"
    if c >= 501: return "501–700"
    if c >= 301: return "301–500"
    if c >= 101: return "101–300"
    if c >=  61: return "61–100"
    if c >=   1: return "1–60"
    return "0"
df["group"] = df["elim_count_today"].map(bucket)

# Recommendation flag (safety + evidence + size)
df["recommend"] = (
    (df["elim_count_today"] >= int(large_min_elim)) &
    (df["archetype_applicable_days"] >= int(min_app_days)) &
    (df["archetype_kept_pct"] >= float(min_kept_pct))
)

# Sort inside group: safest → strongest → evidence
df["__grp_rank"] = df["group"].map({"701+":0,"501–700":1,"301–500":2,"101–300":3,"61–100":4,"1–60":5,"0":6})
df = df.sort_values(
    by=["__grp_rank","recommend","archetype_kept_pct","elim_count_today","archetype_applicable_days","filter_id"],
    ascending=[True, False, False, False, False, True]
).drop(columns="__grp_rank")

# =========================
# Display & export results
# =========================
st.markdown("---")
st.subheader("Recommended large filters (by today’s archetype)")

cols_to_show = [
    "filter_id","name","group","elim_count_today","today_applicable",
    "archetype_applicable_days","archetype_kept_pct",
    "overall_applicable_days","overall_kept_pct",
    "trigger_flag","recommend"
]
st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True, height=min(700, 48 + 28*len(df)))

# Feature signals (WHY)
if show_feature_signals:
    st.markdown("#### Per-feature signals (kept% when only that feature matches today)")
    signals = []
    for dim, t in feat_tables.items():
        if t.empty: continue
        signals.append(
            df[["filter_id"]].merge(t, on="filter_id", how="left")
        )
    if signals:
        sig_df = signals[0]
        for extra in signals[1:]:
            sig_df = sig_df.merge(extra, on="filter_id", how="left")
        # keep only a few strong columns for readability
        keep_cols = ["filter_id"]
        for dim in ["sum_cat","spread","structure","parity","dup","hi_low"]:
            keep_cols += [f"{dim}_applicable_days", f"{dim}_kept_pct"]
        sig_df = df[["filter_id","name"]].merge(sig_df, on="filter_id", how="left")
        st.dataframe(sig_df[["filter_id","name"] + [c for c in keep_cols if c in sig_df.columns]],
                     use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No per-feature attribution available.")

# Download
from datetime import datetime
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_csv = Path(f"archetype_filter_safety_{ts}.csv")
df.to_csv(out_csv, index=False)
st.download_button("Download results CSV", data=out_csv.read_bytes(),
                   file_name=out_csv.name, type="primary")

# Quick recap & tips
with st.expander("How to read this", expanded=False):
    st.markdown(
        """
- **Archetype** = named bundle of seed characteristics (sum band, spread band, structure, parity-major, duplicates, high/low mix).
- **archetype_kept_pct** = among historical days with this *same archetype*, percent of times the filter **did not** block the actual winner *when the filter was applicable*. Higher = safer.
- **elim_count_today** = how many combos this filter removes from **your pasted pool** today. Higher = more efficient.
- **group** = bucketed by elim_count_today so the big hammer filters appear first (701+, 501–700, …).
- **recommend** = passes all three knobs (min eliminations, min archetype kept%, minimum applicable days).
- **trigger_flag** = heuristic: `applicable_if` mentions seed attributes; these are seed-specific clues.
- **Per-feature signals** show kept% when only one characteristic matches today (e.g., same spread band only). This explains *why* a filter looks safe.
        """
    )
