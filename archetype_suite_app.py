# archetype_suite_app.py — one app to build archetype stats + plan large filters
from __future__ import annotations

import io, re, math, os, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# Must exist alongside this file
# (we keep it separate so you can iterate on the analyzer independently)
from archetype_safety import analyze_archetype_safety

# ---------- defaults (you can change in the sidebar) ----------
WINNERS_CSV_DEFAULT = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV_DEFAULT = "lottery_filters_batch_10.csv"
POOL_CSV_DEFAULT    = "today_pool.csv"

# Expected analyzer outputs (we'll auto-create them if missing/stale)
OUT_COMPOSITE = Path("archetype_filter_composite_stats.csv")
OUT_DIMS      = Path("archetype_filter_dimension_stats.csv")
OUT_TOP       = Path("archetype_filter_top_signals.csv")
OUT_DANGER    = Path("archetype_filter_danger_signals.csv")

# ---------- small helpers ----------
def _download_btn(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        return
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(label, buff.getvalue(), file_name=filename, mime="text/csv")

def _read_csv_if_exists(p: Path) -> Optional[pd.DataFrame]:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
    return None

def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0

def files_stale(outputs: List[Path], deps: List[Path]) -> bool:
    """True if any output missing OR older than any dependency."""
    dep_mtime = max((_mtime(p) for p in deps if p.exists()), default=0.0)
    for out in outputs:
        if not out.exists():
            return True
        if _mtime(out) < dep_mtime:
            return True
    return False

# =========================
# Archetype seed utilities
# =========================
VTRAC  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s)]

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def structure_label(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def parity_major_label(digs: List[int]) -> str:
    return "even≥3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even≤2"

def seed_badge(seed: str, prev_seed: str, prev_prev: str) -> Dict[str, str]:
    sd = digits_of(seed)
    hi = sum(1 for d in sd if d >= 5)
    lo = 5 - hi
    even = sum(1 for d in sd if d % 2 == 0)
    odd  = 5 - even
    badge = {
        "Seed": seed,
        "Sum": f"{sum(sd)} ({sum_category(sum(sd))})",
        "Spread": str(max(sd) - min(sd)),
        "Structure": structure_label(sd),
        "High/Low": f"{hi}/{lo}",
        "Even/Odd": f"{even}/{odd} ({parity_major_label(sd)})",
        "V-tracs": ",".join(sorted({str(VTRAC[d]) for d in sd})),
    }
    if prev_seed:
        badge["1-back"] = prev_seed
    if prev_prev:
        badge["2-back"] = prev_prev
    return badge

# =========================
# Filter CSV helpers
# =========================
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
            df[col] = df[col].astype(str).str.strip() \
                .str.replace('"""','"', regex=False) \
                .str.replace("'''","'", regex=False)
            df[col] = df[col].apply(lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s)
    if "enabled" not in df.columns:
        df["enabled"] = True
    return df

def compile_filter(row):
    app = (row.get("applicable_if") or "True")
    expr = (row.get("expression") or "False")
    try:    app_code = compile(app, "<app_if>", "eval")
    except: app_code = compile("False", "<app_if>", "eval")
    try:    expr_code = compile(expr, "<expr>", "eval")
    except: expr_code = compile("False", "<expr>", "eval")
    return app_code, expr_code

def is_seed_specific(text: str) -> bool:
    if not text: return False
    pattern = r"\b(seed_digits|prev_seed_digits|prev_prev_seed_digits|seed_vtracs|seed_counts|prev_pattern|prev_sum_cat|new_seed_digits)\b"
    return bool(re.search(pattern, text))

# =========================
# Pool/env + evaluation
# =========================
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
        'seed_sum': sum(sd),
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted
    }
    return base

def eval_app_expr(app_code, expr_code, env):
    try: applicable = bool(eval(app_code, {"__builtins__": {}}, env))
    except Exception: applicable = False
    if not applicable:
        return False, False
    try: eliminate = bool(eval(expr_code, {"__builtins__": {}}, env))
    except Exception: eliminate = False
    return applicable, eliminate

def elim_set_on_pool(app_code, expr_code, base_env, pool_digits):
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
    app_days = 0; blocked_days = 0
    if not winners or len(winners) < 2:
        return 0, 0, None, None
    for i in range(1, len(winners)):
        seed = winners[i-1]; combo = winners[i]
        env = build_ctx_for_pool(seed, winners[i-2] if i-2>=0 else "", winners[i-3] if i-3>=0 else "")
        env.update({
            "combo": combo,
            "combo_digits": sorted(digits_of(combo)),
            "combo_digits_list": sorted(digits_of(combo)),
            "combo_sum": sum(digits_of(combo)),
            "combo_sum_cat": sum_category(sum(digits_of(combo))),
            "combo_sum_category": sum_category(sum(digits_of(combo))),
            "combo_vtracs": set(VTRAC[d] for d in digits_of(combo)),
        })
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

def winner_preserving_plan(large_df, E_map, names_map, pool_len, winner_idx,
                           target_max=45, beam_width=3, max_steps=12):
    P0 = set(range(pool_len))
    if winner_idx is not None and winner_idx not in P0:
        return None
    large_index = large_df.set_index("filter_id")
    best = {"pool": P0, "steps": []}
    seen = {}

    def score_candidate(fid, elim_now):
        kept = large_index.loc[fid]["hist_kept_rate"]
        days = large_index.loc[fid]["hist_applicable_days"]
        kept01 = (kept/100.0) if pd.notna(kept) else 0.5
        days = days if pd.notna(days) else 0
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

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Archetype Suite — Analyze + Plan", layout="wide")
st.title("Archetype Suite — Analyze history & plan large filters")

with st.sidebar:
    st.subheader("Paths")
    winners_path_str = st.text_input("Winners CSV", WINNERS_CSV_DEFAULT)
    filters_path_str = st.text_input("Filters CSV", FILTERS_CSV_DEFAULT)
    pool_path_str    = st.text_input("Pool CSV (today)", POOL_CSV_DEFAULT)
    pool_col_hint    = st.text_input("Pool column (blank → auto 'combo'/'Result')", "")

    st.subheader("Analyzer knobs")
    min_support  = st.number_input("Min applicable days for a signal", 1, 9999, 12)
    min_lift     = st.number_input("Min lift for TOP (≥)", 1.00, 10.0, 1.10, step=0.01)
    max_lag      = st.number_input("Max lift for DANGER (≤)", 0.10, 1.00, 0.90, step=0.01)
    auto_build   = st.checkbox("Auto-build archetype CSVs when missing/stale", True)
    force_build  = st.button("Force rebuild now")

# ---------- auto build / ensure analyzer outputs ----------
@st.cache_data(show_spinner=False)
def _ensure_archetype_outputs_cached(winners, filters, min_support, min_lift, max_lag, force, dep_sig):
    # dep_sig lets cache bust when files change (we pass mtimes)
    paths = analyze_archetype_safety(
        winners_csv=winners,
        filters_csv=filters,
        out_dir=Path("."),
        min_support_for_signal=int(min_support),
        min_lift_for_top=float(min_lift),
        max_lift_for_danger=float(max_lag),
    )
    # Return dataframes directly so we can use them immediately
    return {
        "comp": pd.read_csv(paths["composite"]),
        "dims": pd.read_csv(paths["dimensions"]),
        "top":  pd.read_csv(paths["top"]),
        "danger": pd.read_csv(paths["danger"]),
    }

def ensure_archetype_outputs() -> Dict[str, Optional[pd.DataFrame]]:
    winners_p = Path(winners_path_str); filters_p = Path(filters_path_str)
    outs = [OUT_COMPOSITE, OUT_DIMS, OUT_TOP, OUT_DANGER]
    deps = [winners_p, filters_p]
    need_build = force_build or (auto_build and files_stale(outs, deps))
    if need_build:
        dep_sig = (str(_mtime(winners_p)) + "|" + str(_mtime(filters_p)) +
                   f"|{min_support}|{min_lift}|{max_lag}|{time.time() if force_build else ''}")
        with st.spinner("Building archetype CSVs from history…"):
            df_map = _ensure_archetype_outputs_cached(
                winners_path_str, filters_path_str,
                min_support, min_lift, max_lag,
                force_build, dep_sig
            )
        return df_map
    # else: load what’s on disk if present
    return {
        "comp": _read_csv_if_exists(OUT_COMPOSITE),
        "dims": _read_csv_if_exists(OUT_DIMS),
        "top":  _read_csv_if_exists(OUT_TOP),
        "danger": _read_csv_if_exists(OUT_DANGER),
    }

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["① Analyze history", "② Plan large filters (today)"])

with tab1:
    st.subheader("Build & view archetype CSVs (fact-based, from winners history)")
    if st.button("Run / refresh now", type="primary"):
        _ = ensure_archetype_outputs()
        st.success("Archetype tables regenerated.")

    # Always try to show what we have
    dfs = ensure_archetype_outputs()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top positive signals**")
        if dfs["top"] is not None:
            st.dataframe(dfs["top"], use_container_width=True, hide_index=True)
            _download_btn(dfs["top"], OUT_TOP.name, "Download TOP (CSV)")
        else:
            st.info("No TOP table yet.")
        st.markdown("**Danger signals**")
        if dfs["danger"] is not None:
            st.dataframe(dfs["danger"], use_container_width=True, hide_index=True)
            _download_btn(dfs["danger"], OUT_DANGER.name, "Download DANGER (CSV)")
        else:
            st.info("No DANGER table yet.")
    with c2:
        st.markdown("**All dimension breakdowns**")
        if dfs["dims"] is not None:
            st.dataframe(dfs["dims"], use_container_width=True, hide_index=True)
            _download_btn(dfs["dims"], OUT_DIMS.name, "Download DIMENSIONS (CSV)")
        else:
            st.info("No DIMENSIONS table yet.")
        st.markdown("**Composite archetypes**")
        if dfs["comp"] is not None:
            st.dataframe(dfs["comp"], use_container_width=True, hide_index=True)
            _download_btn(dfs["comp"], OUT_COMPOSITE.name, "Download COMPOSITE (CSV)")
        else:
            st.info("No COMPOSITE table yet.")

with tab2:
    st.subheader("Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.text_input("Seed (1-back, 5 digits)*", value="", max_chars=5).strip()
    with c2:
        prev_seed = st.text_input("Prev seed (2-back, optional)", value="", max_chars=5).strip()
    with c3:
        prev_prev = st.text_input("Prev-prev seed (3-back, optional)", value="", max_chars=5).strip()

    ids_text = st.text_area("Paste Applicable Filter IDs (comma/space/newline separated)", height=110)
    known_winner = st.text_input("Known winner (5 digits, optional — for winner-preserving backtests)", value="", max_chars=5).strip()

    run_btn = st.button("Run planner", type="primary")

    if not run_btn:
        st.stop()

    # Seed badge
    if not (seed.isdigit() and len(seed) == 5):
        st.error("Seed must be exactly 5 digits.")
        st.stop()
    badge = seed_badge(seed, prev_seed, prev_prev)
    st.markdown("#### Seed profile")
    st.write(badge)

    # Ensure archetype stats exist (auto build if needed)
    dfs = ensure_archetype_outputs()

    # Load filters
    filters_path = Path(filters_path_str)
    if not filters_path.exists():
        st.error(f"Filters CSV not found: {filters_path}")
        st.stop()
    df_filters = load_filters_csv(filters_path)

    ids = [x.strip() for x in re.split(r"[,\s]+", ids_text) if x.strip()]
    if not ids:
        st.error("Paste at least one applicable filter ID.")
        st.stop()
    df_filters = df_filters[df_filters["id"].astype(str).isin(set(ids))].copy()
    if df_filters.empty:
        st.error("None of the pasted IDs were found in the filters CSV.")
        st.stop()

    # Compile
    compiled = {}
    names_map = {}
    for _, r in df_filters.iterrows():
        fid = str(r["id"]).strip()
        names_map[fid] = str(r.get("name","")).strip()
        compiled[fid] = compile_filter(r)

    # Load pool
    pool_path = Path(pool_path_str)
    if not pool_path.exists():
        st.error(f"Pool CSV not found: {pool_path}")
        st.stop()
    df_pool = pd.read_csv(pool_path)
    pool_col = None
    for cand in ([pool_col_hint] if pool_col_hint else []) + ["combo","Combo","result","Result"]:
        if cand and cand in df_pool.columns:
            pool_col = cand
            break
    if not pool_col:
        st.error("Could not find a pool column. Add 'combo' (or 'Result'), or specify its name.")
        st.stop()
    pool_series = df_pool[pool_col].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
    pool_series = pool_series[pool_series.str.fullmatch(r"\d{5}")]
    pool_digits = [digits_of(s) for s in pool_series.tolist()]
    if len(pool_digits) == 0:
        st.error("Pool has no valid 5-digit combos after cleaning.")
        st.stop()

    total_even = sum(1 for cd in pool_digits if (sum(cd) % 2) == 0)
    total_odd  = len(pool_digits) - total_even
    base_env   = build_ctx_for_pool(seed, prev_seed, prev_prev)

    # Optional winners for history
    winners_list = None
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
        else:
            st.info("Could not detect winners column; historical kept% will be blank.")

    # Scan pasted filters
    records = []
    E_map = {}
    for _, r in df_filters.iterrows():
        fid = str(r["id"]).strip()
        app_code, expr_code = compiled[fid]
        E, elim_even, elim_odd = elim_set_on_pool(app_code, expr_code, base_env, pool_digits)
        parity_wiper = ((elim_even == total_even and total_even > 0) or (elim_odd == total_odd and total_odd > 0))
        text_blob = f"{r.get('applicable_if','')} || {r.get('expression','')}"
        seed_specific = is_seed_specific(text_blob)
        app_days = blocked = kept_rate = blocked_rate = None
        if winners_list and len(winners_list) >= 2:
            app_days, blocked, kept_rate, blocked_rate = hist_safety(app_code, expr_code, winners_list)
        E_map[fid] = E
        records.append({
            "filter_id": fid,
            "name": names_map.get(fid, ""),
            "elim_count_on_pool": len(E),
            "elim_even": elim_even,
            "elim_odd": elim_odd,
            "parity_wiper": parity_wiper,
            "seed_specific_trigger": seed_specific,
            "hist_applicable_days": app_days,
            "hist_kept_rate": (None if kept_rate is None else round(100*kept_rate, 2)),
            "hist_blocked_rate": (None if blocked_rate is None else round(100*blocked_rate, 2)),
        })
    df_scan = pd.DataFrame(records)

    # Bucket + views
    def bucket(c):
        c = int(c)
        if c >= 701: return "701+"
        if c >= 501: return "501–700"
        if c >= 301: return "301–500"
        if c >= 101: return "101–300"
        if c >=  61: return "61–100"
        if c >=   1: return "1–60"
        return "0"
    large_df = df_scan[(df_scan["elim_count_on_pool"] >= 200) & (~df_scan["parity_wiper"])].copy()
    large_df["aggression_group"] = large_df["elim_count_on_pool"].map(bucket)
    group_order = ["701+","501–700","301–500","101–300","61–100","1–60","0"]
    rank_map = {g:i for i,g in enumerate(group_order)}
    large_df["__g"] = large_df["aggression_group"].map(rank_map).fillna(999).astype(int)
    large_df = large_df.sort_values(
        by=["__g","hist_kept_rate","hist_applicable_days","elim_count_on_pool","filter_id"],
        ascending=[True, False, False, False, True]
    ).drop(columns="__g")

    trig_df = df_scan[df_scan["seed_specific_trigger"]].copy().sort_values(by=["filter_id"])

    st.markdown("### Large filters (non-parity) – bucketed by eliminations on *your* pool")
    if large_df.empty:
        st.info("No large, non-parity filters among the pasted IDs for this pool.")
    else:
        st.dataframe(
            large_df[[
                "aggression_group","filter_id","name","elim_count_on_pool","elim_even","elim_odd",
                "hist_applicable_days","hist_kept_rate","hist_blocked_rate"
            ]],
            use_container_width=True, hide_index=True, height=420
        )
        # downloads
        st.download_button("Download large filters (CSV)",
                           large_df.to_csv(index=False).encode("utf-8"),
                           file_name="large_filters_detected.csv", mime="text/csv")

        # grouped TXT
        lines = []
        for g in group_order:
            sub = large_df[large_df["aggression_group"] == g]
            if sub.empty: continue
            lines.append(f"===== GROUP {g} =====")
            for _, rr in sub.iterrows():
                lines.append(f"{rr['filter_id']}  | kept%={rr['hist_kept_rate']}  | app_days={rr['hist_applicable_days']}  | elim={rr['elim_count_on_pool']}")
        st.download_button("Download group delineations (TXT)",
                           "\n".join(lines).encode("utf-8"),
                           file_name="large_filter_groups.txt", mime="text/plain")

    st.markdown("### Seed-specific trigger filters")
    if trig_df.empty:
        st.info("No seed-specific trigger filters detected among the pasted IDs.")
    else:
        st.dataframe(
            trig_df[[
                "filter_id","name","elim_count_on_pool","elim_even","elim_odd",
                "hist_applicable_days","hist_kept_rate","hist_blocked_rate","parity_wiper"
            ]],
            use_container_width=True, hide_index=True, height=300
        )
        st.download_button("Download trigger filters (CSV)",
                           trig_df.to_csv(index=False).encode("utf-8"),
                           file_name="trigger_filters_detected.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Winner-preserving best-case plan (≤ 45 kept)")
    if not known_winner:
        st.info("Provide a **Known winner** (for backtests) to compute a winner-preserving plan.")
    else:
        if not (known_winner.isdigit() and len(known_winner) == 5):
            st.error("Known winner must be exactly 5 digits.")
        else:
            candidate_ids = set(large_df["filter_id"].astype(str))
            if not candidate_ids:
                st.warning("No large filters available to plan with.")
            else:
                try:
                    winner_idx = pool_series.tolist().index(known_winner)
                except ValueError:
                    winner_idx = None
                    st.warning("Known winner is not in the provided pool — cannot enforce winner-preserving constraint.")
                E_sub = {fid:E_map[fid] for fid in candidate_ids if fid in E_map}
                plan = winner_preserving_plan(
                    large_df=large_df,
                    E_map=E_sub,
                    names_map={fid:names_map.get(fid,"") for fid in candidate_ids},
                    pool_len=len(pool_digits),
                    winner_idx=winner_idx,
                    target_max=45,
                    beam_width=3,
                    max_steps=12
                )
                if not plan:
                    st.info("No winner-preserving reduction possible with the available large filters.")
                else:
                    steps = plan["steps"]
                    plan_df = pd.DataFrame(steps)
                    st.dataframe(plan_df, use_container_width=True, hide_index=True, height=280)
                    st.download_button("Download plan (CSV)",
                                       plan_df.to_csv(index=False).encode("utf-8"),
                                       file_name="recommended_plan.csv", mime="text/csv")
                    txt = io.StringIO()
                    for i, row in enumerate(steps, 1):
                        txt.write(f"Step {i}: {row['filter_id']}  {row['name']}  | eliminated={row['eliminated_now']}  | remaining={row['remaining_after']}\n")
                    st.download_button("Download plan (TXT)",
                                       txt.getvalue().encode("utf-8"),
                                       file_name="recommended_plan.txt", mime="text/plain")
                    kept_idx = sorted(list(plan["final_pool_idx"]))
                    kept_combos = [pool_series.iloc[i] for i in kept_idx]
                    st.caption(f"Final kept pool size: {len(kept_combos)}")
                    st.dataframe(pd.DataFrame({"combo": kept_combos}).head(100),
                                 use_container_width=True, hide_index=True, height=240)
