from __future__ import annotations
import streamlit as st
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math, re
import pandas as pd
import numpy as np

# =============== UI ===============
st.set_page_config(page_title="Archetype Safe Filter Finder", layout="wide")
st.title("Archetype Safe Filter Finder")

# =============== Defaults ===============
WINNERS_DEFAULT = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_DEFAULT = "lottery_filters_batch_10.csv"
POOL_DEFAULT    = "today_pool.csv"

MIN_LARGE_ELIMS_DEFAULT = 60        # pool eliminations to call a filter "large"
PARITY_WIGGLE_DEFAULT   = 0.12      # detect parity killers if elim share within ±12% of half the pool

# =============== Domain helpers (same spirit as your recommender) ===============
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

def parity_major_label(digs: List[int]) -> str:
    return "even≥3" if sum(d%2==0 for d in digs) >= 3 else "even≤2"

@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:  # blank means "no additional condition"
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    """
    Historical environment for day idx using:
      seed = winners[idx-1]
      winner = winners[idx]
    """
    seed   = winners[idx-1]
    winner = winners[idx]
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(winner))
    env = {
        "combo": winner,
        "combo_digits": set(combo_list),
        "combo_digits_list": combo_list,
        "combo_sum": sum(combo_list),
        "combo_sum_cat": sum_category(sum(combo_list)),
        "combo_sum_category": sum_category(sum(combo_list)),

        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": sum(seed_list),
        "seed_sum_category": sum_category(sum(seed_list)),

        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),

        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),

        "mirror": MIRROR, "vtrac": VTRAC,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
        "classify_structure": classify_structure,  # convenience if used in CSVs
        "parity_major_label": parity_major_label,
    }
    return env

# Seed-only environment (no winner injected)
def build_env_for_seed(seed: str, prev: Optional[str]=None, prevprev: Optional[str]=None) -> Dict[str, object]:
    seed_list = digits_of(seed)
    env = {
        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": sum(seed_list),
        "seed_sum_category": sum_category(sum(seed_list)),
        "spread_seed": max(seed_list) - min(seed_list),
        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "mirror": MIRROR, "vtrac": VTRAC,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
    }
    if prev:
        env["prev"] = prev
        env["prev_digits"] = set(digits_of(prev))
        env["prev_digits_list"] = digits_of(prev)
    if prevprev:
        env["prevprev"] = prevprev
        env["prevprev_digits"] = set(digits_of(prevprev))
        env["prevprev_digits_list"] = digits_of(prevprev)
    return env

# =============== CSV loaders ===============
def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "Result" if "Result" in df.columns else None
    if col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            if (vals.str.fullmatch(r"\d{5}")).all():
                col = c; break
    if col is None:
        raise ValueError("Winners CSV must have a 5-digit column (e.g., 'Result').")
    vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    vals = vals[vals.str.fullmatch(r"\d{5}")]
    return vals.tolist()

def load_filters(path: str) -> List[FilterDef]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
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
        df[col] = df[col].apply(
            lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s
        )
    out: List[FilterDef] = []
    for _, r in df.iterrows():
        out.append(FilterDef(
            str(r["id"]).strip(),
            str(r["name"]).strip(),
            bool(r["enabled"]),
            str(r["applicable_if"]).strip(),
            str(r["expression"]).strip(),
        ))
    return out

def load_pool(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "combo" not in df.columns:
        if "Result" in df.columns:
            df = df.rename(columns={"Result":"combo"})
        else:
            raise ValueError("Pool CSV must have 'combo' or 'Result' column.")
    ser = df["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    return ser[ser.str.fullmatch(r"\d{5}")].tolist()

# =============== Archetype definition ===============
def archetype_of_seed(seed: str) -> Tuple[str, Dict[str,str]]:
    """
    A *seed-only* archetype. We label by key features and return a human explanation dict.
    """
    digs = digits_of(seed)
    low = sum(1 for d in digs if d <= 4)
    high = 5 - low
    lh = "mixed" if (low>0 and high>0) else ("all-low" if high==0 else "all-high")
    dup = "dup" if max(Counter(digs).values()) >= 2 else "no-dup"

    key = {
        "SUM": sum_category(sum(digs)),
        "SPREAD": spread_band(max(digs)-min(digs)),
        "STRUCT": classify_structure(digs),
        "PARITY": parity_major_label(digs),
        "LH": lh,
        "DUP": dup,
    }
    # Canonical short name
    name = f"{key['STRUCT']} | {key['LH']} | {key['PARITY']} | {key['SUM']} | {key['SPREAD']} | {key['DUP']}"
    return name, key

# For history: make an archetype key using the *seed of that day* (winners[i-1]).
def day_arche_key(winners: List[str], i: int) -> str:
    seed = winners[i-1]
    name, key = archetype_of_seed(seed)
    return name

# =============== Filter classifiers (large, parity, seed-specific) ===============
def is_seed_specific(f: FilterDef) -> bool:
    blob = (f.applicable_if + " " + f.expression).lower()
    return any(tok in blob for tok in ["seed", "seed_", "seedsum", "seed_sum", "seeddigits", "seed_digits"])

def detect_parity_killer(f: FilterDef, elim_share: float, wiggle: float) -> bool:
    """
    Heuristic: if name or code mentions even/odd/parity AND elimination share is ~ half the pool.
    """
    txt = (f.name + " " + f.applicable_if + " " + f.expression).lower()
    mentions = any(w in txt for w in ["even", "odd", "parity", "combo_sum%2", "sum%2", "combo sum % 2", "sum % 2"])
    if not mentions:
        return False
    return abs(elim_share - 0.5) <= wiggle

def apply_filter_to_pool(f: FilterDef, seed_env: Dict[str,object], pool: List[str]) -> Tuple[int, int]:
    elim = keep = 0
    for s in pool:
        clist = sorted(digits_of(s))
        env = dict(seed_env)
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
        # if both applicable_if and expression are True -> this combo would be eliminated
        if safe_eval(f.applicable_if, env) and safe_eval(f.expression, env):
            elim += 1
        else:
            keep += 1
    return elim, keep

# =============== Historical safety for an archetype ===============
def historical_safety_for_archetype(
    winners: List[str],
    filters: List[FilterDef],
    arch_name_now: str,
) -> Dict[str, Dict[str, float]]:
    """
    For each filter, evaluate only on days whose *seed* archetype == current archetype.
    When applicable on those days, what % of the winners were *kept*?
    Returns: fid -> {'days':N, 'app':A, 'kept':K, 'kept_pct':K/A}
    """
    N = len(winners)
    out: Dict[str, Dict[str, float]] = {f.fid: {"days": 0, "app": 0, "kept": 0} for f in filters if f.enabled}
    if N < 2:
        return out

    # prebuild envs for speed
    envs = [build_env_for_draw(i, winners) for i in range(1, N)]
    arch_keys = [day_arche_key(winners, i) for i in range(1, N)]

    for i, env in enumerate(envs, start=1):
        if arch_keys[i-1] != arch_name_now:
            continue
        for f in filters:
            if not f.enabled: 
                continue
            out[f.fid]["days"] += 1
            applicable = safe_eval(f.applicable_if, env)
            if applicable:
                out[f.fid]["app"] += 1
                blocked = safe_eval(f.expression, env)  # True means this filter would eliminate today's winner
                if not blocked:
                    out[f.fid]["kept"] += 1

    for fid, d in out.items():
        d["kept_pct"] = (d["kept"] / d["app"] * 100.0) if d["app"] > 0 else None
    return out

# =============== UI – Inputs ===============
colA, colB, colC = st.columns(3)
with colA:
    winners_path = st.text_input("Winners CSV", WINNERS_DEFAULT)
with colB:
    filters_path = st.text_input("Filters CSV", FILTERS_DEFAULT)
with colC:
    pool_path = st.text_input("Pool CSV", POOL_DEFAULT)

col1, col2, col3 = st.columns(3)
with col1:
    seed_override = st.text_input("Override seed (5 digits, optional)", "")
with col2:
    prev1_override = st.text_input("Override 1-back (optional)", "")
with col3:
    prev2_override = st.text_input("Override 2-back (optional)", "")

colX, colY, colZ = st.columns(3)
with colX:
    min_large_elims = st.number_input("Min eliminations to call a filter 'LARGE' (on current pool)",
                                      value=MIN_LARGE_ELIMS_DEFAULT, min_value=1, step=1)
with colY:
    parity_wiggle = st.slider("Parity-killer wiggle (± of 50%)", 0.0, 0.25, PARITY_WIGGLE_DEFAULT, 0.01,
                              help="If a filter eliminates ~half the pool and mentions even/odd, we treat it as a parity eliminator and exclude it.")
with colZ:
    run_btn = st.button("Run archetype analysis", type="primary")

st.markdown("---")

# =============== Run ===============
if run_btn:
    try:
        winners_all = load_winners(winners_path)
        filters_all = load_filters(filters_path)
        pool = load_pool(pool_path)
    except Exception as e:
        st.error(f"Failed to load inputs: {e}")
        st.stop()

    if len(winners_all) < 3:
        st.error("Winners CSV must contain at least 3 rows.")
        st.stop()

    # Establish seed / prevs
    if seed_override and re.fullmatch(r"\d{5}", seed_override):
        seed_now = seed_override
        prev1 = prev1_override if re.fullmatch(r"\d{5}", prev1_override or "") else winners_all[-2]
        prev2 = prev2_override if re.fullmatch(r"\d{5}", prev2_override or "") else winners_all[-3]
    else:
        # By default, use the most recent 3 winners as prevprev, prev, seed
        prev2, prev1, seed_now = winners_all[-3], winners_all[-2], winners_all[-1]

    seed_env = build_env_for_seed(seed_now, prev1, prev2)

    # Current seed archetype
    arch_name, arch_dict = archetype_of_seed(seed_now)

    st.subheader("Current seed archetype")
    c1, c2 = st.columns([1,2])
    with c1:
        st.metric("Seed", seed_now)
        st.metric("Archetype", arch_name)
    with c2:
        st.markdown("**Profile (why this archetype):**")
        st.json(arch_dict, expanded=True)

    # Compute elimination on today's pool for each filter to find LARGE ones; also collect seed-specific applicable filters.
    large_rows = []
    seed_specific_rows = []

    pool_N = max(len(pool), 1)
    for f in filters_all:
        if not f.enabled:
            continue

        # "seed-specific" = filters that reference seed & are applicable now
        if is_seed_specific(f):
            if safe_eval(f.applicable_if, seed_env):
                seed_specific_rows.append({"filter_id": f.fid, "name": f.name, "applicable_now": True})

        # Compute pool eliminations
        elim, keep = apply_filter_to_pool(f, seed_env, pool)
        elim_share = elim / pool_N

        # Detect parity-killers (exclude from LARGE list)
        parity_kill = detect_parity_killer(f, elim_share, parity_wiggle)

        if (elim >= int(min_large_elims)) and (not parity_kill):
            large_rows.append({
                "filter_id": f.fid,
                "name": f.name,
                "elim_count": int(elim),
                "elim_share": round(elim_share*100.0, 2),
                "obj": f,  # stash for later
            })

    # Turn to DataFrames
    df_seed_specific = pd.DataFrame(seed_specific_rows).sort_values("filter_id") if seed_specific_rows else pd.DataFrame(columns=["filter_id","name","applicable_now"])
    df_large = pd.DataFrame(large_rows).sort_values(["elim_count","filter_id"], ascending=[False, True]) if large_rows else pd.DataFrame(columns=["filter_id","name","elim_count","elim_share"])

    if df_large.empty:
        st.warning("No LARGE filters detected for this pool with the current threshold / parity rule.")
        st.stop()

    # Historical safety restricted to the current archetype
    # Evaluate for only the LARGE filters (to keep it fast and relevant)
    fid_to_filter = {r["filter_id"]: r["obj"] for r in large_rows}
    large_filter_list = [fid_to_filter[fid] for fid in df_large["filter_id"].tolist()]

    safety = historical_safety_for_archetype(winners_all, large_filter_list, arch_name)

    # Merge safety into the large table
    def safe_get(fid, k, default):
        d = safety.get(fid, {})
        v = d.get(k, default)
        return v if v is not None else default

    df_large["arch_days"] = df_large["filter_id"].map(lambda fid: safe_get(fid, "days", 0)).astype(int)
    df_large["arch_applicable_days"] = df_large["filter_id"].map(lambda fid: safe_get(fid, "app", 0)).astype(int)
    df_large["arch_kept_days"] = df_large["filter_id"].map(lambda fid: safe_get(fid, "kept", 0)).astype(int)
    df_large["arch_kept_%"] = df_large["filter_id"].map(lambda fid: round(safe_get(fid, "kept_pct", 0.0), 2) if safe_get(fid, "kept_pct", None) is not None else None)

    # Final sort: historically safest first, tie-break by stronger pool reduction
    df_large = df_large.sort_values(
        by=["arch_kept_%","arch_applicable_days","elim_count","filter_id"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    # =============== Output sections ===============
    st.markdown("---")
    st.subheader("Seed-specific (defining) filters — applicable now")
    if df_seed_specific.empty:
        st.info("No seed-specific filters are applicable right now (based on the code heuristics that look for 'seed' in expressions).")
    else:
        st.dataframe(df_seed_specific[["filter_id","name","applicable_now"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("LARGE filters — historically safe for this archetype (and efficient on today’s pool)")

    # A little legend for columns
    st.caption(
        "- **arch_days**: # of historical days with this *seed archetype*\n"
        "- **arch_applicable_days**: of those days, # where this filter was applicable\n"
        "- **arch_kept_%**: when applicable on those days, % of winners *kept* (not blocked)\n"
        "- **elim_count**/**elim_share**: how many / what % of **today’s pool** the filter would eliminate"
    )

    st.dataframe(
        df_large[["filter_id","name","arch_kept_%","arch_applicable_days","arch_days","elim_count","elim_share"]],
        use_container_width=True,
        hide_index=True
    )

    # Export
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"archetype_profile_{ts}"

    out = df_large.copy()
    out.insert(0, "seed", seed_now)
    out.insert(1, "archetype", arch_name)
    for k,v in arch_dict.items():
        out[f"arch_{k.lower()}"] = v

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", data=csv_bytes, file_name=f"{base}.csv", mime="text/csv")

    st.success("Done. Use this table to choose big filters that stay safe for *this* seed archetype, ordered by historical kept% then pool impact.")
