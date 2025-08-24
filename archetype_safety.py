# archetype_safety.py — compute history-based archetype → filter safety tables
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import pandas as pd
import numpy as np

# ---------- small helpers used inside this module ----------

def _digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s)]

def _sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "sum:very_low"
    if 16 <= total <= 24: return "sum:low"
    if 25 <= total <= 33: return "sum:mid"
    return "sum:high"

def _spread_band(spread: int) -> str:
    if spread <= 3: return "spread:0-3"
    if spread <= 5: return "spread:4-5"
    if spread <= 7: return "spread:6-7"
    if spread <= 9: return "spread:8-9"
    return "spread:10+"

def _classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "struct:quint"
    if counts == [4,1]:     return "struct:quad"
    if counts == [3,2]:     return "struct:triple_double"
    if counts == [3,1,1]:   return "struct:triple"
    if counts == [2,2,1]:   return "struct:double_double"
    if counts == [2,1,1,1]: return "struct:double"
    return "struct:single"

def _parity_label(digs: List[int]) -> str:
    even = sum(1 for d in digs if d % 2 == 0)
    odd = 5 - even
    return f"parity:{even}E{odd}O"

def _mixed_hi_lo(digs: List[int]) -> str:
    lo = any(d <= 4 for d in digs)
    hi = any(d >= 5 for d in digs)
    if lo and hi: return "mix:hi+lo"
    if hi and not lo: return "mix:hi_only"
    if lo and not hi: return "mix:lo_only"
    return "mix:unknown"

def _env_for_day(idx: int, winners: List[str]) -> Dict[str, object]:
    """
    Recreate the same eval environment style used in your recommender:
    seed = winners[idx-1], winner combo = winners[idx]
    """
    seed   = winners[idx-1]
    winner = winners[idx]
    seed_list  = _digits_of(seed)
    combo_list = sorted(_digits_of(winner))

    MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
    VTRAC  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

    env = {
        "combo": winner,
        "combo_digits": set(combo_list),
        "combo_digits_list": combo_list,
        "combo_sum": sum(combo_list),
        "combo_sum_cat": _sum_category(sum(combo_list)),
        "combo_sum_category": _sum_category(sum(combo_list)),

        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": sum(seed_list),
        "seed_sum_category": _sum_category(sum(seed_list)),

        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),

        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),

        "mirror": MIRROR,
        "vtrac": VTRAC,

        # minimal safe builtins
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
        "abs": abs,
    }
    return env

def _derive_dimensions(env: Dict[str, object]) -> Dict[str, str]:
    """Return a dict of dimension -> value strings (seed-based archetype traits)."""
    seed = env["seed"]
    winner = env["combo"]
    seed_d = env["seed_digits_list"]
    win_d  = env["combo_digits_list"]

    # overlap & duplicate
    overlap_ge1 = "overlap:>=1" if (len(set(seed_d) & set(win_d)) >= 1) else "overlap:0"
    has_dup     = "dup:yes" if len(win_d) != len(set(win_d)) else "dup:no"

    # parity, mix, sum, spread, structure (winner side)
    parity   = _parity_label(win_d)
    mix      = _mixed_hi_lo(win_d)
    sum_band = _sum_category(sum(win_d))
    spread   = _spread_band(max(win_d)-min(win_d))
    struct   = _classify_structure(win_d)

    # seed-side quick tags that are often useful
    seed_parity   = "seed_" + _parity_label(seed_d).split(":",1)[1]
    seed_struct   = "seed_" + _classify_structure(seed_d).split(":",1)[1]

    # Keep a compact set that maps closely to what you described
    dims = {
        "overlap": overlap_ge1,
        "dup": has_dup,
        "parity": parity,
        "mix": mix,
        "sum": sum_band,
        "spread": spread,
        "struct": struct,
        "seed_parity": seed_parity,
        "seed_struct": seed_struct,
    }
    return dims

def _composite_signature(dims: Dict[str, str]) -> str:
    # deterministic order for readability
    keys = ["overlap","dup","mix","parity","sum","spread","struct","seed_parity","seed_struct"]
    return "|".join(dims[k] for k in keys)

@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

def _load_winners(path: str) -> List[str]:
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
    vals_list = vals.tolist()
    if len(vals_list) < 2:
        raise ValueError("Need at least 2 rows (seed + winner).")
    return vals_list

def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if pd.isna(x): return False
    return str(x).strip().lower() in {"true","1","yes","y"}

def _load_filters(path: str, only_ids: Optional[set[str]] = None) -> List[FilterDef]:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    req = ["id","name","enabled","applicable_if","expression"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV missing column: {r}")

    df["enabled"] = df["enabled"].map(_to_bool)
    # normalize quotes
    for col in ["applicable_if","expression"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        df[col] = df[col].apply(lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s)

    out: List[FilterDef] = []
    for _, r in df.iterrows():
        fid = str(r["id"]).strip()
        if only_ids and fid not in only_ids:
            continue
        out.append(FilterDef(
            fid=fid,
            name=str(r["name"]).strip(),
            enabled=bool(r["enabled"]),
            applicable_if=str(r["applicable_if"]).strip(),
            expression=str(r["expression"]).strip(),
        ))
    # keep only enabled
    out = [f for f in out if f.enabled]
    return out

# ---------- main entry called by your Streamlit app ----------

def analyze_archetype_safety(
    winners_csv: str,
    filters_csv: str,
    out_dir: Path | str,
    *,
    min_support_for_signal: int = 12,
    min_lift_for_top: float = 1.10,
    max_lift_for_danger: float = 0.90,
    applicable_only_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Replay history. For each filter (restricted to applicable_only_ids if given):
      - Baseline pass rate over applicable days
      - Dimension-level pass rates + lift vs baseline
      - Composite-archetype pass rates + lift vs baseline
    Writes 4 CSVs and returns their paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    winners = _load_winners(winners_csv)
    only_ids = set(x.strip() for x in applicable_only_ids) if applicable_only_ids else None
    filters = _load_filters(filters_csv, only_ids=only_ids)

    # Pre-compile per-filter code
    compiled: Dict[str, Tuple[object, object, str]] = {}
    for f in filters:
        try:
            c_app = compile(f.applicable_if if f.applicable_if else "True", "<app_if>", "eval")
        except Exception:
            c_app = compile("False", "<app_if>", "eval")
        try:
            c_blk = compile(f.expression if f.expression else "False", "<expr>", "eval")
        except Exception:
            c_blk = compile("False", "<expr>", "eval")
        compiled[f.fid] = (c_app, c_blk, f.name)

    # Counters
    base_app = Counter()  # per fid: applicable days
    base_pass = Counter() # per fid: days where winner NOT eliminated (pass)
    dim_app  = Counter()  # keyed by (fid, dim_name, dim_value)
    dim_pass = Counter()
    comp_app = Counter()  # keyed by (fid, composite_str)
    comp_pass = Counter()

    # Iterate days 1..N-1 (each day i uses seed=w[i-1], winner=w[i])
    N = len(winners)
    for i in range(1, N):
        env = _env_for_day(i, winners)
        dims = _derive_dimensions(env)
        comp_sig = _composite_signature(dims)

        for f in filters:
            c_app, c_blk, _nm = compiled[f.fid]
            try:
                applicable = bool(eval(c_app, {"__builtins__": {}}, env))
            except Exception:
                applicable = False
            if not applicable:
                continue

            try:
                blocks = bool(eval(c_blk, {"__builtins__": {}}, env))
            except Exception:
                blocks = False

            passes = (not blocks)

            # Baseline
            base_app[f.fid]  += 1
            if passes: base_pass[f.fid] += 1

            # Dimension-level
            for dname, dval in dims.items():
                key = (f.fid, dname, dval)
                dim_app[key] += 1
                if passes: dim_pass[key] += 1

            # Composite
            ckey = (f.fid, comp_sig)
            comp_app[ckey] += 1
            if passes: comp_pass[ckey] += 1

    # Build baseline DF
    rows_base = []
    for fid in base_app.keys():
        a = base_app[fid]
        p = base_pass[fid]
        br = (p / a) if a > 0 else np.nan
        rows_base.append({"filter_id": fid, "baseline_applicable_n": a, "baseline_pass_n": p, "baseline_pass_rate": br})
    df_base = pd.DataFrame(rows_base)
    df_base.set_index("filter_id", inplace=True)

    # Dimension-level DF
    dim_rows = []
    for (fid, dname, dval), a in dim_app.items():
        p = dim_pass[(fid, dname, dval)]
        pr = (p / a) if a > 0 else np.nan
        b_app = df_base.loc[fid, "baseline_applicable_n"] if fid in df_base.index else 0
        b_rate = df_base.loc[fid, "baseline_pass_rate"] if fid in df_base.index else np.nan
        lift = (pr / b_rate) if (a >= min_support_for_signal and b_rate and not np.isnan(b_rate) and b_rate > 0) else np.nan
        dim_rows.append({
            "filter_id": fid,
            "dimension": dname,
            "value": dval,
            "applicable_n": a,
            "pass_n": p,
            "pass_rate": pr,
            "baseline_applicable_n": b_app,
            "baseline_pass_rate": b_rate,
            "lift_vs_baseline": lift,
        })
    df_dims = pd.DataFrame(dim_rows).sort_values(
        ["filter_id","dimension","value"]
    )

    # Add names
    fid_to_name = {f.fid: compiled[f.fid][2] for f in filters}
    if not df_dims.empty:
        df_dims.insert(1, "filter_name", df_dims["filter_id"].map(fid_to_name).fillna(""))

    # Top & Danger signals
    def _is_top(r):
        return (r["applicable_n"] >= min_support_for_signal) and (r["lift_vs_baseline"] >= min_lift_for_top)
    def _is_danger(r):
        return (r["applicable_n"] >= min_support_for_signal) and (r["lift_vs_baseline"] <= max_lift_for_danger)

    df_top = df_dims[df_dims.apply(_is_top, axis=1)].copy()
    df_top = df_top.sort_values(["filter_id","lift_vs_baseline","applicable_n"], ascending=[True, False, False])

    df_danger = df_dims[df_dims.apply(_is_danger, axis=1)].copy()
    df_danger = df_danger.sort_values(["filter_id","lift_vs_baseline","applicable_n"], ascending=[True, True, False])

    # Composite DF
    comp_rows = []
    for (fid, comp_sig), a in comp_app.items():
        p = comp_pass[(fid, comp_sig)]
        pr = (p / a) if a > 0 else np.nan
        b_app = df_base.loc[fid, "baseline_applicable_n"] if fid in df_base.index else 0
        b_rate = df_base.loc[fid, "baseline_pass_rate"] if fid in df_base.index else np.nan
        lift = (pr / b_rate) if (a >= min_support_for_signal and b_rate and not np.isnan(b_rate) and b_rate > 0) else np.nan
        comp_rows.append({
            "filter_id": fid,
            "filter_name": fid_to_name.get(fid, ""),
            "archetype_signature": comp_sig,
            "applicable_n": a,
            "pass_n": p,
            "pass_rate": pr,
            "baseline_applicable_n": b_app,
            "baseline_pass_rate": b_rate,
            "lift_vs_baseline": lift,
        })
    df_comp = pd.DataFrame(comp_rows).sort_values(
        ["filter_id","lift_vs_baseline","applicable_n"], ascending=[True, False, False]
    )

    # Write outputs
    p_dims   = out_dir / "archetype_filter_dimension_stats.csv"
    p_top    = out_dir / "archetype_filter_top_signals.csv"
    p_danger = out_dir / "archetype_filter_danger_signals.csv"
    p_comp   = out_dir / "archetype_filter_composite_stats.csv"

    df_dims.to_csv(p_dims, index=False)
    df_top.to_csv(p_top, index=False)
    df_danger.to_csv(p_danger, index=False)
    df_comp.to_csv(p_comp, index=False)

    return {
        "dimension_stats": str(p_dims),
        "top_signals": str(p_top),
        "danger_signals": str(p_danger),
        "composite_stats": str(p_comp),
    }
