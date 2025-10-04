# large_filters_planner_FULL_v2.py
# Streamlit app: UI-linked Hot/Cold/Due, complete env aliases, mirror helpers,
# safe builtins, neutral placeholders for applicable_if, skip reporter,
# and extra forgiving helpers to cut down runtime/type skips.

from __future__ import annotations

import io
import math
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from collections import Counter

# ==============================
# Page
# ==============================
st.set_page_config(page_title="Large Filters Planner — FULL v2", layout="wide")
st.title("Large Filters Planner — FULL v2 (UI H/C/D + full env + forgiving helpers)")

# ==============================
# Core maps & helpers
# ==============================
VTRAC: Dict[int, int]  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
V_TRAC_GROUPS: Dict[int, int] = VTRAC  # alias used in some CSVs

MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
mirror = MIRROR  # some CSVs reference lowercase

SAFE_BUILTINS = {
    # builtins
    "abs": abs, "int": int, "str": str, "float": float, "round": round,
    "len": len, "sum": sum, "max": max, "min": min, "any": any, "all": all,
    "set": set, "sorted": sorted, "list": list, "tuple": tuple, "dict": dict,
    "range": range, "enumerate": enumerate, "map": map, "filter": filter,
    # safe modules/types
    "math": math, "re": re, "random": random, "Counter": Counter,
}

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

# alias some CSVs use
structure_of = classify_structure

def spread_band(spread: int) -> str:
    if spread <= 3: return "0–3"
    if spread <= 5: return "4–5"
    if spread <= 7: return "6–7"
    if spread <= 9: return "8–9"
    return "10+"

def parse_list_any(text: str) -> List[str]:
    if not text: return []
    raw = text.replace("\t", ",").replace("\n", ",").replace(";", ",").replace(" ", ",")
    return [p.strip() for p in raw.split(",") if p.strip()]

def seed_features(digs: List[int]) -> Tuple[str, str, str]:
    if not digs: return "", "", ""
    s = sum(digs)
    parity = "Even" if s % 2 == 0 else "Odd"
    return sum_category(s), parity, classify_structure(digs)

def seed_profile(seed: str, prev_seed: str = "", prev_prev: str = "") -> Dict[str, object]:
    sd = digits_of(seed) if seed else []
    evens = sum(1 for d in sd if d % 2 == 0)
    odds = len(sd) - evens if sd else 0
    parity_str = f"{evens}E/{odds}O" if len(sd) == 5 else "—"
    total = sum(sd) if sd else 0
    s_cat = sum_category(total)
    s_spread = (max(sd) - min(sd)) if sd else 0
    s_spread_band = spread_band(s_spread)
    c = Counter(sd)
    structure = classify_structure(sd) if sd else "—"
    has_dupe = any(v >= 2 for v in c.values())
    hi = sum(1 for d in sd if d >= 5); lo = sum(1 for d in sd if d <= 4)
    hi_lo = "Hi+Lo" if hi > 0 and lo > 0 else ("All-Hi" if lo == 0 and hi > 0 else ("All-Low" if hi == 0 and lo > 0 else "—"))
    pdigs = set(digits_of(prev_seed)) if prev_seed else set()
    carry = len(set(sd) & pdigs) if sd else 0
    new_cnt = len(set(sd) - pdigs) if sd else 0
    vdiv = len(set(VTRAC[d] for d in sd)) if sd else 0
    signature = f"{structure} • {parity_str} • {s_cat} • {s_spread_band} • {hi_lo}" if sd else "—"
    return {
        "digits": sd, "signature": signature, "sum": total, "sum_cat": s_cat,
        "parity": parity_str, "structure": structure, "spread": s_spread,
        "spread_band": s_spread_band, "hi_lo": hi_lo, "has_dupe": has_dupe,
        "carry_from_prev": carry, "new_vs_prev": new_cnt, "vtrac_diversity": vdiv,
    }

def hot_cold_due(winners_digits: List[List[int]], k: int = 10) -> Tuple[Set[int], Set[int], Set[int]]:
    """Helper for optional auto-fill when user leaves all H/C/D boxes blank."""
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
    hot = {d for d, c in most if c >= thresh}
    least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
    coldk = 4
    cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
    cold = {d for d, c in least if c <= cth}
    last2 = set(d for row in winners_digits[-2:] for d in row)
    due = set(range(10)) - last2
    return hot, cold, due

# ---- Forgiving helpers to reduce runtime/type skips ----
def safe_digits(x):
    """Return list[int] of all digits in x; empty list if none."""
    try:
        return [int(ch) for ch in str(x) if ch.isdigit()]
    except Exception:
        return []

def digit_sum(x):
    """Sum of digits in x; 0 if not applicable."""
    ds = safe_digits(x)
    return sum(ds) if ds else 0

def parity_even(n):
    """True if integer n or digit-sum of string n is even."""
    try:
        if isinstance(n, int):
            return (n % 2) == 0
        return (digit_sum(n) % 2) == 0
    except Exception:
        return False

def vtrac_of(d):
    """VTRAC group of a single digit; returns None if invalid."""
    try:
        d = int(d)
        return VTRAC[d] if d in VTRAC else None
    except Exception:
        return None

def vtrac_count(digs):
    """Number of distinct VTRAC groups across some digits-like input."""
    try:
        return len({VTRAC[int(d)] for d in safe_digits(digs)})
    except Exception:
        return 0

def pair_count(digs):
    """How many digit values appear at least twice."""
    c = Counter(safe_digits(digs))
    return sum(1 for _, v in c.items() if v >= 2)

def has_pair(digs):
    return pair_count(digs) > 0

def safe_div(a, b):
    """Division that returns 0 on ZeroDivisionError or invalid inputs."""
    try:
        return float(a) / float(b)
    except Exception:
        return 0.0

# is_hot/is_cold/is_due depend on env hot_set/cold_set/due_set (injected per env)
def _mk_is_hot(env):
    return lambda d: (int(d) in env.get("hot_set", set())) if str(d).isdigit() else False

def _mk_is_cold(env):
    return lambda d: (int(d) in env.get("cold_set", set())) if str(d).isdigit() else False

def _mk_is_due(env):
    return lambda d: (int(d) in env.get("due_set", set())) if str(d).isdigit() else False

def digits_intersection(a, b):
    return sorted(set(safe_digits(a)) & set(safe_digits(b)))

def digits_union(a, b):
    return sorted(set(safe_digits(a)) | set(safe_digits(b)))

# ==============================
# Environments complete
# ==============================
def _make_prev_pattern(pp: List[int], p: List[int], s: List[int]) -> tuple:
    def _pair(digs):
        if not digs: return ("","")
        tot = sum(digs)
        return (sum_category(tot), "Even" if tot % 2 == 0 else "Odd")
    return (*_pair(pp), *_pair(p), *_pair(s))

def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    """Base env for applicable_if and as the parent of combo envs."""
    sd  = digits_of(seed) if seed else []
    sd2 = digits_of(prev_seed) if prev_seed else []
    sd3 = digits_of(prev_prev_seed) if prev_prev_seed else []

    # convenience
    last2 = set(sd) | set(sd2)
    common_to_both = set(sd) & set(sd2)

    env = {
        # seed & history context
        "seed_digits": sd, "prev_seed_digits": sd2, "prev_prev_seed_digits": sd3,
        "seed_digits_1": sd2, "seed_digits_2": sd3, "seed_digits_3": [],
        "new_seed_digits": list(set(sd) - set(sd2)),
        "seed_counts": Counter(sd), "seed_sum": sum(sd) if sd else 0,
        "prev_sum_cat": sum_category(sum(sd)) if sd else "",
        "seed_vtracs": set(VTRAC[d] for d in sd) if sd else set(),
        "prev_pattern": _make_prev_pattern(sd3, sd2, sd),
        "common_to_both": common_to_both, "last2": last2,

        # core maps
        "VTRAC": VTRAC, "V_TRAC_GROUPS": V_TRAC_GROUPS,
        "mirror": MIRROR, "MIRROR": MIRROR,

        # Temperature lists and aliases — DIRECT from UI
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits":  sorted(set(due_digits)),
        "hot":  sorted(set(hot_digits)),
        "cold": sorted(set(cold_digits)),
        "due":  sorted(set(due_digits)),

        # Sets for fast membership & is_hot/is_cold/is_due
        "hot_set": set(hot_digits), "cold_set": set(cold_digits), "due_set": set(due_digits),

        # Mirror helpers and precomputed mirror sets
        "mirror_of": (lambda d: MIRROR.get(int(d), int(d)) if str(d).isdigit() else d),
        "seed_mirror_digits": sorted({MIRROR[d] for d in sd}) if sd else [],

        # Builtins whitelist (safe)
        **SAFE_BUILTINS,

        # Neutral placeholders so applicable_if can reference combo_* safely
        "combo": "",
        "combo_digits": [],
        "combo_digits_list": [],
        "combo_set": set(),
        "combo_sum": 0,
        "combo_sum_cat": sum_category(0),
        "combo_sum_is_even": False,
        "combo_vtracs": set(),
        "combo_mirror_digits": [],
        "combo_structure": "single",
        "combo_last_digit": None,
        "spread": 0,
        "seed_spread": (max(sd) - min(sd)) if sd else 0,
        "has_mirror_pair": False,
        "mirror_pair_count": 0,
        "mirror_pairs": set(),
        "mirror_overlap_with_seed": 0,

        # helpers
        "sum_category": sum_category, "structure_of": structure_of,
        "safe_digits": safe_digits, "digit_sum": digit_sum, "parity_even": parity_even,
        "vtrac_of": vtrac_of, "vtrac_count": vtrac_count,
        "pair_count": pair_count, "has_pair": has_pair,
        "safe_div": safe_div,
        # is_* are injected after env creation because they close over env sets
        # digits set ops
        "digits_intersection": digits_intersection, "digits_union": digits_union,

        "seed_value": int(seed) if (seed and seed.isdigit()) else None,
        "nan": float('nan'),
        "winner_structure": classify_structure(sd) if sd else "",
    }
    # Late-bind dependent helpers
    env["is_hot"]  = _mk_is_hot(env)
    env["is_cold"] = _mk_is_cold(env)
    env["is_due"]  = _mk_is_due(env)
    return env

def combo_env(base_env: Dict, combo: str) -> Dict:
    """Per-combo env inherits base env and augments with combo_* details."""
    cd = digits_of(combo)
    env = dict(base_env)
    cset = set(cd)

    combo_mirror_digits = sorted({base_env["mirror"][d] for d in cd}) if cd else []
    mirror_pairs = {tuple(sorted((d, base_env["mirror"][d]))) for d in cset
                    if base_env["mirror"][d] in cset and base_env["mirror"][d] != d}
    mirror_pair_count = len(mirror_pairs)
    has_mirror_pair = mirror_pair_count > 0
    mirror_overlap_with_seed = len(cset & set(base_env.get("seed_mirror_digits", [])))

    env.update({
        "combo": combo,
        "combo_digits": sorted(cd),
        "combo_digits_list": sorted(cd),
        "combo_set": cset,
        "combo_sum": sum(cd),
        "combo_sum_cat": sum_category(sum(cd)),
        "combo_sum_is_even": (sum(cd) % 2 == 0),
        "combo_vtracs": set(VTRAC[d] for d in cd),
        "combo_structure": classify_structure(cd),
        "combo_last_digit": cd[-1] if cd else None,

        "spread": (max(cd) - min(cd)) if cd else 0,
        "seed_spread": (max(base_env["seed_digits"]) - min(base_env["seed_digits"])) if base_env["seed_digits"] else 0,

        "combo_mirror_digits": combo_mirror_digits,
        "has_mirror_pair": has_mirror_pair,
        "mirror_pairs": mirror_pairs,
        "mirror_pair_count": mirror_pair_count,
        "mirror_overlap_with_seed": mirror_overlap_with_seed,

        # keep H/C/D sets in per-combo env for is_* helpers
        "hot_set": set(env.get("hot_digits", [])),
        "cold_set": set(env.get("cold_digits", [])),
        "due_set": set(env.get("due_digits", [])),
    })
    # refresh closures
    env["is_hot"]  = _mk_is_hot(env)
    env["is_cold"] = _mk_is_cold(env)
    env["is_due"]  = _mk_is_due(env)
    return env

def build_day_env(winners_list: List[str], i: int,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    """History safety env — uses THE SAME H/C/D lists as the UI (no hidden auto)."""
    seed = winners_list[i-1]
    winner = winners_list[i]
    sd = digits_of(seed)
    cd = digits_of(winner)

    prev_seed = winners_list[i-2] if i-2 >= 0 else ""
    prev_prev = winners_list[i-3] if i-3 >= 0 else ""
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev) if prev_prev else []

    prev_pattern = _make_prev_pattern(ppdigs, pdigs, sd)

    cset = set(cd)
    combo_mirror_digits = sorted({MIRROR[d] for d in cd}) if cd else []
    seed_mirror_digits = sorted({MIRROR[d] for d in sd}) if sd else []
    mirror_pairs = {tuple(sorted((d, MIRROR[d]))) for d in cset if MIRROR[d] in cset and MIRROR[d] != d}
    mirror_pair_count = len(mirror_pairs)
    has_mirror_pair = mirror_pair_count > 0
    mirror_overlap_with_seed = len(cset & set(seed_mirror_digits))

    last2 = set(sd) | set(pdigs)
    common_to_both = set(sd) & set(pdigs)

    env = {
        # seed & prevs
        'seed_digits': sd, 'prev_seed_digits': pdigs, 'prev_prev_seed_digits': ppdigs,
        'seed_digits_1': pdigs, 'seed_digits_2': ppdigs, 'seed_digits_3': [],
        'new_seed_digits': list(set(sd) - set(pdigs)), 'seed_counts': Counter(sd),
        'seed_sum': sum(sd), 'prev_sum_cat': sum_category(sum(sd)), 'prev_pattern': tuple(prev_pattern),
        'seed_vtracs': set(VTRAC[d] for d in sd),

        # winner as "combo" for the day
        'combo': winner, 'combo_digits': sorted(cd), 'combo_digits_list': sorted(cd),
        'combo_set': cset, 'combo_sum': sum(cd), 'combo_sum_cat': sum_category(sum(cd)),
        'combo_sum_is_even': (sum(cd) % 2 == 0),
        'combo_vtracs': set(VTRAC[d] for d in cd),
        'combo_structure': classify_structure(cd),
        'combo_last_digit': cd[-1] if cd else None,

        # mirrors/vtrac maps
        'mirror': MIRROR, 'MIRROR': MIRROR, 'VTRAC': VTRAC, 'V_TRAC_GROUPS': V_TRAC_GROUPS,

        # use UI H/C/D here too
        'hot_digits': sorted(set(hot_digits)), 'cold_digits': sorted(set(cold_digits)), 'due_digits': sorted(set(due_digits)),
        'hot': sorted(set(hot_digits)), 'cold': sorted(set(cold_digits)), 'due': sorted(set(due_digits)),

        # sets for is_* helpers
        'hot_set': set(hot_digits), 'cold_set': set(cold_digits), 'due_set': set(due_digits),

        'mirror_of': (lambda d: MIRROR.get(int(d), int(d)) if str(d).isdigit() else d),
        'combo_mirror_digits': combo_mirror_digits, 'seed_mirror_digits': seed_mirror_digits,
        'has_mirror_pair': has_mirror_pair, 'mirror_pairs': mirror_pairs,
        'mirror_pair_count': mirror_pair_count, 'mirror_overlap_with_seed': mirror_overlap_with_seed,

        # convenience
        'common_to_both': common_to_both, 'last2': last2,
        'spread': (max(cd) - min(cd)) if cd else 0, 'seed_spread': (max(sd) - min(sd)) if sd else 0,

        # builtins + helpers
        **SAFE_BUILTINS, 'sum_category': sum_category, 'structure_of': structure_of,
        'safe_digits': safe_digits, 'digit_sum': digit_sum, 'parity_even': parity_even,
        'vtrac_of': vtrac_of, 'vtrac_count': vtrac_count, 'pair_count': pair_count,
        'has_pair': has_pair, 'safe_div': safe_div,
        'digits_intersection': digits_intersection, 'digits_union': digits_union,

        'seed_value': int(seed) if seed.isdigit() else None, 'nan': float('nan'),
        'winner_structure': classify_structure(sd),
    }
    # Late-bind is_* closures
    env['is_hot']  = _mk_is_hot(env)
    env['is_cold'] = _mk_is_cold(env)
    env['is_due']  = _mk_is_due(env)
    return env

# ==============================
# CSV loaders (pool, winners, filters) — tolerant
# ==============================
@st.cache_data(show_spinner=False)
def load_winners_csv_from_path(path_or_buf) -> List[str]:
    df = pd.read_csv(path_or_buf, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}
    if "result" in cols_lower:   s = df[cols_lower["result"]]
    elif "combo" in cols_lower:  s = df[cols_lower["combo"]]
    else:                        return []
    return [str(x).strip() for x in s.dropna().astype(str)]

def load_pool_from_text_or_csv(text: str, col_hint: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    looks_csv = ("," in text and "\n" in text) or text.lower().startswith("result")
    if looks_csv:
        try:
            df = pd.read_csv(io.StringIO(text), engine="python")
            cols_lower = {c.lower(): c for c in df.columns}
            if col_hint and col_hint in df.columns: s = df[col_hint]
            elif "result" in cols_lower:            s = df[cols_lower["result"]]
            elif "combo" in cols_lower:             s = df[cols_lower["combo"]]
            else:                                    s = df[df.columns[0]]
            return [str(x).strip() for x in s.dropna().astype(str)]
        except Exception:
            pass
    return parse_list_any(text)

def load_pool_from_file(f, col_hint: str) -> List[str]:
    df = pd.read_csv(f, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}
    if col_hint and col_hint in df.columns: s = df[col_hint]
    elif "result" in cols_lower:            s = df[cols_lower["result"]]
    elif "combo" in cols_lower:             s = df[cols_lower["combo"]]
    else:                                    s = df[df.columns[0]]
    return [str(x).strip() for x in s.dropna().astype(str)]

def normalize_filters_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    out = df.copy()

    if "id" not in out.columns and "fid" in out.columns:
        out["id"] = out["fid"]
    if "id" not in out.columns:
        out["id"] = range(1, len(out) + 1)

    if "expression" not in out.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")

    def _clean_expr(x):
        s = str(x)
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        if s.startswith('"""') and s.endswith('"""'): s = s[3:-3]
        if s.startswith('"')   and s.endswith('"'):   s = s[1:-1]
        return s.strip()
    out["expression"] = out["expression"].apply(_clean_expr)

    if "name" not in out.columns:
        out["name"] = out["id"].astype(str)

    if "enabled" in out.columns:
        enabled_norm = out["enabled"].apply(lambda v: str(v).strip().lower() in ("1","true","t","yes","y"))
        out = out[enabled_norm].copy()

    if "applicable_if" not in out.columns or out["applicable_if"].isna().all():
        out["applicable_if"] = "True"

    out = out[out["expression"].astype(str).str.len() > 0].copy()
    return out

def load_filters_from_source(pasted_csv_text: str, uploaded_csv_file, csv_path: str) -> pd.DataFrame:
    # If pasted text looks like a real CSV with an expression column, use it.
    if pasted_csv_text and pasted_csv_text.strip():
        try:
            tmp = pd.read_csv(io.StringIO(pasted_csv_text), engine="python")
            if any(c.strip().lower() == "expression" for c in tmp.columns):
                return normalize_filters_df(tmp)
        except Exception:
            pass
    if uploaded_csv_file is not None:
        df = pd.read_csv(uploaded_csv_file, engine="python")
        return normalize_filters_df(df)
    df = pd.read_csv(csv_path, engine="python")
    return normalize_filters_df(df)

# ==============================
# Error helpers for skip reporting
# ==============================
def _missing_name_from_exc(err: Exception) -> Optional[str]:
    m = re.search(r"name '([^']+)' is not defined", str(err))
    if m: return m.group(1)
    return None

def eval_applicable_with_error(applicable_if: str, base_env: Dict, fid: str, name: str, skip_log: List[dict]) -> bool:
    try:
        code = compile(str(applicable_if), "<applicable_if>", "eval")
    except Exception as e:
        skip_log.append({"filter_id": fid, "name": name, "stage": "applicable_if_compile", "error": str(e), "missing": _missing_name_from_exc(e)})
        return True
    try:
        return bool(eval(code, {"__builtins__": {}}, {**SAFE_BUILTINS, **base_env}))
    except Exception as e:
        skip_log.append({"filter_id": fid, "name": name, "stage": "applicable_if_eval", "error": str(e), "missing": _missing_name_from_exc(e)})
        return True

def eval_filter_on_pool_with_errors(row: pd.Series, pool: List[str], base_env: Dict, skip_log: List[dict]) -> Tuple[Set[str], int, int, int]:
    expr = str(row["expression"]); fid = str(row["id"]); name = str(row.get("name",""))
    try:
        code = compile(expr, "<filter_expr>", "eval")
    except Exception as e:
        skip_log.append({"filter_id": fid, "name": name, "stage": "expr_compile", "error": str(e), "missing": _missing_name_from_exc(e)})
        return set(), 0, 0, 0

    eliminated: Set[str] = set()
    elim_even = elim_odd = 0
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, {**SAFE_BUILTINS, **env})):
                eliminated.add(c)
                if env["combo_sum_is_even"]: elim_even += 1
                else:                        elim_odd  += 1
        except Exception as e:
            skip_log.append({"filter_id": fid, "name": name, "stage": "expr_eval", "combo": c, "error": str(e), "missing": _missing_name_from_exc(e)})
    return eliminated, len(eliminated), elim_even, elim_odd

def safety_on_history_with_errors(expr: str, winners_list: List[str], sd_now: List[int],
                                  fid: str, name: str, skip_log: List[dict],
                                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Tuple[Optional[float], int, int]:
    if not winners_list or len(winners_list) < 2:
        return None, 0, 0
    try:
        code = compile(str(expr), "<hist_expr>", "eval")
    except Exception as e:
        skip_log.append({"filter_id": fid, "name": name, "stage": "history_compile", "error": str(e), "missing": _missing_name_from_exc(e)})
        return None, 0, 0
    total_sim, bad_hits = 0, 0
    for i in range(1, len(winners_list)):
        env = build_day_env(winners_list, i, hot_digits, cold_digits, due_digits)
        if not similar_seed(env["seed_digits"], sd_now):
            continue
        total_sim += 1
        try:
            if bool(eval(code, {"__builtins__": {}}, {**SAFE_BUILTINS, **env})):
                bad_hits += 1
        except Exception as e:
            skip_log.append({"filter_id": fid, "name": name, "stage": "history_eval", "day_index": i, "error": str(e), "missing": _missing_name_from_exc(e)})
    if total_sim == 0:
        return None, 0, 0
    safety = 100.0 * (1.0 - bad_hits / total_sim)
    return safety, total_sim, bad_hits

# ==============================
# Similarity & flags
# ==============================
def similar_seed(sd_hist: List[int], sd_now: List[int]) -> bool:
    if not sd_now or not sd_hist:
        return True
    f_hist = seed_features(sd_hist)
    f_now  = seed_features(sd_now)
    match_count = sum(1 for a, b in zip(f_hist, f_now) if a == b and a != "")
    return match_count >= 2

def is_seed_specific(text: str) -> bool:
    if not text: return False
    pattern = r"\b(seed_digits|prev_seed_digits|prev_prev_seed_digits|seed_vtracs|seed_counts|prev_pattern|prev_sum_cat|new_seed_digits)\b"
    return bool(re.search(pattern, text))

# ==============================
# Archetype lifts (optional)
# ==============================
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
        df = pd.read_csv(csv_path, engine="python")
        fid = _first_col(df, ["filter_id","fid","id"])
        dim = _first_col(df, ["dimension","dim","feature","trait_name"])
        val = _first_col(df, ["value","trait","bucket","bin"])
        lift = None
        for c in df.columns:
            if "lift" in c.lower(): lift = c; break
        kept = _first_col(df, ["kept_rate","kept%","kept_pct"])
        base = _first_col(df, ["baseline_kept_rate","baseline_kept%","baseline_pct"])
        supp = _first_col(df, ["applicable_days","support","n","days","app_days"])
        if not (fid and dim and val): return None
        if not lift:
            if kept and base:
                df["__lift_tmp__"] = pd.to_numeric(df[kept], errors="coerce") / pd.to_numeric(df[base], errors="coerce")
                lift = "__lift_tmp__"
            else:
                return None
        use_cols = [fid, dim, val, lift]
        if supp: use_cols.append(supp)
        out = df[use_cols].dropna().copy()
        out.columns = ["filter_id","dimension","value","lift"] + (["support"] if supp else [])
        out["filter_id"] = out["filter_id"].astype(str).str.strip()
        out["dimension"] = out["dimension"].astype(str).str.strip().lower()
        out["value"] = out["value"].astype(str).str.strip()
        out["lift"] = pd.to_numeric(out["lift"], errors="coerce")
        out = out.dropna(subset=["lift"])
        return out
    except Exception:
        return None

def current_traits_for_match(prof: Dict[str, object]) -> Dict[str, List[str]]:
    return {
        "sum_cat": [str(prof.get("sum_cat",""))],
        "sum_category": [str(prof.get("sum_cat",""))],
        "parity": [str(prof.get("parity",""))],
        "parity_major": ["even>=3" if str(prof.get("parity","")).startswith(("3","4","5")) else "even<=2"],
        "structure": [str(prof.get("structure",""))],
        "spread_band": [str(prof.get("spread_band",""))],
        "spread": [str(prof.get("spread_band",""))],
        "hi_lo": [str(prof.get("hi_lo",""))],
        "has_dupe": ["True" if prof.get("has_dupe") else "False", "Yes" if prof.get("has_dupe") else "No"],
    }

def compute_expected_safety_map(df_for_map: pd.DataFrame, arch_df: Optional[pd.DataFrame], prof: Dict[str, object]) -> Dict[str, float]:
    base_map = {}
    for _, r in df_for_map.iterrows():
        fid = str(r["id"])
        kept = r.get("historic_safety_pct")
        base_map[fid] = float(kept)/100.0 if pd.notna(kept) else 0.5
    if arch_df is None or arch_df.empty:
        return {fid: max(0.05, min(0.99, base_map.get(fid, 0.5))) for fid in base_map}
    trait_dict = current_traits_for_match(prof)
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
                L = max(0.7, min(1.3, L))
                lifts.append(L)
        if lifts:
            gm = math.exp(sum(math.log(x) for x in lifts) / len(lifts))
            expected = base * gm
        else:
            expected = base
        out[fid] = max(0.05, min(0.99, expected))
    return out

# ==============================
# Planners
# ==============================
def greedy_plan(candidates: pd.DataFrame, pool: List[str], base_env: Dict, beam_width: int, max_steps: int, mode: str, exp_map_for_greedy: Dict[str, float]) -> Tuple[List[Dict], List[str]]:
    remaining = set(pool); chosen: List[Dict] = []
    for step in range(int(max_steps)):
        if not remaining: break
        scored = []
        for _, r in candidates.iterrows():
            elim, cnt, _, _ = eval_filter_on_pool_with_errors(r, list(remaining), base_env, skip_log=[])
            if cnt <= 0:
                continue
            if mode == "Safe Filter Explorer":
                w = float(exp_map_for_greedy.get(str(r["id"]), 0.5))
                score = cnt * (0.25 + 0.75 * w)
            else:
                score = cnt
            scored.append((score, cnt, elim, r))
        if not scored: break
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_score, best_cnt, best_elim, best_row = scored[0]
        remaining -= best_elim
        chosen.append({
            "id": best_row["id"], "name": best_row.get("name", ""), "expression": best_row["expression"],
            "eliminated_this_step": best_cnt, "remaining_after": len(remaining), "score": best_score
        })
        if best_cnt == 0: break
    return chosen, sorted(list(remaining))

def best_case_plan_no_winner(large_df, E_map, names_map, pool_len, target_max, exp_safety_map: Dict[str, float]):
    P = set(range(pool_len)); used: Set[str] = set(); steps = []
    while len(P) > target_max:
        best = None; best_score = -1.0
        for fid, E in E_map.items():
            if fid in used: continue
            elim_now = len(P & E)
            if elim_now <= 0: continue
            w = float(exp_safety_map.get(fid, 0.5))
            score = elim_now * (0.25 + 0.75 * w)
            if score > best_score:
                best_score = score; best = (fid, elim_now, w)
        if best is None: break
        fid, elim_now, w = best
        P = P - E_map[fid]; used.add(fid)
        steps.append({"filter_id": fid, "name": names_map.get(fid, ""), "eliminated_now": elim_now,
                      "remaining_after": len(P), "expected_safety_%": round(100.0 * w, 2)})
    if steps: return {"steps": steps, "final_pool_idx": P}
    return None

def winner_preserving_plan(large_df, E_map, names_map, pool_len, winner_idx, target_max=45, beam_width=3, max_steps=12):
    P0 = set(range(pool_len))
    if winner_idx is not None and winner_idx not in P0: return None
    large_index = large_df.set_index("id")
    best = {"pool": P0, "steps": []}; seen = {}

    def score_candidate(fid, elim_now):
        kept = large_index.loc[fid]["historic_safety_pct"]; days = large_index.loc[fid]["similar_days"]
        kept01 = (float(kept)/100.0) if pd.notna(kept) else 0.5; days = float(days) if pd.notna(days) else 0.0
        return elim_now * (0.5 + 0.5*kept01) * math.log1p(days)

    def key(P, used): return (len(P), tuple(sorted(used))[:8])

    def dfs(P, used, log, depth):
        nonlocal best
        if winner_idx is not None and winner_idx not in P: return
        if len(P) <= target_max:
            if len(P) < len(best["pool"]) or (len(P) == len(best["pool"]) and len(log) < len(best["steps"])): best = {"pool": set(P), "steps": list(log)}; return
        if depth >= max_steps:
            if len(P) < len(best["pool"]): best = {"pool": set(P), "steps": list(log)}; return
        k = key(P, used)
        if k in seen and seen[k] <= len(P): return
        seen[k] = len(P)

        cands = []
        for fid, E in E_map.items():
            if fid in used: continue
            elim_now = len(P & E)
            if elim_now <= 0: continue
            if winner_idx is not None and winner_idx in E: continue
            cands.append((fid, elim_now, score_candidate(fid, elim_now)))
        if not cands:
            if len(P) < len(best["pool"]): best = {"pool": set(P), "steps": list(log)}
            return
        cands.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for fid, elim_now, _sc in cands[:beam_width]:
            newP = P - E_map[fid]
            step = {"filter_id": fid, "name": names_map.get(str(fid), ""), "eliminated_now": elim_now, "remaining_after": len(newP)}
            dfs(newP, used | {fid}, log + [step], depth+1)

    dfs(P0, set(), [], 0)
    if best["steps"] and len(best["pool"]) < len(P0): return {"steps": best["steps"], "final_pool_idx": best["pool"]}
    return None

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Mode & Thresholds")
    mode = st.radio("Mode", ["Playlist Reducer", "Safe Filter Explorer"], index=1)
    if mode == "Playlist Reducer":
        default_min_elims = 120; default_beam = 5; default_steps = 15
    else:
        default_min_elims = 60; default_beam = 6; default_steps = 18
    min_elims  = st.number_input("Min eliminations to call it ‘Large’", 1, 99999, value=default_min_elims, step=1)
    beam_width = st.number_input("Greedy beam width", 1, 50, value=default_beam, step=1)
    max_steps  = st.number_input("Greedy max steps", 1, 50, value=default_steps, step=1)
    exclude_parity = st.checkbox("Exclude parity-wipers", value=True)

    st.markdown("---")
    st.header("Archetype Lifts (optional)")
    use_archetype_lifts = st.checkbox("Use archetype-lift CSV if present", value=True)
    arch_path = st.text_input("Archetype-lifts CSV path", value="archetype_filter_dimension_stats.csv")

    st.markdown("---")
    st.header("Planners")
    known_winner = st.text_input("Known winner (5 digits, optional)", value="").strip()
    target_max_bc = st.number_input("Target kept (best-case)", min_value=5, max_value=200, value=45, step=1)
    target_max_wp = st.number_input("Target kept (winner-preserving)", min_value=5, max_value=200, value=45, step=1)
    beam_wp = st.number_input("WP beam width", min_value=1, max_value=20, value=3, step=1)
    steps_wp = st.number_input("WP max steps", min_value=1, max_value=50, value=12, step=1)

# ==============================
# Seed context & Hot/Cold/Due
# ==============================
st.subheader("Seed Context")
c1, c2, c3 = st.columns(3)
seed      = c1.text_input("Seed (prev draw, 5 digits expected but not enforced)", value="")
prev_seed = c2.text_input("Prev Seed (2-back, optional)", value="")
prev_prev = c3.text_input("Prev Prev Seed (3-back, optional)", value="")

st.subheader("Hot / Cold / Due digits")
d1, d2, d3 = st.columns(3)
hot_digits  = [int(x) for x in parse_list_any(d1.text_input("Hot digits (comma-separated)")) if x.isdigit()]
cold_digits = [int(x) for x in parse_list_any(d2.text_input("Cold digits (comma-separated)")) if x.isdigit()]
due_digits  = [int(x) for x in parse_list_any(d3.text_input("Due digits (comma-separated)")) if x.isdigit()]
auto_hcd = st.checkbox("Only if ALL boxes are blank: auto-fill Hot/Cold/Due from recent history", value=True)

# ==============================
# Combo Pool
# ==============================
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV w/ 'Result' column OR tokens separated by newline/space/comma):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV ('Result' or 'combo' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")

pool: List[str] = []
if pool_text.strip():
    try:
        pool = load_pool_from_text_or_csv(pool_text, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to parse pasted pool ➜ {e}"); st.stop()
elif pool_file is not None:
    try:
        pool = load_pool_from_file(pool_file, pool_col_hint)
    except Exception as e:
        st.error(f"Failed to load pool CSV ➜ {e}"); st.stop()
else:
    st.info("Paste combos or upload a pool CSV to continue."); st.stop()

pool = [p for p in pool if p]
st.write(f"**Pool size:** {len(pool)}")

# ==============================
# Winners history CSV
# ==============================
st.subheader("Winners History")
hc1, hc2 = st.columns([2, 1])
history_path = hc1.text_input("Path to winners history CSV", value="DC5_Midday_Full_Cleaned_Expanded.csv")
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

# H/C/D linkage guarantee:
user_provided_hcd = bool(hot_digits or cold_digits or due_digits)
if not user_provided_hcd and auto_hcd and winners_list:
    AUTO_WINDOW = 10
    hist_digits = [digits_of(x) for x in winners_list][-AUTO_WINDOW:]
    h, c, d = hot_cold_due(hist_digits, k=AUTO_WINDOW)
    hot_digits, cold_digits, due_digits = sorted(h), sorted(c), sorted(d)
    st.caption(f"(Auto) Hot/Cold/Due from last {AUTO_WINDOW} winners → hot={hot_digits} • cold={cold_digits} • due={due_digits}")
else:
    st.caption(f"(Using UI) hot={hot_digits} • cold={cold_digits} • due={due_digits}")

sd_now = digits_of(seed)

# ==============================
# Filters: IDs + CSV (pasted / upload / path)
# ==============================
st.subheader("Filters")
fids_text = st.text_area("Paste applicable Filter IDs (optional; comma / space / newline separated):", height=90)
filters_pasted_csv = st.text_area("Paste Filters CSV content (optional):", height=150, help="If provided, this CSV is used (must include 'expression').")
filters_file_up = st.file_uploader("Or upload Filters CSV (used if pasted CSV is empty)", type=["csv"])
filters_csv_path = st.text_input("Or path to Filters CSV (used if pasted/upload empty)", value="lottery_filters_batch_10.csv")

try:
    filters_df_full = load_filters_from_source(filters_pasted_csv, filters_file_up, filters_csv_path)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}"); st.stop()

# If the "CSV content" box actually contains just IDs, include them too:
if filters_pasted_csv and "expression" not in filters_pasted_csv.split("\n", 1)[0].lower():
    fids_text = (fids_text + "," + filters_pasted_csv) if fids_text else filters_pasted_csv

applicable_ids = set(parse_list_any(fids_text))
if applicable_ids:
    id_str   = filters_df_full["id"].astype(str)
    name_str = filters_df_full.get("name","").astype(str)
    mask = id_str.isin(applicable_ids) | name_str.isin(applicable_ids)
    filters_df = filters_df_full[mask].copy()
else:
    filters_df = filters_df_full.copy()

if "enabled" in filters_df.columns:
    enabled_norm = filters_df["enabled"].apply(lambda v: str(v).strip().lower() in ("1","true","t","yes","y"))
    filters_df = filters_df[enabled_norm].copy()

st.write(f"**Filters loaded (pre-eval): {len(filters_df)}**")
if len(filters_df) == 0:
    st.warning("0 filters available. Check the CSV path/content, 'enabled' values, expression quoting, or your ID selection.")
    st.stop()

# ==============================
# RUN
# ==============================
run = st.button("▶ Run Planner + Recommender", type="primary", use_container_width=False)
if not run:
    st.stop()

# ==============================
# Build base env & evaluate
# ==============================
base_env = make_base_env(seed, prev_seed, prev_prev, hot_digits, cold_digits, due_digits)

st.info(
    f"Using Hot/Cold/Due in ALL evaluations → hot={hot_digits} • cold={cold_digits} • due={due_digits}",
    icon="🔥"
)

st.subheader("Evaluating filters on current pool…")
rows = []; E_map: Dict[str, Set[int]] = {}; names_map: Dict[str, str] = {}
pool_digits = [digits_of(s) for s in pool]
total_even = sum(1 for cd in pool_digits if (sum(cd) % 2) == 0); total_odd = len(pool_digits) - total_even
skip_log: List[dict] = []

for _, r in filters_df.iterrows():
    fid = str(r["id"]); nm = str(r.get("name",""))
    if not eval_applicable_with_error(r.get("applicable_if", "True"), base_env, fid, nm, skip_log):
        continue
    elim_set, cnt, elim_even, elim_odd = eval_filter_on_pool_with_errors(r, pool, base_env, skip_log)
    names_map[fid] = nm; E_map[fid] = {pool.index(c) for c in elim_set}
    parity_wiper = ((elim_even == total_even and total_even > 0) or (elim_odd == total_odd and total_odd > 0))
    text_blob = f"{r.get('applicable_if','')} || {r.get('expression','')}"
    seed_specific = is_seed_specific(text_blob)
    s_pct, sims, bad = safety_on_history_with_errors(r["expression"], winners_list, sd_now, fid, nm, skip_log, hot_digits, cold_digits, due_digits)
    rows.append({
        "id": r["id"], "name": nm, "expression": r["expression"],
        "elim_count_on_pool": cnt, "elim_pct_on_pool": (cnt / len(pool) * 100.0) if pool else 0.0,
        "elim_even": elim_even, "elim_odd": elim_odd, "parity_wiper": parity_wiper,
        "seed_specific_trigger": seed_specific, "historic_safety_pct": None if s_pct is None else round(s_pct, 2),
        "similar_days": sims, "bad_hits": bad,
    })

scored_df = pd.DataFrame(rows)
if scored_df.empty:
    st.warning("No filters evaluated (empty)."); st.stop()

# ==============================
# Skip report
# ==============================
if skip_log:
    st.subheader("⚠️ Skipped / Errored filters")
    skip_df = pd.DataFrame(skip_log)
    agg = (skip_df.groupby(["filter_id","name","stage"])["error"]
           .count().reset_index().rename(columns={"error":"count"})
           .sort_values(["count","filter_id"], ascending=[False, True]))
    st.dataframe(agg, use_container_width=True, height=min(340, 60 + 28*len(agg)))
    st.download_button("Download full skip log (CSV)", skip_df.to_csv(index=False), "filter_skip_log.csv", "text/csv")
else:
    st.info("No evaluation errors detected while compiling/evaluating filters.")

# ==============================
# Expected safety + recommendations
# ==============================
arch_df = load_archetype_dimension_lifts(Path(arch_path)) if use_archetype_lifts else None

# Seed profile for archetype traits (optional)
if seed and seed.isdigit() and len(seed) == 5:
    prof = seed_profile(seed, prev_seed, prev_prev)
else:
    prof = {"sum_cat":"", "parity":"", "structure":"", "spread_band":""}

def _expected_map_for(df):
    if df.empty: return {}
    return compute_expected_safety_map(df.rename(columns={"id": "id"}), arch_df, prof)

exp_safety_map_all = _expected_map_for(scored_df)

def attach_expected_safety(df):
    if df.empty: return df
    out = df.copy()
    out["expected_safety_pct"] = out["id"].astype(str).map(
        lambda fid: round(100.0 * float(exp_safety_map_all.get(str(fid), 0.5)), 2)
    )
    return out

scored_df = attach_expected_safety(scored_df)

# ==============================
# Candidate “Large” + Recommendations
# ==============================
large_df = scored_df[scored_df["elim_count_on_pool"] >= int(min_elims)].copy()
if exclude_parity and "parity_wiper" in large_df.columns:
    large_df = large_df[~large_df["parity_wiper"]].copy()

if "expected_safety_pct" not in large_df.columns:
    large_df = attach_expected_safety(large_df)

def rec_score(row):
    w = (row["expected_safety_pct"] or 50.0) / 100.0
    return row["elim_count_on_pool"] * (0.25 + 0.75*w)

large_df["recommend_score"] = large_df.apply(rec_score, axis=1)
large_df = large_df.sort_values(
    by=["recommend_score", "expected_safety_pct", "elim_count_on_pool"],
    ascending=[False, False, False]
)

st.write(f"**Large filters (≥ {min_elims} eliminated):** {len(large_df)}")
st.dataframe(
    large_df[[
        "id","name","elim_count_on_pool","elim_pct_on_pool",
        "historic_safety_pct","expected_safety_pct",
        "similar_days","bad_hits","parity_wiper","recommend_score"
    ]],
    use_container_width=True
)

st.subheader("Recommended • Safe but Effective")
rec_df = large_df.copy().sort_values(by=["recommend_score"], ascending=False)
st.dataframe(
    rec_df[["id","name","elim_count_on_pool","historic_safety_pct","expected_safety_pct","recommend_score"]],
    use_container_width=True, height=min(360, 60 + 28*len(rec_df))
)

# ==============================
# Greedy planning (fast)
# ==============================
st.subheader("Planner (greedy)")
if large_df.empty:
    st.info("No candidates meet the 'Large' threshold; nothing to plan.")
    kept_after = pool
    plan = []
else:
    candidates = large_df[["id", "name", "expression"]].copy()
    exp_map_for_greedy = {str(r["id"]): (r.get("expected_safety_pct") or 50.0)/100.0 for _, r in large_df.iterrows()}
    plan, kept_after = greedy_plan(
        candidates=candidates,
        pool=pool,
        base_env=base_env,
        beam_width=int(beam_width),
        max_steps=int(max_steps),
        mode=mode,
        exp_map_for_greedy=exp_map_for_greedy
    )

st.write(f"**Kept combos after greedy plan:** {len(kept_after)} / {len(pool)}")
if plan:
    plan_df = pd.DataFrame(plan)
    plan_df["expected_safety_pct"] = plan_df["id"].astype(str).map(
        lambda fid: round(100.0 * float(exp_safety_map_all.get(str(fid), 0.5)), 2)
    )
    st.write("**Chosen sequence (in order):**")
    st.dataframe(
        plan_df[["id","name","expression","eliminated_this_step","remaining_after","expected_safety_pct","score"]],
        use_container_width=True
    )

# ==============================
# Best-case & Winner-preserving planners
# ==============================
st.subheader("Best-case plan — Large filters only")
if large_df.empty:
    st.info("No Large filters available for best-case planning.")
else:
    E_sub = {str(fid): {i for i in E_map.get(str(fid), set())} for fid in large_df["id"].astype(str)}
    pool_len = len(pool)
    names_sub = {str(fid): names_map.get(str(fid), "") for fid in E_sub.keys()}
    plan_best = best_case_plan_no_winner(
        large_df=large_df.rename(columns={"id":"id"}),
        E_map=E_sub,
        names_map=names_sub,
        pool_len=pool_len,
        target_max=int(target_max_bc),
        exp_safety_map={fid: float(exp_safety_map_all.get(fid, 0.5)) for fid in E_sub.keys()}
    )
    if plan_best is None:
        st.info("Best-case plan could not reach the target with available Large filters.")
    else:
        bc_df = pd.DataFrame(plan_best["steps"])
        st.dataframe(bc_df, use_container_width=True, hide_index=True, height=min(320, 60 + 28*len(bc_df)))
        kept_idx = sorted(list(plan_best["final_pool_idx"]))
        kept_combos_bc = [pool[i] for i in kept_idx]
        st.caption(f"Best-case final kept pool size: {len(kept_combos_bc)}")
        st.dataframe(pd.DataFrame({"Result": kept_combos_bc}).head(100), use_container_width=True, height=260)

st.subheader("Winner-preserving plan — Large filters only")
if not known_winner:
    st.info("Provide a 5-digit **Known winner** in the sidebar to compute a winner-preserving plan.")
else:
    if len(known_winner) != 5 or not known_winner.isdigit():
        st.warning("Known winner must be exactly 5 digits.")
    else:
        try:
            winner_idx = pool.index(known_winner)
        except ValueError:
            winner_idx = None
        if winner_idx is None or large_df.empty:
            st.info("Winner not found in pool, or no Large filters.")
        else:
            E_sub = {str(fid): {i for i in E_map.get(str(fid), set())} for fid in large_df["id"].astype(str)}
            names_sub = {str(fid): names_map.get(str(fid), "") for fid in E_sub.keys()}
            plan_wp = winner_preserving_plan(
                large_df=large_df.rename(columns={"id":"id"}),
                E_map=E_sub,
                names_map=names_sub,
                pool_len=len(pool),
                winner_idx=winner_idx,
                target_max=int(target_max_wp),
                beam_width=int(beam_wp),
                max_steps=int(steps_wp)
            )
            if plan_wp is None:
                st.info("No winner-preserving reduction possible with the available Large filters.")
            else:
                wp_df = pd.DataFrame(plan_wp["steps"])
                wp_df["expected_safety_pct"] = wp_df["filter_id"].astype(str).map(
                    lambda fid: round(100.0 * float(exp_safety_map_all.get(str(fid), 0.5)), 2)
                )
                st.dataframe(
                    wp_df[["filter_id","name","eliminated_now","remaining_after","expected_safety_pct"]],
                    use_container_width=True, hide_index=True, height=min(320, 60 + 28*len(wp_df))
                )
                kept_idx = sorted(list(plan_wp["final_pool_idx"]))
                kept_combos_wp = [pool[i] for i in kept_idx]
                st.caption(f"Winner-preserving final kept pool size: {len(kept_combos_wp)}")
                st.dataframe(pd.DataFrame({"Result": kept_combos_wp}).head(100), use_container_width=True, height=260)

# ==============================
# Downloads
# ==============================
st.subheader("Downloads")
kept_df = pd.DataFrame({"Result": kept_after}) if 'kept_after' in locals() else pd.DataFrame({"Result": pool})
removed = sorted(set(pool) - set(kept_df["Result"].tolist()))
removed_df = pd.DataFrame({"Result": removed})

cA, cB = st.columns(2)
cA.download_button("Download KEPT combos (CSV)", kept_df.to_csv(index=False), file_name="kept_combos.csv", mime="text/csv")
cA.download_button("Download KEPT combos (TXT)", "\n".join(kept_df['Result'].tolist()), file_name="kept_combos.txt", mime="text/plain")
cB.download_button("Download REMOVED combos (CSV)", removed_df.to_csv(index=False), file_name="removed_combos.csv", mime="text/csv")
cB.download_button("Download REMOVED combos (TXT)", "\n".join(removed), file_name="removed_combos.txt", mime="text/plain")

with st.expander("Column guide", expanded=False):
    st.markdown(
        """
- **elim_count_on_pool** – Eliminations on your provided pool (aggression metric).
- **elim_even / elim_odd** – Split of eliminations; helps spot parity-wipers.
- **parity_wiper** – True if a filter wipes all evens or all odds in your pool (optionally excluded).
- **seed_specific_trigger** – Filter references seed/prev-seed signals (seed_digits, seed_vtracs, prev_pattern, etc.).
- **historic_safety_pct** – On similar seeds in history, % of days the filter **kept** the true winner (higher = safer).
- **expected_safety_pct** – Safety adjusted for current seed archetype (or 50% baseline if no history).
- **recommend_score** – “Safe but effective” score = eliminations × expected safety.
- **Best-case plan** – Greedy order of Large filters to reach ≤ target using expected safety.
- **Winner-preserving plan** – Beam-search order of Large filters that keeps a known winner (for backtests).
        """
    )
