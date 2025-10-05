# large_filters_planner_full.py
# Streamlit app — single file version with baked-in alias rules + tester-style normalization
# Run: streamlit run large_filters_planner_full.py

from __future__ import annotations

import io
import math
import re
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Large Filters Planner — FULL", layout="wide")
st.title("Large Filters Planner — FULL (compat mode)")

# ----------------------------
# Constants: VTRAC / MIRROR maps (standard v4)
# ----------------------------
VTRAC: Dict[int, int]  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
V_TRAC_GROUPS: Dict[int, int] = VTRAC
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
mirror = MIRROR  # alias

# ----------------------------
# Safe eval sandbox
# ----------------------------
SAFE_BUILTINS = {
    "abs": abs, "int": int, "str": str, "float": float, "round": round,
    "len": len, "sum": sum, "max": max, "min": min, "any": any, "all": all,
    "set": set, "sorted": sorted, "list": list, "tuple": tuple, "dict": dict,
    "range": range, "enumerate": enumerate, "zip": zip,
    "Counter": Counter, "True": True, "False": False, "None": None,
}

def safe_eval(expr: str, env: Dict) -> bool:
    """Evaluate an expression safely; any non-True result is treated as False."""
    try:
        val = eval(expr, {"__builtins__": {}}, {**SAFE_BUILTINS, **env})
        return bool(val)
    except Exception:
        return False

# ----------------------------
# Utilities
# ----------------------------
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def camel_to_snake(s: str) -> str:
    return _CAMEL_RE.sub('_', s).lower()

def ascii_sanitize(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.replace('“','"').replace('”','"').replace("’","'").replace("‘","'")
    s = s.replace("–","-").replace("—","-")
    return s

def digits_of(x) -> List[int]:
    return [int(ch) for ch in str(x) if ch.isdigit()]

def digit_sum(x) -> int:
    ds = digits_of(x); return sum(ds) if ds else 0

def even_count(x): return sum(1 for d in digits_of(x) if d % 2 == 0)
def odd_count(x):  return sum(1 for d in digits_of(x) if d % 2 == 1)
def high_count(x): return sum(1 for d in digits_of(x) if d >= 5)
def low_count(x):  return sum(1 for d in digits_of(x) if d <= 4)

def first_digit(x): ds = digits_of(x); return ds[0] if ds else None
def last_digit(x):  ds = digits_of(x); return ds[-1] if ds else None
def last_two_digits(x): ds = digits_of(x); return ds[-2:] if len(ds) >= 2 else ds[:]

def structure_of(digs: List[int]) -> str:
    c = Counter(digs)
    counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def spread_band(spread: int) -> str:
    if spread <= 3: return "0–3"
    if spread <= 5: return "4–5"
    if spread <= 7: return "6–7"
    if spread <= 9: return "8–9"
    return "10+"

def percent_hot(x, hot_set):  ds = digits_of(x); return (100.0 * sum(1 for d in ds if d in hot_set)  / len(ds)) if ds else 0.0
def percent_cold(x, cold_set):ds = digits_of(x); return (100.0 * sum(1 for d in ds if d in cold_set) / len(ds)) if ds else 0.0
def percent_due(x, due_set):  ds = digits_of(x); return (100.0 * sum(1 for d in ds if d in due_set)  / len(ds)) if ds else 0.0

def contains_mirror_pair(x):
    s = set(digits_of(x))
    return any((d in s and MIRROR[d] in s and MIRROR[d] != d) for d in s)

def vtrac_group_of_digit(d: int) -> Optional[int]:
    try: return VTRAC[int(d)]
    except Exception: return None

def vtrac_digits(x) -> List[int]:
    return [VTRAC[d] for d in digits_of(x) if d in VTRAC]

def vtrac_hist(x) -> Dict[int, int]:
    return dict(Counter(vtrac_digits(x)))

def count_vtrac_groups(x) -> int:
    return len(set(vtrac_digits(x)))

def has_vtrac_group(x, g) -> bool:
    try: g = int(g)
    except Exception: return False
    return g in set(vtrac_digits(x))

def last_vtrac(x) -> Optional[int]:
    ds = digits_of(x)
    return VTRAC[ds[-1]] if ds else None

def parse_list_any(text: str) -> List[str]:
    if not text: return []
    raw = text.replace("\t", ",").replace("\n", ",").replace(";", ",")
    return [p.strip() for p in raw.split(",") if p.strip()]

# ----------------------------
# Tester-style expression normalization
# ----------------------------
def normalize_expr(expr: str) -> str:
    """Make expressions robust to casing, spacing and common shorthands (tester parity)."""
    if not expr: return ""
    x = ascii_sanitize(expr)

    # selective camelCase -> snake_case for known tokens
    tokens = [
        "hotDigits","coldDigits","dueDigits","mirrorPairs","mirrorDigits","seedDigits",
        "comboDigits","percentHot","percentCold","percentDue","evenCount","oddCount",
        "highCount","lowCount","firstDigit","lastDigit","lastTwo","last2",
        "digitSum","sumDigits","vtracSet","vtracLast","vtracGroups","spreadBand",
        "comboSum","comboSet","seedSet","isEven","isOdd","parityEven"
    ]
    for t in tokens:
        if t in x:
            x = x.replace(t, camel_to_snake(t))

    # Common shorthand rewrites
    x = re.sub(r'\blast2\b', 'last_two_digits(combo_digits)', x)
    x = re.sub(r'(?<!\w)sum(?!\s*\()', 'digit_sum(combo_digits)', x)
    x = re.sub(r'\bdigit_sum\(\s*combo_digits\s*\)', 'digit_sum(combo_digits)', x)  # idempotent
    x = x.replace('parity_even', 'combo_sum_is_even').replace('is_even', 'combo_sum_is_even')
    x = re.sub(r'\bis_odd\b', 'not combo_sum_is_even', x)
    x = re.sub(r'\bstruct(ure)?\b', 'combo_structure', x)
    x = re.sub(r'\bspread_band\(\s*spread\s*\)', 'spread_band(spread)', x)
    x = re.sub(r'\bvtrac_last\b', 'combo_last_vtrac', x)
    return x

# ----------------------------
# AUTO ALIAS RULES (baked)
# Map many free-form names to canonical env variables or helper calls (as strings)
# If a value looks like a function call (contains "("), we resolve at eval time.
# ----------------------------
AUTO_ALIAS_RULES: Dict[str, str] = {
    # hot/cold/due
    "hotnumbers": "hot_digits", "hotdigits": "hot_digits", "hot": "hot_digits", "hot_set": "hot_set",
    "coldnumbers": "cold_digits","colddigits": "cold_digits","cold": "cold_digits","cold_set": "cold_set",
    "duenumbers": "due_digits","duedigits": "due_digits","due": "due_digits","due_set": "due_set",
    "counthot": "sum(1 for d in combo_digits if d in hot_set)",
    "countcold":"sum(1 for d in combo_digits if d in cold_set)",
    "countdue": "sum(1 for d in combo_digits if d in due_set)",
    "percenthot": "percent_hot(combo_digits, hot_set)",
    "percentcold":"percent_cold(combo_digits, cold_set)",
    "percentdue": "percent_due(combo_digits, due_set)",

    # mirror
    "mirrordigits": "combo_mirror_digits", "mirrors": "combo_mirror_digits", "mirrorset": "set(combo_mirror_digits)",
    "mirrorpairs": "contains_mirror_pair(combo_digits)",
    "hasmirrorpair": "contains_mirror_pair(combo_digits)", "hasmirror": "contains_mirror_pair(combo_digits)",
    "seedmirrordigits": "seed_mirror_digits",

    # combo/seed
    "combodigits": "combo_digits", "comboset": "set(combo_digits)", "combo": "combo",
    "seeddigits": "seed_digits", "seedset": "set(seed_digits)",

    # parity & counts
    "parity": "combo_sum_is_even", "parityeven": "combo_sum_is_even",
    "iseven": "combo_sum_is_even", "isodd": "not combo_sum_is_even",
    "evencount": "even_count(combo_digits)", "oddcount": "odd_count(combo_digits)",
    "highcount": "high_count(combo_digits)", "lowcount": "low_count(combo_digits)",
    "isallhigh": "all(d >= 5 for d in combo_digits)", "isalllow": "all(d <= 4 for d in combo_digits)",

    # positionals
    "firstdigit": "first_digit(combo_digits)", "first": "first_digit(combo_digits)",
    "lastdigit": "last_digit(combo_digits)", "last": "last_digit(combo_digits)",
    "lasttwo": "last_two_digits(combo_digits)", "last2": "last_two_digits(combo_digits)",

    # sums/spreads/structure
    "sumdigits": "digit_sum(combo_digits)", "digitsum": "digit_sum(combo_digits)", "sum": "digit_sum(combo_digits)",
    "spreadband": "spread_band(spread)", "spread": "spread",
    "structure": "combo_structure", "struct": "combo_structure",

    # vtrac
    "vtrack": "VTRAC", "vtracks": "VTRAC", "vtracgroups": "VTRAC",
    "vtracset": "combo_vtracs", "combovtracs": "combo_vtracs", "seedvtracs": "seed_vtracs",
    "vtraclast": "combo_last_vtrac", "lastvtrac": "combo_last_vtrac",
}

def _norm_ident(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    return s.lower()

def apply_auto_alias(name: str, env: Dict) -> Tuple[bool, Optional[str]]:
    """
    If 'name' is a free symbol (identifier) and present in AUTO_ALIAS_RULES,
    return (True, mapped_expression_string).
    """
    key = _norm_ident(name)
    if key in AUTO_ALIAS_RULES:
        return True, AUTO_ALIAS_RULES[key]
    if key.endswith("s") and key[:-1] in AUTO_ALIAS_RULES:
        return True, AUTO_ALIAS_RULES[key[:-1]]
    if key.startswith("is_") and key[3:] in AUTO_ALIAS_RULES:
        return True, AUTO_ALIAS_RULES[key[3:]]
    if key.startswith("has_") and key[4:] in AUTO_ALIAS_RULES:
        return True, AUTO_ALIAS_RULES[key[4:]]
    return False, None

# ----------------------------
# Per-combo evaluation env (compat mode)
# ----------------------------
def inject_vtrac_aliases(env: Dict) -> Dict:
    vtrac_map = VTRAC
    sd = env.get("seed_digits", [])
    cd = env.get("combo_digits", [])
    seed_vtracs_set = set(VTRAC[d] for d in sd) if sd else set()
    combo_vtracs_set = set(VTRAC[d] for d in cd) if cd else set()
    env["seed_last_vtrac"] = last_vtrac(sd) if sd else None
    env["combo_last_vtrac"] = last_vtrac(cd) if cd else None

    # dictionary + function aliases
    env.update({
        "VTRAC": vtrac_map, "V_TRAC_GROUPS": vtrac_map, "vtrac": vtrac_map,
        "vtrac_of": vtrac_group_of_digit, "get_vtrac": vtrac_group_of_digit,
        "vtrac_digits": vtrac_digits, "vtrac_hist": vtrac_hist,
        "vtrac_count": count_vtrac_groups, "count_vtrac_groups": count_vtrac_groups,
        "has_vtrac_group": has_vtrac_group, "last_vtrac": last_vtrac,
        "seed_vtracs": seed_vtracs_set, "combo_vtracs": combo_vtracs_set,
    })
    return env

def combo_env_compat(base_env: dict, combo: str) -> dict:
    cd = digits_of(combo)
    cs = set(cd)
    combo_sum = sum(cd) if cd else 0
    combo_sum_is_even = (combo_sum % 2 == 0)
    spread = (max(cd) - min(cd)) if cd else 0
    structure = structure_of(cd) if cd else "single"
    last_d = cd[-1] if cd else None

    combo_mirror_digits = [MIRROR[d] for d in cd] if cd else []
    combo_vtracs = set(VTRAC[d] for d in cd) if cd else set()
    combo_last_vtrac = (VTRAC[last_d] if last_d is not None else None)

    # from base_env / UI
    hot_digits = list(base_env.get("hot_digits", []))
    cold_digits = list(base_env.get("cold_digits", []))
    due_digits = list(base_env.get("due_digits", []))
    hot_set = set(hot_digits); cold_set = set(cold_digits); due_set = set(due_digits)

    env = dict(base_env)
    env.update({
        "combo": combo, "combo_digits": cd, "combo_set": cs,
        "combo_sum": combo_sum, "combo_sum_is_even": combo_sum_is_even,
        "spread": spread, "combo_structure": structure, "combo_last_digit": last_d,
        "combo_mirror_digits": combo_mirror_digits,
        "combo_vtracs": combo_vtracs, "combo_last_vtrac": combo_last_vtrac,
        "hot_digits": hot_digits, "cold_digits": cold_digits, "due_digits": due_digits,
        "hot_set": hot_set, "cold_set": cold_set, "due_set": due_set,
        # helpers exposed
        "digit_sum": digit_sum, "even_count": even_count, "odd_count": odd_count,
        "high_count": high_count, "low_count": low_count,
        "first_digit": first_digit, "last_digit": last_digit, "last_two_digits": last_two_digits,
        "structure_of": structure_of, "spread_band": spread_band,
        "percent_hot": lambda x: percent_hot(x, hot_set),
        "percent_cold": lambda x: percent_cold(x, cold_set),
        "percent_due": lambda x: percent_due(x, due_set),
        "contains_mirror_pair": contains_mirror_pair,
    })
    env = inject_vtrac_aliases(env)

    # populate simple alias names with direct values (non-calls)
    for k, rhs in AUTO_ALIAS_RULES.items():
        if "(" not in rhs and ")" not in rhs:
            env.setdefault(k, env.get(rhs, None))

    # legacy short names
    env.setdefault("hot", hot_digits)
    env.setdefault("cold", cold_digits)
    env.setdefault("due", due_digits)
    return env

# ----------------------------
# UI — Inputs
# ----------------------------
with st.sidebar:
    st.header("Hot/Cold/Due (optional)")
    hot_input = st.text_input("Hot digits (comma-separated)", "")
    cold_input = st.text_input("Cold digits (comma-separated)", "")
    due_input = st.text_input("Due digits (comma-separated)", "")

    st.header("Options")
    mode_keep = st.selectbox("Generation Method", ["1-digit", "2-digit", "plain"], index=0)
    hide_zero_elims = st.checkbox("Hide filters with 0 initial eliminations", value=True)

def parse_digits_field(s: str) -> List[int]:
    digs = []
    for tok in parse_list_any(s):
        if tok.isdigit():
            digs.append(int(tok))
    return digs

hot_digits = parse_digits_field(hot_input)
cold_digits = parse_digits_field(cold_input)
due_digits = parse_digits_field(due_input)

base_env = {
    "hot_digits": hot_digits, "cold_digits": cold_digits, "due_digits": due_digits,
    "hot_set": set(hot_digits), "cold_set": set(cold_digits), "due_set": set(due_digits),
    "MIRROR": MIRROR, "mirror": MIRROR, "VTRAC": VTRAC, "V_TRAC_GROUPS": V_TRAC_GROUPS,
}

st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV with 'Result' or free tokens separated by newline/space/comma)", height=120)
pool_file = st.file_uploader("Or upload combo pool CSV (Result/combo column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", "Result")

def load_pool_from_text(t: str) -> List[str]:
    tokens = parse_list_any(t)
    return tokens

def load_pool_from_csv(data: bytes, col_hint: str) -> List[str]:
    df = pd.read_csv(io.BytesIO(data))
    col = None
    for c in df.columns:
        lc = c.lower()
        if lc == col_hint.lower() or lc in ("result","combo"):
            col = c; break
    if col is None and len(df.columns) >= 1:
        col = df.columns[0]
    return [str(x) for x in df[col].dropna().astype(str).tolist()]

pool: List[str] = []
if pool_text.strip():
    pool = load_pool_from_text(pool_text)
elif pool_file is not None:
    pool = load_pool_from_csv(pool_file.read(), pool_col_hint)

st.caption(f"Loaded pool combos: {len(pool)}")

st.subheader("Filters")
filters_text = st.text_area("Paste Filters CSV content (id,expression[,name][,applicable_if][,enabled])", height=160)
filters_file = st.file_uploader("Or upload Filters CSV", type=["csv"])
filters_path = st.text_input("Or path to Filters CSV (server/local)", "")

def load_filters_from_csv_bytes(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    return df

def canonicalize_filters(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id","expression"}
    low_cols = {c.lower(): c for c in df.columns}
    # normalize column names by lower-casing known keys
    for want in ["id","name","expression","applicable_if","enabled"]:
        if want not in df.columns and want in low_cols:
            df.rename(columns={low_cols[want]: want}, inplace=True)

    if "id" not in df.columns:   df["id"] = range(1, len(df)+1)
    if "name" not in df.columns: df["name"] = df["id"].astype(str)
    if "expression" not in df.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    if "applicable_if" not in df.columns: df["applicable_if"] = "True"
    if "enabled" not in df.columns: df["enabled"] = True

    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["expression"] = df["expression"].astype(str)
    df["applicable_if"] = df["applicable_if"].fillna("True").astype(str)
    df["enabled"] = df["enabled"].fillna(True).astype(bool)
    return df

filters_df: Optional[pd.DataFrame] = None
if filters_text.strip():
    filters_df = canonicalize_filters(pd.read_csv(io.StringIO(filters_text)))
elif filters_file is not None:
    filters_df = canonicalize_filters(load_filters_from_csv_bytes(filters_file.read()))
elif filters_path.strip():
    p = Path(filters_path)
    filters_df = canonicalize_filters(pd.read_csv(p))

if filters_df is None:
    st.info("Paste filters CSV or upload a file to continue.")
    st.stop()

# ----------------------------
# Run Planner
# ----------------------------
run = st.button("Run Planner + Recommender", type="primary", use_container_width=True)
if not run:
    st.stop()

# normalize expressions once
filters_df = filters_df.copy()
filters_df["expr_norm"] = filters_df["expression"].apply(normalize_expr)
filters_df["app_norm"]  = filters_df["applicable_if"].apply(normalize_expr)

skip_rows = []
elim_counts = []
remaining_after = len(pool)

# Evaluate each filter against the pool
current_pool = pool[:]

for _, row in filters_df.iterrows():
    fid = str(row["id"]); fname = row["name"]
    expr = row["expr_norm"]
    app_if = row["app_norm"]
    enabled = bool(row["enabled"])

    if not enabled:
        elim_counts.append((fid, fname, 0, len(current_pool)))
        continue

    kept = []
    eliminated_now = 0
    had_error = False

    for cmb in current_pool:
        env = combo_env_compat(base_env, cmb)

        # First, check applicability
        try:
            applies = safe_eval(app_if, env) if app_if else True
        except Exception as e:
            applies = False
            had_error = True
            skip_rows.append({
                "filter_id": fid, "name": fname, "stage": "applicable_if_eval",
                "missing": "", "error": str(e), "combo": cmb
            })

        # Then, evaluate filter expression (True means eliminate)
        ok = False
        if applies:
            expr_eval = expr
            # expand single-word alias-only expressions, e.g., "hotdigits > 2" stays as-is,
            # but if expression is exactly an unknown symbol, try alias directly
            try:
                ok = safe_eval(expr_eval, env)
            except Exception as e:
                had_error = True
                skip_rows.append({
                    "filter_id": fid, "name": fname, "stage": "expression_eval",
                    "missing": "", "error": str(e), "combo": cmb
                })
                ok = False

        if ok:
            eliminated_now += 1
        else:
            kept.append(cmb)

    current_pool = kept
    remaining_after = len(current_pool)
    elim_counts.append((fid, fname, eliminated_now, remaining_after))

# ----------------------------
# Results
# ----------------------------
res_df = pd.DataFrame(elim_counts, columns=["filter_id","name","eliminated_this_step","remaining_after"])
if hide_zero_elims:
    res_show = res_df[res_df["eliminated_this_step"] > 0].copy()
else:
    res_show = res_df.copy()

st.markdown("### Filter Diagnostics")
st.caption(f"Triggered filters: {res_show[res_show.eliminated_this_step>0].shape[0]} / {filters_df.shape[0]}")
st.dataframe(res_show, use_container_width=True, height=380)

# Remaining combinations
with st.expander("Show remaining combinations"):
    st.write(pd.DataFrame({"combo": current_pool}))

# Skip log
skip_df = pd.DataFrame(skip_rows)
if skip_df.empty:
    st.success("All filters evaluated without recorded errors.")
else:
    st.warning(f"{len(skip_df)} evaluation issues recorded (see table below & download).")
    st.dataframe(skip_df, height=260, use_container_width=True)
    csv = skip_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download skip log CSV", data=csv, file_name="filter_skip_log.csv", mime="text/csv", use_container_width=True)

# Simple recommendation summary
st.markdown("### Recommendation (simple)")
st.write(
    "Filters are applied in the given order; a filter *eliminates* a combo if its expression evaluates **True**. "
    "Remaining combos after the final step are shown above."
)
