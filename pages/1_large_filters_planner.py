# 1_Large_Filters_Planner.py
# UI preserved. Only filter-processing logic has been upgraded to match PythonFilterTester behavior.

from __future__ import annotations
import io, re, math, random, unicodedata
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

import pandas as pd
import streamlit as st

# ---------------- Page ----------------
st.set_page_config(page_title="Large Filters Planner", layout="wide")
st.title("Large Filters Planner")

# ---------------- Constants / Maps (logic only) ----------------
# VTRAC v4
VTRAC: Dict[int, int]  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

SAFE_BUILTINS = {
    "abs": abs, "int": int, "str": str, "float": float, "round": round,
    "len": len, "sum": sum, "max": max, "min": min, "any": any, "all": all,
    "set": set, "sorted": sorted, "list": list, "tuple": tuple, "dict": dict,
    "range": range, "enumerate": enumerate, "map": map, "filter": filter,
    "math": math, "re": re, "random": random, "Counter": Counter,
    "True": True, "False": False, "None": None,
}

# ---------------- Small helpers ----------------
def parse_list_any(text: str) -> List[str]:
    if not text: return []
    raw = text.replace("\t", ",").replace("\n", ",").replace(";", ",").replace(" ", ",")
    return [p.strip() for p in raw.split(",") if p.strip()]

def digits_of(s: str) -> List[int]:
    s = str(s).strip()
    return [int(ch) for ch in s if ch.isdigit()]

def safe_digits(x): 
    try: return [int(ch) for ch in str(x) if ch.isdigit()]
    except Exception: return []

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]: return "quint"
    if counts == [4,1]: return "quad"
    if counts == [3,2]: return "triple_double"
    if counts == [3,1,1]: return "triple"
    if counts == [2,2,1]: return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def even_count(x): return sum(1 for d in safe_digits(x) if d % 2 == 0)
def odd_count(x):  return sum(1 for d in safe_digits(x) if d % 2 == 1)
def high_count(x): return sum(1 for d in safe_digits(x) if d >= 5)
def low_count(x):  return sum(1 for d in safe_digits(x) if d <= 4)

def first_digit(x): ds = safe_digits(x); return ds[0] if ds else None
def last_digit(x):  ds = safe_digits(x); return ds[-1] if ds else None
def last_two_digits(x): ds = safe_digits(x); return ds[-2:] if len(ds) >= 2 else ds
def digit_sum(x): return sum(safe_digits(x))
def digit_span(x):  ds = safe_digits(x); return (max(ds) - min(ds)) if ds else 0
def has_triplet(x): c=Counter(safe_digits(x)); return max(c.values()) if c else 0 >= 3

def vtrac_of(d): 
    try: d=int(d); return VTRAC.get(d)
    except Exception: return None

def contains_mirror_pair(x):
    s = set(safe_digits(x))
    return any((d in s and MIRROR[d] in s and MIRROR[d] != d) for d in s)

# --------- H/C/D helpers bound to UI values ----------
def _mk_is_hot(env):  return lambda d: (str(d).isdigit() and int(d) in env.get("hot_set", set()))
def _mk_is_cold(env): return lambda d: (str(d).isdigit() and int(d) in env.get("cold_set", set()))
def _mk_is_due(env):  return lambda d: (str(d).isdigit() and int(d) in env.get("due_set", set()))

def count_in_hot(x, hot_set=None):
    hs = hot_set if hot_set is not None else set(st.session_state.get("hot_digits", []))
    return sum(1 for d in safe_digits(x) if d in hs)

def count_in_cold(x, cold_set=None):
    cs = cold_set if cold_set is not None else set(st.session_state.get("cold_digits", []))
    return sum(1 for d in safe_digits(x) if d in cs)

def count_in_due(x, due_set=None):
    ds = due_set if due_set is not None else set(st.session_state.get("due_digits", []))
    return sum(1 for d in safe_digits(x) if d in ds)

# --------- Expression normalization (tester-style) ----------
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
def _camel_to_snake(s: str) -> str: return _CAMEL_RE.sub('_', s).lower()
def _ascii(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    return (s.replace('“','"').replace('”','"')
             .replace("’","'").replace("‘","'")
             .replace("–","-").replace("—","-"))

def _wb_replace(text: str, mapping: Dict[str, str]) -> str:
    if not mapping: return text
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for k, v in items:
        text = re.sub(rf"\b{k}\b", v)
    return text

# Spelling/term variations taken from the tester app behavior
_VARIATION_MAP: Dict[str, str] = {
    # H/C/D
    "hotDigits":"hot_digits", "hotdigits":"hot_digits", "hotnumbers":"hot_digits", "hot":"hot_digits",
    "coldDigits":"cold_digits", "colddigits":"cold_digits", "coldnumbers":"cold_digits", "cold":"cold_digits",
    "dueDigits":"due_digits", "duedigits":"due_digits", "duenumbers":"due_digits", "due":"due_digits",
    "percentHot":"count_in_hot(combo_digits, hot_set)",   # % terms mapped to counts for compatibility
    "percentCold":"count_in_cold(combo_digits, cold_set)",
    "percentDue":"count_in_due(combo_digits, due_set)",
    "countHot":"sum(1 for d in combo_digits if d in hot_set)",
    "countCold":"sum(1 for d in combo_digits if d in cold_set)",
    "countDue":"sum(1 for d in combo_digits if d in due_set)",
    # mirror
    "mirrorDigits":"combo_mirror_digits","mirrordigits":"combo_mirror_digits","mirrors":"combo_mirror_digits",
    "mirrorSet":"set(combo_mirror_digits)","mirrorset":"set(combo_mirror_digits)",
    "mirrorPairs":"contains_mirror_pair(combo_digits)","hasMirrorPair":"contains_mirror_pair(combo_digits)",
    # combo/seed
    "comboDigits":"combo_digits","combodigits":"combo_digits",
    "comboSet":"set(combo_digits)","comboset":"set(combo_digits)",
    "seedDigits":"seed_digits","seeddigits":"seed_digits",
    "seedSet":"set(seed_digits)","seedset":"set(seed_digits)",
    # parity & counts
    "parityEven":"combo_sum_is_even", "isEven":"combo_sum_is_even", "isOdd":"not combo_sum_is_even",
    "evenCount":"even_count(combo_digits)", "oddCount":"odd_count(combo_digits)",
    "highCount":"high_count(combo_digits)", "lowCount":"low_count(combo_digits)",
    # positional
    "firstDigit":"first_digit(combo_digits)", "lastDigit":"last_digit(combo_digits)",
    "lastTwo":"last_two_digits(combo_digits)", "last2":"last_two_digits(combo_digits)",
    # sums/structure
    "sumDigits":"digit_sum(combo_digits)","digitSum":"digit_sum(combo_digits)","sum":"digit_sum(combo_digits)",
    "structure":"combo_structure",
    # vtrac
    "vtrack":"VTRAC","vtracks":"VTRAC","vtracGroups":"VTRAC",
    "vtracSet":"combo_vtracs","vtracLast":"combo_last_vtrac","lastVtrac":"combo_last_vtrac",
}

def normalize_expr(expr: str) -> str:
    if not expr: return ""
    x = _ascii(expr)
    # convert camelCase tokens we know about
    for t in ["hotDigits","coldDigits","dueDigits","mirrorPairs","mirrorDigits",
              "seedDigits","comboDigits","evenCount","oddCount","highCount","lowCount",
              "firstDigit","lastDigit","lastTwo","last2",
              "digitSum","sumDigits","vtracSet","vtracLast","vtracGroups",
              "comboSet","seedSet","isEven","isOdd","parityEven"]:
        if t in x: x = x.replace(t, _camel_to_snake(t))
    # word-boundary replacements
    x = _wb_replace(x, _VARIATION_MAP)
    # normalize strict operators the tester never uses
    x = x.replace("!==", "!=")
    return x

def _clean_expr(s: str) -> str:
    s = str(s or "").strip().strip('"').strip("'")
    return normalize_expr(s)

# ---------------- CSV loaders (unchanged UI expectations) ----------------
@st.cache_data(show_spinner=False)
def load_pool_from_text_or_csv(text: str, col_hint: str) -> List[str]:
    text = text.strip()
    if not text: return []
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

@st.cache_data(show_spinner=False)
def load_filters_from_source(pasted_csv_text: str, uploaded_csv_file, csv_path: str) -> pd.DataFrame:
    if pasted_csv_text and pasted_csv_text.strip():
        df = pd.read_csv(io.StringIO(pasted_csv_text), engine="python")
        return normalize_filters_df(df)
    if uploaded_csv_file is not None:
        df = pd.read_csv(uploaded_csv_file, engine="python")
        return normalize_filters_df(df)
    df = pd.read_csv(csv_path, engine="python")
    return normalize_filters_df(df)

def normalize_filters_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([{k.lower(): v for k, v in row.items()} for row in df.to_dict(orient="records")])
    if "id" not in out.columns and "fid" in out.columns:
        out["id"] = out["fid"]
    if "id" not in out.columns:
        out["id"] = range(1, len(out) + 1)
    if "expression" not in out.columns:
        raise ValueError("Filters CSV must include an 'expression' column.")
    out["expression"] = out["expression"].map(_clean_expr)
    if "name" not in out.columns:
        out["name"] = out["id"].astype(str)
    if "applicable_if" not in out.columns or out["applicable_if"].isna().all():
        out["applicable_if"] = "True"
    else:
        out["applicable_if"] = out["applicable_if"].map(_clean_expr)
    if "enabled" not in out.columns:
        out["enabled"] = True

    rows = []
    for _, r in out.iterrows():
        rr = dict(r)
        try:
            rr["applicable_code"] = compile(rr.get("applicable_if", "True") or "True", "<applicable>", "eval")
            rr["expr_code"]        = compile(rr.get("expression", "False") or "False", "<expr>", "eval")
        except SyntaxError as e:
            rr["enabled"] = False
            rr["compile_error"] = str(e)
        rows.append(rr)
    out = pd.DataFrame(rows)
    out = out[out["enabled"].astype(str).str.lower().isin(("1","true","t","yes","y"))].copy()
    return out

# ---------------- Environments (bound to UI) ----------------
def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str, prev_prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    env = {
        "seed_digits": digits_of(seed) if seed else [],
        "prev_seed_digits": digits_of(prev_seed) if prev_seed else [],
        "prev_prev_seed_digits": digits_of(prev_prev_seed) if prev_prev_seed else [],
        "prev_prev_prev_seed_digits": digits_of(prev_prev_prev_seed) if prev_prev_prev_seed else [],
        "VTRAC": VTRAC, "MIRROR": MIRROR, "mirror": MIRROR,
        "hot_digits": sorted(set(hot_digits)), "cold_digits": sorted(set(cold_digits)), "due_digits": sorted(set(due_digits)),
        "hot_set": set(hot_digits), "cold_set": set(cold_digits), "due_set": set(due_digits),
        "safe_digits": safe_digits, "digits_of": digits_of, "digit_sum": digit_sum, "digit_span": digit_span,
        "classify_structure": classify_structure, "contains_mirror_pair": contains_mirror_pair, "vtrac_of": vtrac_of,
        "even_count": even_count, "odd_count": odd_count, "high_count": high_count, "low_count": low_count,
        "first_digit": first_digit, "last_digit": last_digit, "last_two_digits": last_two_digits,
        "has_triplet": has_triplet, **SAFE_BUILTINS,
        # placeholders populated per-combo
        "combo": "", "combo_digits": [], "combo_set": set(), "combo_sum": 0,
        "combo_sum_is_even": False, "combo_last_digit": None, "combo_structure": "single",
    }
    env["is_hot"]  = _mk_is_hot(env)
    env["is_cold"] = _mk_is_cold(env)
    env["is_due"]  = _mk_is_due(env)
    return env

def combo_env(base_env: Dict, combo: str) -> Dict:
    cd = digits_of(combo)
    env = dict(base_env)
    env.update({
        "combo": combo,
        "combo_digits": cd,
        "combo_set": set(cd),
        "combo_sum": sum(cd),
        "combo_sum_is_even": (sum(cd) % 2 == 0),
        "combo_last_digit": cd[-1] if cd else None,
        "combo_structure": classify_structure(cd),
        "combo_mirror_digits": [MIRROR[d] for d in cd] if cd else [],
        "combo_vtracs": set(VTRAC[d] for d in cd) if cd else set(),
        "combo_last_vtrac": (VTRAC[cd[-1]] if cd else None),
    })
    env["is_hot"]  = _mk_is_hot(env)
    env["is_cold"] = _mk_is_cold(env)
    env["is_due"]  = _mk_is_due(env)
    return env

# ---------------- Evaluators ----------------
def eval_applicable(row: pd.Series, base_env: Dict) -> bool:
    try:
        return bool(eval(row["applicable_code"], {"__builtins__": {}}, base_env))
    except Exception:
        return True  # permissive like the tester

def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict) -> Tuple[Set[str], int]:
    eliminated: Set[str] = set()
    code = row["expr_code"]
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            # match tester behavior: skip on error, don't crash the batch
            pass
    return eliminated, len(eliminated)

# --------------------------- UI (unchanged layout) ---------------------------

# (A) Hot / Cold / Due (manual)
st.subheader("Hot / Cold / Due digits (optional)")
cc1, cc2, cc3 = st.columns(3)
hot_digits  = [int(x) for x in parse_list_any(cc1.text_input("Hot digits (comma-separated)")) if x.isdigit()]
cold_digits = [int(x) for x in parse_list_any(cc2.text_input("Cold digits (comma-separated)")) if x.isdigit()]
due_digits  = [int(x) for x in parse_list_any(cc3.text_input("Due digits (comma-separated)")) if x.isdigit()]
st.session_state["hot_digits"] = hot_digits
st.session_state["cold_digits"] = cold_digits
st.session_state["due_digits"]  = due_digits

# (B) Combo Pool
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV w/ 'Result' column OR tokens separated by newline/space/comma):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV ('Result' or 'combo' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")

pool: List[str] = []
try:
    if pool_text.strip():
        pool = load_pool_from_text_or_csv(pool_text, pool_col_hint)
    elif pool_file is not None:
        pool = load_pool_from_file(pool_file, pool_col_hint)
except Exception as e:
    st.error(f"Pool import failed: {e}")

st.caption(f"Pool size: {len(pool)}")

# (C) Draw History (4 back) — always visible (UI preserved)
st.subheader("Draw History (4 back)")
s1, s2, s3, s4 = st.columns(4)
seed      = s1.text_input("Known winner (0-back)", value="")
prev_seed = s2.text_input("Draw 1-back", value="")
prev_prev = s3.text_input("Draw 2-back", value="")
prev_prev_prev = s4.text_input("Draw 3-back", value="")

# (D) Filters
st.subheader("Filters")
fids_text = st.text_area("Paste applicable Filter IDs (optional; comma / space / newline separated):", height=90)
filters_pasted_csv = st.text_area("Paste Filters CSV content (optional):", height=150)
filters_file_up = st.file_uploader("Or upload Filters CSV (used if pasted CSV is empty)", type=["csv"])
filters_csv_path = st.text_input("Or path to Filters CSV (used if pasted/upload empty)", value="lottery_filters_batch_10.csv")

try:
    filters_df_full = load_filters_from_source(filters_pasted_csv, filters_file_up, filters_csv_path)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    filters_df_full = pd.DataFrame(columns=["id","name","expression","enabled","applicable_if"])

# Optional restriction by IDs or names
applicable_ids = set(parse_list_any(fids_text))
if applicable_ids and len(filters_df_full):
    id_str   = filters_df_full["id"].astype(str)
    name_str = filters_df_full.get("name","").astype(str)
    mask = id_str.isin(applicable_ids) | name_str.isin(applicable_ids)
    filters_df = filters_df_full[mask].copy()
else:
    filters_df = filters_df_full.copy()

st.caption(f"Filters loaded: {len(filters_df)}")

# (E) Run
run = st.button("▶ Run Planner", type="primary", disabled=(len(pool)==0 or len(filters_df)==0))

if run:
    base_env = make_base_env(seed, prev_seed, prev_prev, prev_prev_prev, hot_digits, cold_digits, due_digits)
    pool_list = list(pool)

    rows = []
    for _, r in filters_df.iterrows():
        if not eval_applicable(r, base_env):
            continue
        elim, cnt = eval_filter_on_pool(r, pool_list, base_env)
        rows.append({
            "id": r["id"],
            "name": r.get("name",""),
            "expression": r["expression"],
            "eliminated_now": cnt
        })

    if rows:
        out = pd.DataFrame(rows).sort_values("eliminated_now", ascending=False)
        st.subheader("Filter Diagnostics")
        st.dataframe(out, use_container_width=True)
        st.subheader("Large filters on current pool")
        st.dataframe(out[out["eliminated_now"]>0], use_container_width=True)
    else:
        st.info("No filters evaluated / nothing eliminated.")
