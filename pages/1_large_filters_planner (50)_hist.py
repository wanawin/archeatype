# 1_Large_Filters_Planner.py
# Large Filters Planner — Archetyper
# Updates in this build:
# - Expose seed_sum (= sum(seed_digits)) in the eval environment.
# - Add aliases: seedSum, sumSeedDigits -> seed_sum.
# - All prior fixes retained: hybrid mirror, robust eval scoping, H/C/D, vtracs,
#   phrase normalization ("mirror of X"), skip reporting, token panel always visible.

from __future__ import annotations
import io, math, os, random, re, unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Large Filters Planner — Archetyper", layout="wide")
st.title("Large Filters Planner — Archetyper")

# ── Constants ─────────────────────────────────────────────────────────────────
VTRAC: Dict[int, int] = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR_MAP: Dict[int, int] = {0:5,5:0, 1:6,6:1, 2:7,7:2, 3:8,8:3, 4:9,9:4}

SAFE_BUILTINS = {
    "abs":abs, "int":int, "str":str, "float":float, "round":round, "len":len, "sum":sum,
    "max":max, "min":min, "any":any, "all":all, "set":set, "sorted":sorted, "list":list,
    "tuple":tuple, "dict":dict, "range":range, "enumerate":enumerate, "map":map, "filter":filter,
    "math":math, "re":re, "random":random, "Counter":Counter, "True":True, "False":False,
    "None":None, "nan":float("nan"),
}

# ── Helpers exposed to expressions ────────────────────────────────────────────
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

def digit_sum(x): return sum(safe_digits(x))
def digit_span(x):
    ds = safe_digits(x)
    return (max(ds) - min(ds)) if ds else 0

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
def has_triplet(x): c = Counter(safe_digits(x)); return (max(c.values()) if c else 0) >= 3

def vtrac_of(d):
    try: d = int(d); return VTRAC.get(d)
    except Exception: return None

def contains_mirror_pair(x):
    s = set(safe_digits(x))
    return any((d in s and MIRROR_MAP.get(d) in s and MIRROR_MAP[d] != d) for d in s)

# Hybrid 'mirror' that works both as a callable and a dict (.get / [key])
class _MirrorHybrid:
    def __init__(self, mapping: Dict[int,int]): self._m = dict(mapping)
    def __call__(self, x):
        try:
            x = int(x)
        except Exception:
            ds = [int(ch) for ch in str(x) if ch.isdigit()]
            if not ds: return x
            md = [self._m.get(d, d) for d in ds]
            try: return int("".join(str(d) for d in md))
            except Exception: return "".join(str(d) for d in md)
        if 0 <= x <= 9: return self._m.get(x, x)
        ds = [int(ch) for ch in str(x)]
        md = [self._m.get(d, d) for d in ds]
        try: return int("".join(str(d) for d in md))
        except Exception: return "".join(str(d) for d in md)
    # dict-like
    def get(self, k, default=None): return self._m.get(int(k), default)
    def __getitem__(self, k): return self._m[int(k)]
    def __contains__(self, k):
        try: return int(k) in self._m
        except Exception: return False
    def items(self): return self._m.items()
    def keys(self): return self._m.keys()
    def values(self): return self._m.values()

MIRROR = _MirrorHybrid(MIRROR_MAP)  # hybrid object

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

def digital_root(n: int) -> int:
    try: n = int(n)
    except Exception: n = 0
    if n == 0: return 0
    m = n % 9
    return 9 if m == 0 else m

def as_int_from_digits(digs) -> int:
    if not digs: return 0
    try: return int("".join(str(d) for d in digs))
    except Exception: return 0

# ── Expression normalization (aliases) ────────────────────────────────────────
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
def _camel_to_snake(s: str) -> str: return _CAMEL_RE.sub("_", s).lower()
def _ascii(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    return (s.replace("“", '"').replace("”", '"').replace("’","'").replace("‘","'")
             .replace("–","-").replace("—","-"))

def _wb_replace(text: str, mapping: Dict[str,str]) -> str:
    if not mapping: return text
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for k, v in items:
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text

def _normalize_phrases(x: str) -> str:
    x = re.sub(r"\bmirror\s+of\s*\(", "mirror(", x, flags=re.IGNORECASE)
    x = re.sub(r"\bmirror\s+of\s+([A-Za-z_][A-Za-z0-9_]*)", r"mirror(\1)", x, flags=re.IGNORECASE)
    return x

_VARIATION_MAP: Dict[str,str] = {
    # H/C/D
    "hotDigits":"hot_digits","hotdigits":"hot_digits","hotnumbers":"hot_digits","hot":"hot_digits",
    "coldDigits":"cold_digits","colddigits":"cold_digits","coldnumbers":"cold_digits","cold":"cold_digits",
    "dueDigits":"due_digits","duedigits":"due_digits","duenumbers":"due_digits","due":"due_digits",
    "percentHot":"count_in_hot(combo_digits, hot_set)",
    "percentCold":"count_in_cold(combo_digits, cold_set)",
    "percentDue":"count_in_due(combo_digits, due_set)",
    "countHot":"sum(1 for d in combo_digits if d in hot_set)",
    "countCold":"sum(1 for d in combo_digits if d in cold_set)",
    "countDue":"sum(1 for d in combo_digits if d in due_set)",
    # mirror
    "mirrorDigits":"combo_mirror_digits","mirrordigits":"combo_mirror_digits","mirrors":"combo_mirror_digits",
    "mirrorSet":"set(combo_mirror_digits)","mirrorset":"set(combo_mirror_digits)",
    "mirrorPairs":"contains_mirror_pair(combo_digits)","hasMirrorPair":"contains_mirror_pair(combo_digits)",
    "mirrorOf":"mirror","mirror_of":"mirror",
    # combo/seed
    "comboDigits":"combo_digits","combodigits":"combo_digits",
    "comboSet":"set(combo_digits)","comboset":"set(combo_digits)",
    "seedDigits":"seed_digits","seeddigits":"seed_digits",
    "seedSet":"set(seed_digits)","seedset":"set(seed_digits)",
    # seed digit aliases
    "seed_digits_1":"prev_seed_digits",
    "seed_digits_2":"prev_prev_seed_digits",
    "seed_digits_3":"prev_prev_prev_seed_digits",
    # vtrac (v4)
    "vtrack":"VTRAC","vtracks":"VTRAC","vtracGroups":"VTRAC",
    "vtracSet":"combo_vtracs","vtracLast":"combo_last_vtrac","lastVtrac":"combo_last_vtrac",
    "seedVtracs":"seed_vtracs","seed_vtracs":"seed_vtracs",
    # parity/counts
    "parityEven":"combo_sum_is_even","isEven":"combo_sum_is_even","isOdd":"not combo_sum_is_even",
    "evenCount":"even_count(combo_digits)","oddCount":"odd_count(combo_digits)",
    "highCount":"high_count(combo_digits)","lowCount":"low_count(combo_digits)",
    # positional
    "firstDigit":"first_digit(combo_digits)","lastDigit":"last_digit(combo_digits)",
    "lastTwo":"last_two_digits(combo_digits)","last2":"last_two_digits(combo_digits)",
    # sums / structure
    "sumDigits":"digit_sum(combo_digits)","digitSum":"digit_sum(combo_digits)",
    "comboSum":"digit_sum(combo_digits)",
    "structure":"combo_structure",
    # value & root sum
    "seedValue":"seed_value","comboValue":"combo_value",
    "rootSum":"digital_root","seedRootSum":"seed_root_sum","comboRootSum":"combo_root_sum",
    # NEW: seed sum aliases
    "seedSum":"seed_sum","sumSeedDigits":"seed_sum",
    # NaN
    "NaN":"nan",
}
# Intentionally DO NOT map bare 'sum' (keep built-in sum()).

def normalize_expr(expr: str) -> str:
    if not expr: return ""
    x = _ascii(expr)
    x = _normalize_phrases(x)
    for t in [
        "hotDigits","coldDigits","dueDigits","mirrorPairs","mirrorDigits","seedDigits","comboDigits",
        "percentHot","percentCold","percentDue","evenCount","oddCount","highCount","lowCount",
        "firstDigit","lastDigit","lastTwo","last2","digitSum","sumDigits","vtracSet","vtracLast","vtracGroups",
        "comboSum","comboSet","seedSet","isEven","isOdd","parityEven","structure",
        "seedValue","comboValue","seedRootSum","comboRootSum","rootSum",
        "seed_digits_1","seed_digits_2","seed_digits_3","seedVtracs","seed_vtracs",
        "seedSum","sumSeedDigits",
    ]:
        if t in x: x = x.replace(t, _camel_to_snake(t))
    x = _wb_replace(x, _VARIATION_MAP)
    x = x.replace("!==", "!=")
    return x

def _clean_expr(s: str) -> str:
    s = str(s or "").strip().strip('"').strip("'")
    return normalize_expr(s)

# ── CSV loaders ───────────────────────────────────────────────────────────────
def _pick_col(df: pd.DataFrame, hint: str) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}
    if hint and hint in df.columns: return df[hint]
    if "result" in cols_lower: return df[cols_lower["result"]]
    if "combo" in cols_lower:  return df[cols_lower["combo"]]
    return df[df.columns[0]]

def load_pool_from_text_or_csv(text: str, col_hint: str) -> List[str]:
    text = text.strip()
    if not text: return []
    looks_csv = ("," in text and "\n" in text) or text.lower().startswith("result")
    if looks_csv:
        try:
            df = pd.read_csv(io.StringIO(text), engine="python")
            s = _pick_col(df, col_hint)
            return [str(x).strip() for x in s.dropna().astype(str)]
        except Exception:
            pass
    return parse_list_any(text)

def load_pool_from_file(f, col_hint: str) -> List[str]:
    df = pd.read_csv(f, engine="python")
    s = _pick_col(df, col_hint)
    return [str(x).strip() for x in s.dropna().astype(str)]

def normalize_filters_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([{k.lower(): v for k, v in row.items()} for row in df.to_dict(orient="records")])
    if "id" not in out.columns and "fid" in out.columns: out["id"] = out["fid"]
    if "id" not in out.columns: out["id"] = range(1, len(out) + 1)
    if "expression" not in out.columns: raise ValueError("Filters CSV must include an 'expression' column.")
    out["expression"] = out["expression"].map(_clean_expr)
    if "name" not in out.columns: out["name"] = out["id"].astype(str)
    if "applicable_if" not in out.columns or out["applicable_if"].isna().all():
        out["applicable_if"] = "True"
    else:
        out["applicable_if"] = out["applicable_if"].map(_clean_expr)
    if "enabled" not in out.columns: out["enabled"] = True

    rows = []
    for _, r in out.iterrows():
        rr = dict(r)
        try:
            rr["applicable_code"] = compile(rr.get("applicable_if", "True") or "True", "<applicable>", "eval")
        except SyntaxError as e:
            rr["applicable_code"] = compile("True", "<applicable>", "eval")
            rr["compile_error_applicable"] = str(e)
        try:
            rr["expr_code"] = compile(rr.get("expression", "False") or "False", "<expr>", "eval")
        except SyntaxError as e:
            rr["expr_code"] = compile("False", "<expr>", "eval")
            rr["compile_error_expr"] = str(e)
        rows.append(rr)
    return pd.DataFrame(rows)

def load_filters_from_source(pasted_csv_text: str, uploaded_csv_file, csv_path: str) -> pd.DataFrame:
    if pasted_csv_text and pasted_csv_text.strip():
        df = pd.read_csv(io.StringIO(pasted_csv_text), engine="python")
        return normalize_filters_df(df)
    if uploaded_csv_file is not None:
        df = pd.read_csv(uploaded_csv_file, engine="python")
        return normalize_filters_df(df)
    df = pd.read_csv(csv_path, engine="python")
    return normalize_filters_df(df)

# ── Environments ──────────────────────────────────────────────────────────────
def make_base_env(seed, prev_seed, prev_prev_seed, prev_prev_prev_seed,
                  hot_digits, cold_digits, due_digits) -> Dict:
    env = {
        "seed_digits": digits_of(seed) if seed else [],
        "prev_seed_digits": digits_of(prev_seed) if prev_seed else [],
        "prev_prev_seed_digits": digits_of(prev_prev_seed) if prev_prev_seed else [],
        "prev_prev_prev_seed_digits": digits_of(prev_prev_prev_seed) if prev_prev_prev_seed else [],
        "VTRAC": VTRAC,
        "MIRROR": MIRROR,          # UPPERCASE alias
        "mirror": MIRROR,          # lowercase alias (CSV uses 'mirror')
        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits":  sorted(set(due_digits)),
        "hot_set": set(hot_digits), "cold_set": set(cold_digits), "due_set": set(due_digits),
        "digits_of": digits_of, "safe_digits": safe_digits, "digit_sum": digit_sum, "even_count": even_count,
        "odd_count": odd_count, "high_count": high_count, "low_count": low_count, "first_digit": first_digit,
        "last_digit": last_digit, "last_two_digits": last_two_digits, "digit_span": digit_span,
        "classify_structure": classify_structure, "has_triplet": has_triplet, "contains_mirror_pair": contains_mirror_pair,
        "vtrac_of": vtrac_of, "digital_root": digital_root,
        **SAFE_BUILTINS,
        # combo placeholders
        "combo": "", "combo_digits": [], "combo_set": set(), "combo_sum": 0, "combo_sum_is_even": False,
        "combo_last_digit": None, "combo_structure": "single",
    }
    # seed scalars + root sums + vtracs
    seed_value = as_int_from_digits(env["seed_digits"])
    prev_seed_value = as_int_from_digits(env["prev_seed_digits"])
    prev_prev_seed_value = as_int_from_digits(env["prev_prev_seed_digits"])
    prev_prev_prev_seed_value = as_int_from_digits(env["prev_prev_prev_seed_digits"])
    # NEW: seed_sum value (used by some filters)
    seed_sum_val = sum(env["seed_digits"])
    env.update({
        "seed_value": seed_value, "prev_seed_value": prev_seed_value,
        "prev_prev_seed_value": prev_prev_seed_value, "prev_prev_prev_seed_value": prev_prev_prev_seed_value,
        "seed_root_sum": digital_root(seed_value), "prev_seed_root_sum": digital_root(prev_seed_value),
        "prev_prev_seed_root_sum": digital_root(prev_prev_seed_value),
        "prev_prev_prev_seed_root_sum": digital_root(prev_prev_prev_seed_value),
        "seed_sum": seed_sum_val,  # <<< exposed
        "seed_vtracs": set(VTRAC[d] for d in env["seed_digits"]) if env["seed_digits"] else set(),
        "prev_seed_vtracs": set(VTRAC[d] for d in env["prev_seed_digits"]) if env["prev_seed_digits"] else set(),
        "prev_prev_seed_vtracs": set(VTRAC[d] for d in env["prev_prev_seed_digits"]) if env["prev_prev_seed_digits"] else set(),
        "prev_prev_prev_seed_vtracs": set(VTRAC[d] for d in env["prev_prev_prev_seed_digits"]) if env["prev_prev_prev_seed_digits"] else set(),
    })
    env["is_hot"] = _mk_is_hot(env); env["is_cold"] = _mk_is_cold(env); env["is_due"] = _mk_is_due(env)
    return env

def combo_env(base_env: Dict, combo: str) -> Dict:
    cd = digits_of(combo)
    env = dict(base_env)
    combo_value = as_int_from_digits(cd)
    env.update({
        "combo": combo, "combo_digits": cd, "combo_set": set(cd),
        "combo_sum": sum(cd), "combo_sum_is_even": (sum(cd) % 2 == 0),
        "combo_last_digit": (cd[-1] if cd else None), "combo_structure": classify_structure(cd),
        "combo_mirror_digits": [MIRROR_MAP.get(d, d) for d in cd] if cd else [],
        "combo_vtracs": set(VTRAC[d] for d in cd) if cd else set(),
        "combo_last_vtrac": (VTRAC[cd[-1]] if cd else None),
        "combo_value": combo_value, "combo_root_sum": digital_root(combo_value),
    })
    env["is_hot"] = _mk_is_hot(env); env["is_cold"] = _mk_is_cold(env); env["is_due"] = _mk_is_due(env)
    return env

# ── Evaluators (robust scoping + error collection) ────────────────────────────
def eval_applicable(row: pd.Series, base_env: Dict) -> bool:
    try:
        globs = {"__builtins__": {}}
        globs.update(base_env)
        return bool(eval(row["applicable_code"], globs, base_env))
    except Exception:
        return True

NAME_ERR = re.compile(r"name '([^']+)' is not defined")

def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict,
                        runtime_errors_accum: Dict[str,dict], token_counts: Dict[str,int]) -> Tuple[Set[str], int]:
    eliminated = set()
    code = row["expr_code"]; fid = str(row.get("id","")); fname = str(row.get("name",""))
    for c in pool:
        env = combo_env(base_env, c)
        try:
            globs = {"__builtins__": {}}
            globs.update(env)
            if bool(eval(code, globs, env)):
                eliminated.add(c)
        except Exception as e:
            rec = runtime_errors_accum.setdefault(
                fid, {"id":fid, "name":fname, "error_type":"runtime", "error_count":0, "first_error":""}
            )
            rec["error_count"] += 1
            if not rec["first_error"]:
                rec["first_error"] = f"{type(e).__name__}: {e}"
            m = NAME_ERR.search(str(e))
            if m: token_counts[m.group(1)] += 1
            continue
    return eliminated, len(eliminated)

# ── Sidebar (unchanged) ───────────────────────────────────────────────────────
st.sidebar.subheader("Mode")
mode = st.sidebar.radio("", ["Playlist Reducer","Safe Filter Explorer"], index=1, label_visibility="collapsed")
min_large = int(st.sidebar.number_input("Min eliminations to call it 'Large'", 1, 1000, 60))
greedy_beam = int(st.sidebar.number_input("Greedy beam width", 1, 50, 6))
greedy_steps = int(st.sidebar.number_input("Greedy max steps", 1, 200, 18))
exclude_parity_wipers = st.sidebar.checkbox("Exclude parity-wipers", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Archetype Lifts (optional)")
use_archetype_lift = st.sidebar.checkbox("Use archetype-lift CSV if present", value=True)
arch_path = st.sidebar.text_input("Archetype-lifts CSV path", value="archetype_filter_dimension.csv")

# ── Center UI (unchanged) ─────────────────────────────────────────────────────
st.subheader("Hot / Cold / Due digits (optional)")
cc1, cc2, cc3 = st.columns(3)
hot_digits = [int(x) for x in parse_list_any(cc1.text_input("Hot digits (comma-separated)")) if x.isdigit()]
cold_digits = [int(x) for x in parse_list_any(cc2.text_input("Cold digits (comma-separated)")) if x.isdigit()]
due_digits  = [int(x) for x in parse_list_any(cc3.text_input("Due digits (comma-separated)")) if x.isdigit()]
st.session_state["hot_digits"] = hot_digits; st.session_state["cold_digits"] = cold_digits; st.session_state["due_digits"] = due_digits

st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV w/ 'Result' column OR tokens separated by newline/space/comma):", height=140)
pool_file = st.file_uploader("Or upload combo pool CSV ('Result' or 'combo' column)", type=["csv"])
pool_col_hint = st.text_input("Pool column name hint (default 'Result')", value="Result")
pool: List[str] = []
try:
    if pool_text.strip(): pool = load_pool_from_text_or_csv(pool_text, pool_col_hint)
    elif pool_file is not None: pool = load_pool_from_file(pool_file, pool_col_hint)
except Exception as e:
    st.error(f"Pool import failed: {e}")
st.caption(f"Pool size: {len(pool)}")

st.subheader("Draw History (4 back)")
s1, s2, s3, s4 = st.columns(4)
seed           = s1.text_input("Known winner (0-back)", value="")
prev_seed      = s2.text_input("Draw 1-back", value="")
prev_prev      = s3.text_input("Draw 2-back", value="")
prev_prev_prev = s4.text_input("Draw 3-back", value="")

st.subheader("Filters")
fids_text = st.text_area("Paste applicable Filter IDs (optional; comma / space / newline separated):", height=90)
filters_pasted_csv = st.text_area("Paste Filters CSV content (optional):", height=150)
filters_file_up = st.file_uploader("Or upload Filters CSV (used if pasted CSV is empty)", type=["csv"])
filters_csv_path = st.text_input("Or path to Filters CSV (used if pasted/upload empty)", value="lottery_filters_batch_10.csv")

try:
    filters_df_full = load_filters_from_source(filters_pasted_csv, filters_file_up, filters_csv_path)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}")
    filters_df_full = pd.DataFrame(columns=["id","name","expression","enabled","applicable_if","compile_error_applicable","compile_error_expr"])

applicable_ids = set(parse_list_any(fids_text))
if applicable_ids and len(filters_df_full):
    id_str   = filters_df_full["id"].astype(str)
    name_str = filters_df_full.get("name","").astype(str)
    mask = id_str.isin(applicable_ids) | name_str.isin(applicable_ids)
    filters_df = filters_df_full[mask].copy()
else:
    filters_df = filters_df_full.copy()

if exclude_parity_wipers and len(filters_df):
    kw = ["parity-wiper","parity wiper","wipe parity"]
    mask = ~(
        filters_df.get("name","").astype(str).str.lower().str.contains("|".join(kw))
        | filters_df["expression"].astype(str).str.lower().str.contains("|".join(kw))
    )
    filters_df = filters_df[mask].copy()
st.caption(f"Filters loaded: {len(filters_df)}")

# ── Run & cache ───────────────────────────────────────────────────────────────
def run_planner_and_cache():
    base_env = make_base_env(
        seed, prev_seed, prev_prev, prev_prev_prev,
        st.session_state.get("hot_digits", []),
        st.session_state.get("cold_digits", []),
        st.session_state.get("due_digits", []),
    )
    pool_list = list(pool)

    rows = []; runtime_errors = {}; token_freq = defaultdict(int)

    for _, r in filters_df.iterrows():
        if str(r.get("compile_error_applicable","")).strip():
            fid=str(r.get("id","")); fname=str(r.get("name",""))
            runtime_errors[fid]={"id":fid,"name":fname,"error_type":"compile","error_count":1,
                                 "first_error":f"applicable_if: {r['compile_error_applicable']}"}
        if str(r.get("compile_error_expr","")).strip():
            fid=str(r.get("id","")); fname=str(r.get("name",""))
            rec=runtime_errors.setdefault(fid, {"id":fid,"name":fname,"error_type":"compile","error_count":0,"first_error":""})
            rec["error_count"]+=1
            if not rec["first_error"]:
                rec["first_error"]=f"expression: {r['compile_error_expr']}"

        if not eval_applicable(r, base_env): continue
        eliminated, cnt = eval_filter_on_pool(r, pool_list, base_env, runtime_errors, token_freq)
        rows.append({"id": r["id"], "name": r.get("name",""), "expression": r["expression"], "eliminated_now": cnt})

    out = pd.DataFrame(rows).sort_values("eliminated_now", ascending=False) if rows else pd.DataFrame()
    pool_n = max(1, len(pool_list))
    if not out.empty:
        out["expected_safety_%"] = (1.0 - out["eliminated_now"] / pool_n) * 100.0
        if use_archetype_lift and arch_path and os.path.exists(arch_path):
            try:
                lift_df = pd.read_csv(arch_path, engine="python")
                fid_col = "id" if "id" in lift_df.columns else ("filter_id" if "filter_id" in lift_df.columns else None)
                lift_col = "lift" if "lift" in lift_df.columns else ("safety_lift" if "safety_lift" in lift_df.columns else None)
                if fid_col and lift_col:
                    lift_df = lift_df[[fid_col, lift_col]].rename(columns={fid_col:"id", lift_col:"lift"})
                    out = out.merge(lift_df, how="left", on="id")
                    out["expected_safety_lifted_%"] = out["expected_safety_%"] * (out["lift"].fillna(1.0))
            except Exception as e:
                st.warning(f"Archetype lift merge skipped: {e}")

    large_only = out[out["eliminated_now"] >= min_large] if not out.empty else pd.DataFrame()

    # Winner-preserving plan
    winner_plan = pd.DataFrame()
    if seed and not out.empty:
        rows_keep = []
        for _, r in out.iterrows():
            row_full = filters_df.loc[filters_df["id"]==r["id"]].iloc[0]
            env_seed = combo_env(base_env, seed)
            try:
                globs = {"__builtins__": {}}; globs.update(env_seed)
                knocks_winner = bool(eval(row_full["expr_code"], globs, env_seed))
            except Exception:
                knocks_winner = False
            if not knocks_winner and r["eliminated_now"] >= min_large:
                rows_keep.append(r)
        winner_plan = pd.DataFrame(rows_keep)

    # Reducer (playlist mode)
    kept_combos = list(pool_list); removed = set(); applied_ids = []
    if not out.empty and mode == "Playlist Reducer":
        current_pool = set(kept_combos); candidates = out.copy()
        for _ in range(greedy_steps):
            if candidates.empty or not current_pool: break
            cand_rows = []
            for _, r in candidates.iterrows():
                fid=r["id"]; row_full=filters_df.loc[filters_df["id"]==fid].iloc[0]
                elim,cnt=eval_filter_on_pool(row_full, list(current_pool), base_env, {}, defaultdict(int))
                cand_rows.append((fid, r.get("name",""), cnt, elim))
            cand_rows.sort(key=lambda x: x[2], reverse=True)
            top = cand_rows[:greedy_beam]
            for fid, nm, cnt, elimset in top:
                if cnt <= 0: continue
                applied_ids.append(str(fid))
                current_pool -= set(elimset)
                removed |= set(elimset)
            candidates = candidates[~candidates["id"].astype(str).isin(applied_ids)]
        kept_combos = sorted(list(current_pool))

    kept_df = pd.DataFrame({"Result": kept_combos})
    rem_df  = pd.DataFrame({"Result": sorted(list(removed))})

    # Skips + undefined-token frequency
    skipped_records = dict(runtime_errors)
    for _, r in filters_df.iterrows():
        if str(r.get("compile_error_applicable","")).strip() or str(r.get("compile_error_expr","")).strip():
            fid=str(r.get("id","")); fname=str(r.get("name",""))
            rec=skipped_records.setdefault(fid, {"id":fid,"name":fname,"error_type":"compile","error_count":0,"first_error":""})
            if str(r.get("compile_error_applicable","")).strip():
                rec["error_count"]+=1
                if not rec["first_error"]: rec["first_error"]=f"applicable_if: {r['compile_error_applicable']}"
            if str(r.get("compile_error_expr","")).strip():
                rec["error_count"]+=1
                if not rec["first_error"]: rec["first_error"]=f"expression: {r['compile_error_expr']}"

    skipped_df = pd.DataFrame(list(skipped_records.values())) if skipped_records else pd.DataFrame()
    tok_df = pd.DataFrame(sorted(token_freq.items(), key=lambda kv: kv[1], reverse=True),
                          columns=["symbol","error_count"]) if token_freq else pd.DataFrame()

    st.session_state["last_run"] = {
        "out": out, "large_only": large_only, "winner_plan": winner_plan,
        "kept_df": kept_df, "rem_df": rem_df, "skipped_df": skipped_df, "tok_df": tok_df,
    }

run = st.button("▶ Run Planner + Recommender", type="primary", disabled=(len(pool)==0 or len(filters_df)==0))
if run:
    run_planner_and_cache()

# ── Render from cache (downloads don’t reset) ─────────────────────────────────
if "last_run" in st.session_state and st.session_state["last_run"]:
    R = st.session_state["last_run"]
    out, large_only = R["out"], R["large_only"]
    winner_plan = R["winner_plan"]
    kept_df, rem_df = R["kept_df"], R["rem_df"]
    skipped_df, tok_df = R["skipped_df"], R["tok_df"]

    st.subheader("Filter Diagnostics")
    if out is None or out.empty:
        st.info("No filters evaluated / nothing eliminated.")
    else:
        st.dataframe(out, use_container_width=True)

    if large_only is not None and not large_only.empty:
        st.subheader("Best-case plan — Large filters only")
        st.dataframe(large_only, use_container_width=True)

    st.subheader("Winner-preserving plan — Large filters only")
    if seed:
        if winner_plan is not None and not winner_plan.empty:
            st.dataframe(winner_plan, use_container_width=True)
        else:
            st.caption("No large filters that both eliminate many and keep the known winner.")
    else:
        st.caption("Provide a 5-digit Known winner to compute a winner-preserving plan.")

    st.subheader("Reducer result (mode-aware)")
    st.write(f"Best-case final kept pool size: {0 if kept_df is None else len(kept_df)}")
    if kept_df is not None and not kept_df.empty:
        st.dataframe(kept_df.head(200), use_container_width=True)

    st.subheader("Downloads")
    kept_csv = kept_df.to_csv(index=False) if kept_df is not None else "Result\n"
    st.download_button("Download KEPT combos (CSV)", kept_csv, "kept_combos.csv", "text/csv")
    kept_txt = "\n".join(kept_df["Result"].tolist()) if kept_df is not None else ""
    st.download_button("Download KEPT combos (TXT)", kept_txt, "kept_combos.txt", "text/plain")

    rem_csv = rem_df.to_csv(index=False) if rem_df is not None else "Result\n"
    st.download_button("Download REMOVED combos (CSV)", rem_csv, "removed_combos.csv", "text/csv")
    rem_txt = "\n".join(rem_df["Result"].tolist()) if rem_df is not None else ""
    st.download_button("Download REMOVED combos (TXT)", rem_txt, "removed_combos.txt", "text/plain")

    st.subheader("Skipped / Failed filters")
    if skipped_df is not None and not skipped_df.empty:
        st.dataframe(skipped_df.sort_values(["error_type","error_count"], ascending=[True, False]), use_container_width=True)
    else:
        st.caption("No skipped/failed filters — all parsed & evaluated.")
    st.download_button(
        "Download skipped/failed filters (CSV)",
        (skipped_df.to_csv(index=False) if skipped_df is not None else "id,name,error_type,error_count,first_error\n"),
        "filter_skips.csv", "text/csv"
    )

    st.subheader("Undefined token frequency (from NameError)")
    if tok_df is not None and not tok_df.empty:
        st.bar_chart(tok_df.set_index("symbol")["error_count"])
        st.dataframe(tok_df, use_container_width=True)
    else:
        st.caption("No NameErrors (undefined symbols) detected in this run.")
    st.download_button(
        "Download undefined-token frequency (CSV)",
        (tok_df.to_csv(index=False) if tok_df is not None else "symbol,error_count\n"),
        "skip_report_todo_symbols.csv", "text/csv"
    )
else:
    st.info("Load your pool and filters, then click **Run Planner + Recommender** to see results.")



# ===================== HFR ADDITIVE SECTION (DO NOT REMOVE EXISTING CAPABILITIES) =====================
# Historical Safety Recommender (HFR): additive, namespaced, and optional via sidebar toggle.
# This section introduces uploaders for history & filters and ranks filters by historical safety for seeds
# similar to the current input seed. It does NOT modify or remove any existing logic/UI above.

from collections import Counter
from datetime import datetime
import pandas as pd

# ---- HFR constants and helpers (namespaced) ----
HFR_V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
HFR_MIRROR_PAIRS = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def HFR_digits_from_str(s: str):
    s = ''.join(ch for ch in str(s) if ch.isdigit())
    return [int(ch) for ch in s]

def HFR_sum_of_digits(digs):
    return sum(digs)

def HFR_parity_counts(digs):
    ev = sum(1 for d in digs if d % 2 == 0)
    od = len(digs) - ev
    return ev, od

def HFR_high_low_counts(digs, low_set={0,1,2,3,4}, high_set={5,6,7,8,9}):
    lo = sum(1 for d in digs if d in low_set)
    hi = sum(1 for d in digs if d in high_set)
    return hi, lo

def HFR_vtrac_groups(digs):
    return [HFR_V_TRAC_GROUPS[d] for d in digs]

def HFR_spread(digs):
    return max(digs) - min(digs) if digs else 0

def HFR_stringify_digits(digs):
    return ''.join(str(d) for d in digs)

def HFR_sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    return 'High'

def HFR_seed_profile(digs):
    s = HFR_sum_of_digits(digs)
    ev, od = HFR_parity_counts(digs)
    hi, lo = HFR_high_low_counts(digs)
    vt = HFR_vtrac_groups(digs)
    spr = HFR_spread(digs)
    return {
        "seed_str": HFR_stringify_digits(digs),
        "sum": s,
        "sum_category": HFR_sum_category(s),
        "even": ev,
        "odd": od,
        "high": hi,
        "low": lo,
        "vtrac": vt,
        "spread": spr,
    }

def HFR_similarity_score(p_now: dict, p_hist: dict, weights=None):
    if weights is None:
        weights = {
            "sum": 1.0, "sum_category": 1.5, "even": 0.6, "odd": 0.6,
            "high": 0.8, "low": 0.8, "spread": 0.8, "vtrac": 1.2
        }
    score = 0.0
    score += weights["sum"] * (1.0 - min(abs(p_now["sum"] - p_hist["sum"]) / 45.0, 1.0))
    score += weights["spread"] * (1.0 - min(abs(p_now["spread"] - p_hist["spread"]) / 9.0, 1.0))
    score += weights["even"] * (1.0 - min(abs(p_now["even"] - p_hist["even"]) / 5.0, 1.0))
    score += weights["odd"] * (1.0 - min(abs(p_now["odd"] - p_hist["odd"]) / 5.0, 1.0))
    score += weights["high"] * (1.0 - min(abs(p_now["high"] - p_hist["high"]) / 5.0, 1.0))
    score += weights["low"] * (1.0 - min(abs(p_now["low"] - p_hist["low"]) / 5.0, 1.0))
    score += weights["sum_category"] * (1.0 if p_now["sum_category"] == p_hist["sum_category"] else 0.0)
    now_v = Counter(p_now["vtrac"]); hist_v = Counter(p_hist["vtrac"])
    overlap = sum(min(now_v[k], hist_v[k]) for k in now_v.keys() & hist_v.keys())
    score += weights["vtrac"] * (overlap / 5.0)
    return score

def HFR_safe_eval_expression(expr: str, context: dict) -> bool:
    return bool(eval(expr, {"__builtins__": {}}, dict(context)))

def HFR_render():
    import streamlit as st
    st.markdown("---")
    st.header("Historical Safety Recommender (Additive)")
    st.caption("Ranks filters by how safely they preserved winners in *historically similar* seed contexts.")

    c1, c2 = st.columns(2)
    with c1:
        hist_file = st.file_uploader("Upload seed/winner history CSV or TXT", type=["csv","txt"],
                                     help="Order is preserved as provided (chronological *or* reverse).")
        reverse_hint = st.checkbox("History file is reverse chronological (newest first)", value=True)
    with c2:
        filt_file = st.file_uploader("Upload Filters CSV (id + expression required)", type=["csv"])

    seed_input = st.text_input("Current seed (5 digits, e.g., 27500)", value="27500")
    seed_digits_now = HFR_digits_from_str(seed_input)
    if len(seed_digits_now) != 5:
        st.error("Seed must be 5 digits.")
        return
    seed_prof_now = HFR_seed_profile(seed_digits_now)
    with st.expander("Seed profile", expanded=False):
        st.json(seed_prof_now)

    with st.expander("Similarity weights & neighborhood", expanded=False):
        w_sum = st.slider("Weight: Sum", 0.0, 3.0, 1.0, 0.1)
        w_sumcat = st.slider("Weight: Sum Category", 0.0, 3.0, 1.5, 0.1)
        w_even = st.slider("Weight: Even Count", 0.0, 3.0, 0.6, 0.1)
        w_odd = st.slider("Weight: Odd Count", 0.0, 3.0, 0.6, 0.1)
        w_high = st.slider("Weight: High Count", 0.0, 3.0, 0.8, 0.1)
        w_low = st.slider("Weight: Low Count", 0.0, 3.0, 0.8, 0.1)
        w_spread = st.slider("Weight: Spread", 0.0, 3.0, 0.8, 0.1)
        w_vtrac = st.slider("Weight: V-Trac Overlap", 0.0, 3.0, 1.2, 0.1)
        weights = {"sum": w_sum, "sum_category": w_sumcat, "even": w_even, "odd": w_odd,
                   "high": w_high, "low": w_low, "spread": w_spread, "vtrac": w_vtrac}
        k_neighbors = st.slider("Top-K neighbors", 10, 300, 100, 5)
        min_similarity = st.slider("Minimum similarity", 0.0, 8.0, 4.0, 0.1)

    run = st.button("Run Historical Safety")
    if not run:
        return

    if not hist_file:
        st.error("Upload a history file.")
        return
    if not filt_file:
        st.error("Upload a filters CSV.")
        return

    # Load history (robust)
    try:
        if hist_file.name.lower().endswith(".txt"):
            raw = hist_file.read().decode("utf-8", errors="ignore").strip().splitlines()
            rows = []
            for line in raw:
                parts = [p.strip() for p in line.replace("\\t", " ").replace("  ", " ").split(",")]
                if len(parts) == 1:
                    parts = [p.strip() for p in line.split()]
                if len(parts) >= 2:
                    rows.append(parts[:3])
            df_hist = pd.DataFrame(rows, columns=["seed","winner","date"][:len(rows[0])])
        else:
            df_hist = pd.read_csv(hist_file)
    except Exception as e:
        st.error(f"Could not read history: {e}")
        return

    # Normalize columns
    cols_lower = {c.lower(): c for c in df_hist.columns}
    seed_col = next((cols_lower[k] for k in ["seed","prev","previous","prev_seed","seed_value"] if k in cols_lower), None)
    winner_col = next((cols_lower[k] for k in ["winner","result","current","draw","win_value","result_value"] if k in cols_lower), None)
    if seed_col is None or winner_col is None:
        st.error(f"History file must include seed & winner columns. Found: {list(df_hist.columns)}")
        return
    df_hist = df_hist[[seed_col, winner_col] + [c for c in df_hist.columns if c not in (seed_col, winner_col)]].copy()
    df_hist.rename(columns={seed_col:"seed", winner_col:"winner"}, inplace=True)
    df_hist["seed_digits"] = df_hist["seed"].apply(HFR_digits_from_str)
    df_hist["winner_digits"] = df_hist["winner"].apply(HFR_digits_from_str)
    df_hist = df_hist[df_hist["seed_digits"].apply(len) == 5]
    df_hist = df_hist[df_hist["winner_digits"].apply(len) == 5].reset_index(drop=True)

    # Profiles
    df_hist["seed_sum"] = df_hist["seed_digits"].apply(HFR_sum_of_digits)
    df_hist["seed_sum_category"] = df_hist["seed_sum"].apply(HFR_sum_category)
    df_hist["seed_even"] = df_hist["seed_digits"].apply(lambda d: HFR_parity_counts(d)[0])
    df_hist["seed_odd"] = df_hist["seed_digits"].apply(lambda d: HFR_parity_counts(d)[1])
    df_hist["seed_high"] = df_hist["seed_digits"].apply(lambda d: HFR_high_low_counts(d)[0])
    df_hist["seed_low"] = df_hist["seed_digits"].apply(lambda d: HFR_high_low_counts(d)[1])
    df_hist["seed_vtrac"] = df_hist["seed_digits"].apply(HFR_vtrac_groups)
    df_hist["seed_spread"] = df_hist["seed_digits"].apply(HFR_spread)

    # Similarities
    sims = []
    for _, row in df_hist.iterrows():
        ph = {k: row[k] for k in ["seed_sum","seed_sum_category","seed_even","seed_odd","seed_high","seed_low","seed_vtrac","seed_spread"]}
        # rename keys to match p_now structure
        ph = {
            "sum": ph["seed_sum"],
            "sum_category": ph["seed_sum_category"],
            "even": ph["seed_even"],
            "odd": ph["seed_odd"],
            "high": ph["seed_high"],
            "low": ph["seed_low"],
            "vtrac": ph["seed_vtrac"],
            "spread": ph["seed_spread"],
        }
        sims.append(HFR_similarity_score(seed_prof_now, ph, weights))
    df_hist["similarity"] = sims

    nbrs = df_hist[df_hist["similarity"] >= min_similarity].copy().sort_values("similarity", ascending=False).head(k_neighbors)
    if nbrs.empty:
        st.warning("No historical neighbors met the threshold. Adjust weights/thresholds.")
        return
    st.success(f"Using {len(nbrs)} similar historical seeds.")

    # Load filters
    try:
        df_filters = pd.read_csv(filt_file)
    except Exception as e:
        st.error(f"Could not read filters CSV: {e}")
        return
    cols_lower = {c.lower(): c for c in df_filters.columns}
    id_col = next((cols_lower[k] for k in ["id","filter_id","fid"] if k in cols_lower), None)
    expr_col = next((cols_lower[k] for k in ["expression","expr","rule"] if k in cols_lower), None)
    name_col = next((cols_lower[k] for k in ["name","layman_explanation","layman","description","title"] if k in cols_lower), None) or id_col
    if id_col is None or expr_col is None:
        st.error("Filters CSV must include at least 'id' and 'expression'.")
        return
    df_filters = df_filters[[id_col, name_col, expr_col] + [c for c in df_filters.columns if c not in (id_col, name_col, expr_col)]].copy()
    df_filters.rename(columns={id_col:"id", name_col:"name", expr_col:"expression"}, inplace=True)

    # Evaluate historical safety
    results = []
    for _, f in df_filters.iterrows():
        fid, fname, expr = str(f["id"]), str(f["name"]), str(f["expression"]).strip()
        if not expr:
            continue
        eliminated, total = 0, 0
        for _, row in nbrs.iterrows():
            seed_d = row["seed_digits"]; win_d = row["winner_digits"]
            context = {
                "seed_digits": seed_d, "winner_digits": win_d,
                "seed_sum": HFR_sum_of_digits(seed_d), "winner_sum": HFR_sum_of_digits(win_d),
                "seed_even": HFR_parity_counts(seed_d)[0], "seed_odd": HFR_parity_counts(seed_d)[1],
                "winner_even": HFR_parity_counts(win_d)[0], "winner_odd": HFR_parity_counts(win_d)[1],
                "seed_high": HFR_high_low_counts(seed_d)[0], "seed_low": HFR_high_low_counts(seed_d)[1],
                "winner_high": HFR_high_low_counts(win_d)[0], "winner_low": HFR_high_low_counts(win_d)[1],
                "seed_vtrac": HFR_vtrac_groups(seed_d), "winner_vtrac": HFR_vtrac_groups(win_d),
                "seed_spread": HFR_spread(seed_d), "winner_spread": HFR_spread(win_d),
                "mirror": HFR_MIRROR_PAIRS, "vtrac_map": HFR_V_TRAC_GROUPS,
                # convenience aliases used by many CSVs:
                "combo_digits": win_d, "combo_sum": HFR_sum_of_digits(win_d), "combo_spread": HFR_spread(win_d)
            }
            try:
                would_eliminate = HFR_safe_eval_expression(expr, context)
            except Exception:
                would_eliminate = False
            total += 1
            if would_eliminate:
                eliminated += 1
        safety = 1.0 - (eliminated/total if total else 0.0)
        results.append({
            "id": fid, "name": fname, "expression": expr,
            "similar_cases": total, "eliminated_winner_in_similar": eliminated,
            "historical_safety": round(safety, 6)
        })
    if not results:
        st.error("No evaluable filters found.")
        return

    import pandas as pd
    res = pd.DataFrame(results).sort_values(["historical_safety","similar_cases"], ascending=[False, False]).reset_index(drop=True)
    st.subheader("Recommended Filters — Historical Safety First")
    st.dataframe(res, hide_index=True, use_container_width=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("Download Historical Safety CSV", res.to_csv(index=False).encode("utf-8"),
                       file_name=f"historical_recommendations_{ts}.csv", mime="text/csv")

# ---- Sidebar toggle to render HFR without altering existing UI ----
try:
    import streamlit as st
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Historical Safety")
        _HFR_SHOW = st.checkbox("Show Historical Safety Recommender", value=False,
                                help="Adds a safety-ranked filter recommender based on historically similar seeds.")
    if _HFR_SHOW:
        HFR_render()
except Exception:
    # Do not fail the app if Streamlit isn't in context at import time.
    pass
# =================== END HFR ADDITIVE SECTION ===================
