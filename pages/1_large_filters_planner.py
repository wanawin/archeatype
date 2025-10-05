
# 1_large_filters_planner_UNIFIED.py
# Full Streamlit app — unified build with:
#  • Original planner UI
#  • 4-draws-back inputs for filters
#  • Tester-style spelling variations normalization
#  • Compile-once evaluation model
#  • H/C/D, Mirror, VTRAC v4 wired directly from UI
#
# Run: streamlit run 1_large_filters_planner_UNIFIED.py

from __future__ import annotations
import io, re, math, random, unicodedata
from typing import Dict, List, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

# ---------------- Page ----------------
st.set_page_config(page_title="Large Filters Planner — Unified", layout="wide")
st.title("Large Filters Planner — Unified")

# ---------------- Constants / Maps ----------------
# VTRAC v4
VTRAC: Dict[int, int]  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
V_TRAC_GROUPS: Dict[int, int] = VTRAC  # legacy alias

# Mirror pairs
MIRROR: Dict[int, int] = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
mirror = MIRROR  # alias, some CSVs use lowercase

SAFE_BUILTINS = {
    "abs": abs, "int": int, "str": str, "float": float, "round": round,
    "len": len, "sum": sum, "max": max, "min": min, "any": any, "all": all,
    "set": set, "sorted": sorted, "list": list, "tuple": tuple, "dict": dict,
    "range": range, "enumerate": enumerate, "map": map, "filter": filter,
    "math": math, "re": re, "random": random, "Counter": Counter,
    "True": True, "False": False, "None": None,
}

# ---------------- Basic helpers ----------------
def parse_list_any(text: str) -> List[str]:
    if not text: return []
    raw = text.replace("\t", ",").replace("\n", ",").replace(";", ",").replace(" ", ",")
    return [p.strip() for p in raw.split(",") if p.strip()]

def digits_of(s: str) -> List[int]:
    s = str(s).strip()
    return [int(ch) for ch in s if ch.isdigit()]

def safe_digits(x):
    try:
        return [int(ch) for ch in str(x) if ch.isdigit()]
    except Exception:
        return []

def digit_sum(x):
    ds = safe_digits(x)
    return sum(ds) if ds else 0

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

def spread_band(spread: int) -> str:
    if spread <= 3: return "0–3"
    if spread <= 5: return "4–5"
    if spread <= 7: return "6–7"
    if spread <= 9: return "8–9"
    return "10+"

# H/C/D helpers
def _mk_is_hot(env):  return lambda d: (str(d).isdigit() and int(d) in env.get("hot_set", set()))
def _mk_is_cold(env): return lambda d: (str(d).isdigit() and int(d) in env.get("cold_set", set()))
def _mk_is_due(env):  return lambda d: (str(d).isdigit() and int(d) in env.get("due_set", set()))

def even_count(x): return sum(1 for d in safe_digits(x) if d % 2 == 0)
def odd_count(x):  return sum(1 for d in safe_digits(x) if d % 2 == 1)
def high_count(x): return sum(1 for d in safe_digits(x) if d >= 5)
def low_count(x):  return sum(1 for d in safe_digits(x) if d <= 4)

def first_digit(x): ds = safe_digits(x); return ds[0] if ds else None
def last_digit(x):  ds = safe_digits(x); return ds[-1] if ds else None
def last_two_digits(x): ds = safe_digits(x); return ds[-2:] if len(ds) >= 2 else ds
def unique_count(x): return len(set(safe_digits(x)))
def max_repeat(x):  c = Counter(safe_digits(x)); return max(c.values()) if c else 0
def has_triplet(x): return max_repeat(x) >= 3
def digit_span(x):  ds = safe_digits(x); return (max(ds) - min(ds)) if ds else 0

def vtrac_of(d):
    try:
        d = int(d); return VTRAC[d] if d in VTRAC else None
    except Exception:
        return None

def count_vtrac_groups(x) -> int:
    return len(set(VTRAC[d] for d in safe_digits(x) if d in VTRAC))

def contains_mirror_pair(x):
    s = set(safe_digits(x))
    return any((d in s and MIRROR[d] in s and MIRROR[d] != d) for d in s)

def count_in_hot(x, hot_set=None):
    hs = hot_set if hot_set is not None else set(st.session_state.get("hot_digits", []))
    return sum(1 for d in safe_digits(x) if d in hs)

def count_in_cold(x, cold_set=None):
    cs = cold_set if cold_set is not None else set(st.session_state.get("cold_digits", []))
    return sum(1 for d in safe_digits(x) if d in cs)

def count_in_due(x, due_set=None):
    ds = due_set if due_set is not None else set(st.session_state.get("due_digits", []))
    return sum(1 for d in safe_digits(x) if d in ds)

def percent_hot(x, hot_set=None):
    ds = safe_digits(x);  return (100.0 * count_in_hot(ds, hot_set) / len(ds)) if ds else 0.0
def percent_cold(x, cold_set=None):
    ds = safe_digits(x); return (100.0 * count_in_cold(ds, cold_set) / len(ds)) if ds else 0.0
def percent_due(x, due_set=None):
    ds = safe_digits(x);  return (100.0 * count_in_due(ds, due_set) / len(ds)) if ds else 0.0

# ---------------- Variation handling (tester-style) ----------------
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def _camel_to_snake(s: str) -> str:
    return _CAMEL_RE.sub('_', s).lower()

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
        text = re.sub(rf"\b{k}\b", v, text)
    return text

_VARIATION_MAP: Dict[str, str] = {
    # hot/cold/due
    "hotDigits": "hot_digits", "hotdigits": "hot_digits", "hotnumbers": "hot_digits", "hot": "hot_digits",
    "coldDigits": "cold_digits", "colddigits": "cold_digits", "coldnumbers": "cold_digits", "cold": "cold_digits",
    "dueDigits": "due_digits", "duedigits": "due_digits", "duenumbers": "due_digits", "due": "due_digits",
    "percentHot": "percent_hot(combo_digits, hot_set)",
    "percentCold": "percent_cold(combo_digits, cold_set)",
    "percentDue": "percent_due(combo_digits, due_set)",
    "countHot": "sum(1 for d in combo_digits if d in hot_set)",
    "countCold": "sum(1 for d in combo_digits if d in cold_set)",
    "countDue": "sum(1 for d in combo_digits if d in due_set)",
    # mirror
    "mirrorDigits": "combo_mirror_digits", "mirrordigits": "combo_mirror_digits", "mirrors": "combo_mirror_digits",
    "mirrorSet": "set(combo_mirror_digits)", "mirrorset": "set(combo_mirror_digits)",
    "mirrorPairs": "contains_mirror_pair(combo_digits)", "hasMirrorPair": "contains_mirror_pair(combo_digits)",
    "hasmirrorpair": "contains_mirror_pair(combo_digits)", "hasmirror": "contains_mirror_pair(combo_digits)",
    # combo / seed
    "comboDigits": "combo_digits", "combodigits": "combo_digits",
    "comboSet": "set(combo_digits)", "comboset": "set(combo_digits)",
    "seedDigits": "seed_digits", "seeddigits": "seed_digits",
    "seedSet": "set(seed_digits)", "seedset": "set(seed_digits)",
    # parity / counts
    "parityEven": "combo_sum_is_even", "parityeven": "combo_sum_is_even",
    "isEven": "combo_sum_is_even", "iseven": "combo_sum_is_even",
    "isOdd": "not combo_sum_is_even", "isodd": "not combo_sum_is_even",
    "evenCount": "even_count(combo_digits)", "oddCount": "odd_count(combo_digits)",
    "highCount": "high_count(combo_digits)", "lowCount": "low_count(combo_digits)",
    "isAllHigh": "all(d >= 5 for d in combo_digits)", "isallhigh": "all(d >= 5 for d in combo_digits)",
    "isAllLow": "all(d <= 4 for d in combo_digits)", "isalllow": "all(d <= 4 for d in combo_digits)",
    # positional
    "firstDigit": "first_digit(combo_digits)", "firstdigit": "first_digit(combo_digits)",
    "lastDigit": "last_digit(combo_digits)", "lastdigit": "last_digit(combo_digits)",
    "lastTwo": "last_two_digits(combo_digits)", "lasttwo": "last_two_digits(combo_digits)", "last2": "last_two_digits(combo_digits)",
    # sums / spreads / structure
    "sumDigits": "digit_sum(combo_digits)", "sumdigits": "digit_sum(combo_digits)",
    "digitSum": "digit_sum(combo_digits)", "digitsum": "digit_sum(combo_digits)",
    "sum": "digit_sum(combo_digits)",
    "spreadBand": "spread_band(spread)", "spreadband": "spread_band(spread)",
    "structure": "combo_structure", "struct": "combo_structure",
    # vtrac
    "vtrack": "VTRAC", "vtracks": "VTRAC", "vtracGroups": "VTRAC", "vtracgroups": "VTRAC",
    "vtracSet": "combo_vtracs", "vtracset": "combo_vtracs",
    "comboVtracs": "combo_vtracs", "combovtracs": "combo_vtracs",
    "vtracLast": "combo_last_vtrac", "vtraclast": "combo_last_vtrac",
    "lastVtrac": "combo_last_vtrac", "lastvtrac": "combo_last_vtrac",
}

def normalize_expr(expr: str) -> str:
    if not expr: return ""
    x = _ascii(expr)
    # optional camel→snake for known tokens
    for t in ["hotDigits","coldDigits","dueDigits","mirrorPairs","mirrorDigits",
              "seedDigits","comboDigits","percentHot","percentCold","percentDue",
              "evenCount","oddCount","highCount","lowCount",
              "firstDigit","lastDigit","lastTwo","last2",
              "digitSum","sumDigits","vtracSet","vtracLast","vtracGroups","spreadBand",
              "comboSum","comboSet","seedSet","isEven","isOdd","parityEven"]:
        if t in x:
            x = x.replace(t, _camel_to_snake(t))
    x = _wb_replace(x, _VARIATION_MAP)
    return x

def _clean_expr(s: str) -> str:
    s = str(s or "").strip().strip('"').strip("'")
    s = s.replace("!==", "!=")
    return normalize_expr(s)

# ---------------- CSV loaders ----------------
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
        applicable = rr.get("applicable_if", "True") or "True"
        expr = rr.get("expression", "False") or "False"
        try:
            rr["applicable_code"] = compile(applicable, "<applicable>", "eval")
            rr["expr_code"] = compile(expr, "<expr>", "eval")
        except SyntaxError as e:
            rr["enabled"] = False
            rr["compile_error"] = str(e)
        rows.append(rr)
    out = pd.DataFrame(rows)
    enabled_norm = out["enabled"].apply(lambda v: str(v).strip().lower() in ("1","true","t","yes","y"))
    out = out[enabled_norm].copy()
    return out

# ---------------- Env builders (4 draws back) ----------------
def make_base_env(seed: str, prev_seed: str, prev_prev_seed: str, prev_prev_prev_seed: str,
                  hot_digits: List[int], cold_digits: List[int], due_digits: List[int]) -> Dict:
    sd  = digits_of(seed) if seed else []
    sd2 = digits_of(prev_seed) if prev_seed else []
    sd3 = digits_of(prev_prev_seed) if prev_prev_seed else []
    sd4 = digits_of(prev_prev_prev_seed) if prev_prev_prev_seed else []

    env = {
        "seed_digits": sd,
        "prev_seed_digits": sd2,
        "prev_prev_seed_digits": sd3,
        "prev_prev_prev_seed_digits": sd4,

        "VTRAC": VTRAC, "V_TRAC_GROUPS": V_TRAC_GROUPS,
        "mirror": MIRROR, "MIRROR": MIRROR,

        "hot_digits": sorted(set(hot_digits)),
        "cold_digits": sorted(set(cold_digits)),
        "due_digits": sorted(set(due_digits)),
        "hot_set": set(hot_digits), "cold_set": set(cold_digits), "due_set": set(due_digits),

        "sum_category": sum_category, "structure_of": classify_structure,
        "safe_digits": safe_digits, "digit_sum": digit_sum,
        "even_count": even_count, "odd_count": odd_count, "high_count": high_count, "low_count": low_count,
        "first_digit": first_digit, "last_digit": last_digit, "last_two_digits": last_two_digits,
        "unique_count": unique_count, "max_repeat": max_repeat, "has_triplet": has_triplet,
        "digit_span": digit_span, "contains_mirror_pair": contains_mirror_pair,
        "vtrac_of": vtrac_of, "count_vtrac_groups": count_vtrac_groups,
        "count_in_hot": count_in_hot, "count_in_cold": count_in_cold, "count_in_due": count_in_due,
        "percent_hot": percent_hot, "percent_cold": percent_cold, "percent_due": percent_due,
        **SAFE_BUILTINS,

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
    cset = set(cd)
    env.update({
        "combo": combo,
        "combo_digits": cd,
        "combo_set": cset,
        "combo_sum": sum(cd),
        "combo_sum_is_even": (sum(cd) % 2 == 0),
        "combo_last_digit": cd[-1] if cd else None,
        "combo_structure": classify_structure(cd),
        "hot_set": set(env.get("hot_digits", [])),
        "cold_set": set(env.get("cold_digits", [])),
        "due_set": set(env.get("due_digits", [])),
        "combo_mirror_digits": [MIRROR[d] for d in cd] if cd else [],
        "contains_mirror_pair": contains_mirror_pair(cd),
        "combo_vtracs": set(VTRAC[d] for d in cd) if cd else set(),
        "combo_last_vtrac": (VTRAC[cd[-1]] if cd else None),
    })
    env["is_hot"]  = _mk_is_hot(env)
    env["is_cold"] = _mk_is_cold(env)
    env["is_due"]  = _mk_is_due(env)
    return env

# ---------------- Filter evaluation ----------------
def eval_applicable(row: pd.Series, base_env: Dict) -> bool:
    try:
        return bool(eval(row["applicable_code"], {"__builtins__": {}}, base_env))
    except Exception:
        return True

def eval_filter_on_pool(row: pd.Series, pool: List[str], base_env: Dict) -> Tuple[Set[str], int]:
    eliminated: Set[str] = set()
    code = row["expr_code"]
    for c in pool:
        env = combo_env(base_env, c)
        try:
            if bool(eval(code, {"__builtins__": {}}, env)):
                eliminated.add(c)
        except Exception:
            pass
    return eliminated, len(eliminated)

# ---------------- Planner ----------------
def greedy_plan(candidates: pd.DataFrame, pool: List[str], base_env: Dict,
                beam_width: int, max_steps: int) -> Tuple[List[Dict], List[str]]:
    remaining = set(pool)
    chosen: List[Dict] = []
    for step in range(int(max_steps)):
        if not remaining: break
        scored = []
        for _, r in candidates.iterrows():
            if not eval_applicable(r, base_env):
                continue
            elim, cnt = eval_filter_on_pool(r, list(remaining), base_env)
            if cnt > 0:
                scored.append((cnt, elim, r))
        if not scored: break
        scored.sort(key=lambda x: x[0], reverse=True)
        best_cnt, best_elim, best_row = scored[0]
        remaining -= best_elim
        chosen.append({
            "id": best_row["id"],
            "name": best_row.get("name", ""),
            "expression": best_row["expression"],
            "eliminated_this_step": best_cnt,
            "remaining_after": len(remaining),
        })
        if best_cnt == 0: break
    return chosen, sorted(list(remaining))

# ---------------- UI ----------------
st.subheader("Hot / Cold / Due digits")
d1, d2, d3 = st.columns(3)
hot_digits  = [int(x) for x in parse_list_any(d1.text_input("Hot digits (comma-separated)")) if x.isdigit()]
cold_digits = [int(x) for x in parse_list_any(d2.text_input("Cold digits (comma-separated)")) if x.isdigit()]
due_digits  = [int(x) for x in parse_list_any(d3.text_input("Due digits (comma-separated)")) if x.isdigit()]
auto_hcd = st.checkbox("Only if ALL boxes are blank: auto-fill Hot/Cold/Due from recent history", value=True)

st.session_state["hot_digits"] = hot_digits
st.session_state["cold_digits"] = cold_digits
st.session_state["due_digits"]  = due_digits

st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos (CSV ‘Result’ column OR tokens separated by newline/space/comma):", height=140)
pool_file = st.file_uploader("Or upload pool CSV (‘Result’ or ‘combo’ column)", type=["csv"])
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

st.subheader("Winners History")
hc1, hc2 = st.columns([2,1])
history_path = hc1.text_input("Path to winners history CSV", value="DC5_Midday_Full_Cleaned_Expanded.csv")
history_upload = hc2.file_uploader("Or upload history CSV", type=["csv"], key="hist_up")

winners_list: List[str] = []
if history_upload is not None:
    try:
        winners_list = load_winners_csv_from_path(history_upload)
    except Exception as e:
        st.warning(f"Uploaded history CSV failed: {e}. Will try path.")
if not winners_list:
    try:
        winners_list = load_winners_csv_from_path(history_path)
    except Exception as e:
        st.warning(f"History path failed: {e}. Continuing without history safety.")

user_provided_hcd = bool(hot_digits or cold_digits or due_digits)
if not user_provided_hcd and auto_hcd and winners_list:
    AUTO_WINDOW = 10
    hist_digits = [digits_of(x) for x in winners_list][-AUTO_WINDOW:]
    flat = [d for row in hist_digits for d in row]
    cnt = Counter(flat)
    if cnt:
        most = cnt.most_common(); topk = 6
        thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
        hot_digits = sorted({d for d, c in most if c >= thresh})
        least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
        coldk = 4; cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
        cold_digits = sorted({d for d, c in least if c <= cth})
        last2 = set(d for row in hist_digits[-2:] for d in row)
        due_digits = sorted(set(range(10)) - last2)
        st.caption(f"(Auto) Hot/Cold/Due from last {AUTO_WINDOW} winners → hot={hot_digits} • cold={cold_digits} • due={due_digits}")
else:
    st.caption(f"(Using UI) hot={hot_digits} • cold={cold_digits} • due={due_digits}")

# Draw history — 4 back
st.subheader("Draw History (4 back)")
c1, c2, c3, c4 = st.columns(4)
seed      = c1.text_input("Known winner (0‑back)", value="")
prev_seed = c2.text_input("Draw 1‑back", value="")
prev_prev = c3.text_input("Draw 2‑back", value="")
prev_prev_prev = c4.text_input("Draw 3‑back", value="")

st.subheader("Filters")
fids_text = st.text_area("Paste applicable Filter IDs (optional; comma / space / newline separated):", height=90)
filters_pasted_csv = st.text_area("Paste Filters CSV content (optional):", height=150, help="If provided, this CSV is used (must include 'expression').")
filters_file_up = st.file_uploader("Or upload Filters CSV (used if pasted CSV is empty)", type=["csv"])
filters_csv_path = st.text_input("Or path to Filters CSV (used if pasted/upload empty)", value="lottery_filters_batch_10.csv")

try:
    filters_df_full = load_filters_from_source(filters_pasted_csv, filters_file_up, filters_csv_path)
except Exception as e:
    st.error(f"Failed to load Filters CSV ➜ {e}"); st.stop()

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

st.write(f"**Filters loaded (pre-eval): {len(filters_df)}")
if len(filters_df) == 0:
    st.warning("0 filters available. Check CSV path/content, 'enabled' values, expressions, or ID selection.")
    st.stop()

st.sidebar.header("Mode & Thresholds")
mode = st.sidebar.radio("Mode", ["Playlist Reducer", "Safe Filter Explorer"], index=1)
min_elims_to_call_it_large = int(st.sidebar.number_input("Min eliminations to call it ‘Large’", 1, 5000, 60))
beam_width = int(st.sidebar.number_input("Greedy beam width", 1, 20, 6))
max_steps  = int(st.sidebar.number_input("Greedy max steps", 1, 50, 18))
exclude_parity_wipers = st.sidebar.checkbox("Exclude parity-wipers", value=True)

run = st.button("▶ Run Planner + Recommender", type="primary")
if not run:
    st.stop()

# Build env with 4 draws back
base_env = make_base_env(seed, prev_seed, prev_prev, prev_prev_prev, hot_digits, cold_digits, due_digits)
st.info(f"Using Hot/Cold/Due everywhere → hot={hot_digits} • cold={cold_digits} • due={due_digits}", icon="🔥")

candidates = filters_df.copy()
if exclude_parity_wipers and "parity_wiper" in candidates.columns:
    candidates = candidates[~candidates["parity_wiper"].astype(str).str.lower().isin(["1","true","t","yes","y"])]

scores = []
pool_list = list(pool)
for _, r in candidates.iterrows():
    if not eval_applicable(r, base_env):
        continue
    elim, cnt = eval_filter_on_pool(r, pool_list, base_env)
    scores.append({"id": r["id"], "name": r.get("name",""), "expression": r["expression"], "eliminated_now": cnt})

scored_df = pd.DataFrame(scores).sort_values("eliminated_now", ascending=False)
st.subheader("Filter Diagnostics (first pass)")
st.dataframe(scored_df.head(200), use_container_width=True)

chosen, remaining = greedy_plan(candidates, pool_list, base_env, beam_width, max_steps)

st.subheader("Chosen sequence (in order):")
st.dataframe(pd.DataFrame(chosen), use_container_width=True)

st.subheader("Best‑case plan — Large filters only")
st.caption("Sorted by immediate eliminations on current pool")
st.dataframe(scored_df[scored_df["eliminated_now"]>=min_elims_to_call_it_large], use_container_width=True)

st.subheader("Remaining pool after plan")
st.write(len(remaining))
st.code("\n".join(remaining[:500]))
