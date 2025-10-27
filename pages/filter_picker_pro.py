# file: filter_picker_pro.py
import io, csv, ast, math, re, unicodedata
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# 0) Safe built-ins + utility shims

ALLOWED_BUILTINS = {
    "len": len, "sum": sum, "any": any, "all": all,
    "set": set, "range": range, "sorted": sorted,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "tuple": tuple, "list": list, "dict": dict,
    "zip": zip, "map": map, "enumerate": enumerate,
    "Counter": Counter, "math": math,
}

def ord_safe(x):
    try:
        if isinstance(x, str) and len(x) == 1:
            return __builtins__.ord(x)
        if isinstance(x, int):
            return __builtins__.ord(str(x))
        s = str(x)
        return __builtins__.ord(s[0]) if s else 0
    except Exception:
        return 0

def ord(x):  # noqa: A001
    return ord_safe(x)

_leading_zero_int = re.compile(r'(?<![\w])0+(\d+)(?!\s*\.)')

def _sanitize_numeric_literals(code: str) -> str:
    return _leading_zero_int.sub(r"\1", code)

def _normalize_unicode(code: str) -> str:
    if not isinstance(code, str):
        return code
    s = unicodedata.normalize("NFKC", code)
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s

def _preprocess_expr(raw: str) -> str:
    return _sanitize_numeric_literals(_normalize_unicode(raw))

def _eval(code_or_obj, ctx):
    if isinstance(code_or_obj, str):
        src = _preprocess_expr(code_or_obj)
    else:
        src = code_or_obj
    g = {"__builtins__": ALLOWED_BUILTINS, "ord": ord_safe}
    return eval(src, g, ctx)

def sum_category(total: int) -> str:
    if 0 <= total <= 14:  return 'Very Low'
    if 15 <= total <= 20: return 'Low'
    if 21 <= total <= 26: return 'Mid'
    return 'High'

def structure_of(digits: List[int]) -> str:
    c = sorted(Counter(digits).values(), reverse=True)
    if c == [1,1,1,1,1]: return 'SINGLE'
    if c == [2,1,1,1]:   return 'DOUBLE'
    if c == [2,2,1]:     return 'DOUBLE-DOUBLE'
    if c == [3,1,1]:     return 'TRIPLE'
    if c == [3,2]:       return 'TRIPLE-DOUBLE'
    if c == [4,1]:       return 'QUAD'
    if c == [5]:         return 'QUINT'
    return f'OTHER-{c}'

# V-TRAC & MIRROR maps
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR_PAIRS  = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def mirror_of(x):
    try:
        if isinstance(x, str): x = int(x)
        return MIRROR_PAIRS[int(x)]
    except Exception:
        return None

# aliases seen in CSVs
MIRROR = MIRROR_PAIRS
mirror = MIRROR
mirrir = MIRROR
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS

PRIMES = {2,3,5,7}
EVENS  = {0,2,4,6,8}
ODDS   = {1,3,5,7,9}

# ───────────────────────────────────────────────────────────────────────────────
# 1) UI

st.set_page_config(page_title="Filter Picker — Universal (CF + LL sets)", layout="wide")
st.title("Filter Picker (Hybrid I/O) — Universal CSV + CF/LL variables")

with st.expander("Paste current pool — supports continuous digits", expanded=True):
    pool_text = st.text_area(
        "Pool (comma-separated or one per line). Duplicates ignored.",
        height=200,
        placeholder="e.g. 01234, 98765, ...  or  01234\n98765\n..."
    )

c1, c2, c3 = st.columns([1.2,1,1])
with c1:
    filt_file = st.file_uploader("Upload master filter CSV", type=["csv"])
with c2:
    hist_file = st.file_uploader("Upload history (TXT or CSV)", type=["txt","csv"])
with c3:
    chronology = st.radio("History order", ["Earliest→Latest", "Latest→Earliest"], index=0, horizontal=True)

ids_text = st.text_input("Optional: paste applicable filter IDs (comma/space separated). Leave blank to consider all compiled rows.")

cA, cB, cC = st.columns([1,1,1])
with cA:
    min_keep = st.slider("Min winner survival (bundle)", 0.50, 0.99, 0.75, 0.01)
with cB:
    target_survivors = st.number_input("Target survivors (bundle)", 10, 2000, 50, 10)
with cC:
    redundancy_pen = st.slider("Redundancy penalty (diversity)", 0.0, 1.0, 0.2, 0.05)

st.button("Compute / Rebuild", type="primary", use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Pool

def _norm_pool(text: str) -> List[str]:
    if not text: return []
    raw = text.replace("\r"," ").replace("\n", ",")
    toks = [t.strip() for t in raw.split(",")]
    out = []
    for t in toks:
        if not t: continue
        s = "".join(ch for ch in t if ch.isdigit())
        if len(s)==5: out.append(s)
    return sorted(set(out))

pool = _norm_pool(pool_text)

# ───────────────────────────────────────────────────────────────────────────────
# 3) Load CSV

def _enabled_value(val: str) -> bool:
    s = (val or "").strip().lower()
    return s in {'"""true"""','"true"','true','1','yes','y'}

def load_filters_csv(file) -> List[Dict[str,Any]]:
    if not file: return []
    data = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    reader = csv.DictReader(io.StringIO(data))
    out = []
    for raw in reader:
        row = { (k or "").strip().lower(): (v or "").strip() for k,v in raw.items() }
        fid = (row.get('id') or row.get('fid') or "").strip()
        name = (row.get('name') or "").strip()
        applicable = _preprocess_expr((row.get('applicable_if') or "True").strip().strip("'").strip('"'))
        expr = _preprocess_expr((row.get('expression') or "False").strip().strip("'").strip('"'))
        enabled = _enabled_value(row.get('enabled',''))

        try:
            ast.parse(f"({applicable})", mode='eval'); app_code = compile(applicable, '<app>', 'eval')
        except SyntaxError:
            app_code = applicable
        try:
            ast.parse(f"({expr})", mode='eval'); expr_code = compile(expr, '<expr>', 'eval')
        except SyntaxError:
            expr_code = expr

        out.append({"id":fid, "name":name, "enabled":enabled,
                    "app_code":app_code, "expr_code":expr_code,
                    "raw_app":applicable, "raw_expr":expr})
    return out

filters_csv = load_filters_csv(filt_file)

requested_ids: set[str] = set()
if ids_text.strip():
    requested_ids = {t.strip().upper() for t in re.split(r"[,\s]+", ids_text) if t.strip()}

all_csv_ids = {r["id"].upper() for r in filters_csv}
missing_ids = sorted(list(requested_ids - all_csv_ids)) if requested_ids else []

rows_for_compile = [r for r in filters_csv if (not requested_ids or r["id"].upper() in requested_ids)]

# ───────────────────────────────────────────────────────────────────────────────
# 4) History, letters (prev vs current), CF/LL sets

def _read_history(file) -> List[str]:
    if not file: return []
    body = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    digs = re.findall(r"\b\d{5}\b", body)
    return digs

raw_hist = _read_history(hist_file)
if raw_hist and chronology == "Latest→Earliest":
    raw_hist = list(reversed(raw_hist))

def _winner_triples(arr: List[str]) -> List[Tuple[str,str,str]]:
    return [(arr[i-2], arr[i-1], arr[i]) for i in range(2, len(arr))]

triples = _winner_triples(raw_hist)

def build_letter_mapping(hist: List[str]) -> Dict[int, str]:
    if not hist:
        return {d: "ABCDEFGHIJ"[d] for d in range(10)}
    ctr = Counter(int(ch) for w in hist for ch in w)
    order = sorted(range(10), key=lambda d: (-ctr[d], d))  # freq desc, digit asc
    letters = list("ABCDEFGHIJ")
    return {d: letters[i] for i, d in enumerate(order)}

# current mapping from all history; previous mapping from history excluding latest seed (if available)
current_map = build_letter_mapping(raw_hist)
prev_hist   = raw_hist[:-1] if len(raw_hist) >= 1 else raw_hist
previous_map = build_letter_mapping(prev_hist)

# helpers: letter rank index (A=1..J=10)
def letter_rank(letter: str) -> int:
    return "ABCDEFGHIJ".index(letter) + 1 if letter in "ABCDEFGHIJ" else 10

# derive core letters for prev & current seeds (if history provided)
prev_seed  = triples[-1][1] if triples else ""
seed_now   = triples[-1][2] if triples else (pool[0] if pool else "00000")

def core_letters_from_seed(seed: str, mapping: Dict[int,str]) -> List[str]:
    return [mapping[int(ch)] for ch in seed] if seed else []

prev_core_letters = core_letters_from_seed(prev_seed, previous_map)
core_letters      = core_letters_from_seed(seed_now,  current_map)

# digit->letter lookups (int & str keys)
digit_current_letters = {d: current_map[d] for d in range(10)}
digit_current_letters.update({str(d): current_map[d] for d in range(10)})

digit_previous_letters = {d: previous_map[d] for d in range(10)}
digit_previous_letters.update({str(d): previous_map[d] for d in range(10)})

# cooled digits: rank moved toward cooler (higher index) vs previous
cooled_digits_int = []
for d in range(10):
    r_prev = letter_rank(previous_map[d])
    r_cur  = letter_rank(current_map[d])
    if r_cur > r_prev:
        cooled_digits_int.append(d)
cooled_digits_str = [str(d) for d in cooled_digits_int]

# ring digits: rank adjacent (±1) to ranks of any current core digit
seed_digits_int = [int(x) for x in seed_now] if seed_now else []
core_ranks = {letter_rank(current_map[d]) for d in seed_digits_int}
ring_digits_int = []
for d in range(10):
    r = letter_rank(current_map[d])
    if any(abs(r - cr) == 1 for cr in core_ranks):
        ring_digits_int.append(d)
ring_digits_str = [str(d) for d in ring_digits_int]

# new_core_digits: in current seed but not in previous seed
prev_seed_digits_int = [int(x) for x in prev_seed] if prev_seed else []
new_core_digits_int = sorted(set(seed_digits_int) - set(prev_seed_digits_int))
new_core_digits_str = [str(d) for d in new_core_digits_int]

# loser_7_9 set (both types)
loser_7_9_int = [7,8,9]
loser_7_9_str = ["7","8","9"]

# quick hot/cold from history (optional)
ctr_all = Counter(int(ch) for w in raw_hist for ch in w) if raw_hist else Counter()
freq_order = [d for d,_ in ctr_all.most_common()] + [d for d in range(10) if d not in ctr_all]
hot_digits_int  = freq_order[:4] if freq_order else []
cold_digits_int = freq_order[-4:] if freq_order else []
hot_digits_str  = [str(d) for d in hot_digits_int]
cold_digits_str = [str(d) for d in cold_digits_int]

# ───────────────────────────────────────────────────────────────────────────────
# 5) Context generator (now includes *_s variants + all CF/LL sets)

def gen_ctx_for_combo(seed:str, prev:str, prev2:str, combo:str):
    seed_digits = [int(x) for x in seed] if seed else []
    prev_digits = [int(x) for x in prev] if prev else []
    prev2_digits = [int(x) for x in prev2] if prev2 else []
    cdigits = [int(x) for x in combo] if combo else []

    ctx = {
        # basic winners/combos
        "seed_value": int(seed) if seed else 0,
        "seed_sum": sum(seed_digits),
        "seed_sum_last_digit": (sum(seed_digits) % 10) if seed_digits else 0,
        "prev_seed_sum": sum(prev_digits) if prev_digits else None,
        "prev_prev_seed_sum": sum(prev2_digits) if prev2_digits else None,
        "seed_digits": seed_digits,
        "seed_digits_s": list(seed) if seed else [],
        "prev_seed_digits": prev_digits,
        "prev_prev_seed_digits": prev2_digits,
        "combo_digits": cdigits,
        "combo_digits_s": list(combo) if combo else [],
        "combo_sum": sum(cdigits),
        "combo_structure": structure_of(cdigits),
        "winner_structure": structure_of(seed_digits),
        # vtrac/mirror
        "MIRROR": MIRROR_PAIRS, "mirror": MIRROR_PAIRS, "mirrir": MIRROR_PAIRS,
        "MIRROR_PAIRS": MIRROR_PAIRS, "mirror_of": mirror_of,
        "V_TRAC_GROUPS": V_TRAC_GROUPS, "VTRAC_GROUPS": V_TRAC_GROUPS,
        "V_TRAC": V_TRAC_GROUPS, "VTRAC_GROUP": V_TRAC_GROUPS, "vtrac": V_TRAC_GROUPS,
        # helpers
        "sum_category": sum_category, "structure_of": structure_of,
        "Counter": Counter, "PRIMES": PRIMES, "EVENS": EVENS, "ODDS": ODDS,
        "ord": ord_safe,
        # CF letters
        "digit_current_letters": digit_current_letters,
        "digit_previous_letters": digit_previous_letters,
        "core_letters": core_letters,
        "prev_core_letters": prev_core_letters,
        # LL-style sets (both int and string lists available)
        "cooled_digits": cooled_digits_int,
        "cooled_digits_s": cooled_digits_str,
        "ring_digits": ring_digits_int,
        "ring_digits_s": ring_digits_str,
        "new_core_digits": new_core_digits_int,
        "new_core_digits_s": new_core_digits_str,
        "loser_7_9": loser_7_9_int,
        "loser_7_9_s": loser_7_9_str,
        "hot_digits": hot_digits_int,
        "hot_digits_s": hot_digits_str,
        "cold_digits": cold_digits_int,
        "cold_digits_s": cold_digits_str,
    }
    return ctx

# ───────────────────────────────────────────────────────────────────────────────
# 6) Compile rows, status, skipped report

def _compiles(rows):
    compiled, skipped = [], []
    dummy = gen_ctx_for_combo(seed_now, prev_seed, "", seed_now)
    for r in rows:
        try:
            _eval(r["raw_app"], dummy)
            _eval(r["raw_expr"], dummy)
            compiled.append(r)
        except Exception as e:
            skipped.append((r["id"], r["name"], str(e), r.get("raw_app",""), r["raw_expr"]))
    return compiled, skipped

compiled_rows, skipped_rows = _compiles(rows_for_compile)

if requested_ids:
    compiled_ids = {r["id"].upper() for r in compiled_rows}
    skipped_ids  = {sid.upper() for sid,_,_,_,_ in skipped_rows}
    missing_now  = sorted(list(requested_ids - (compiled_ids | skipped_ids)))
    st.info(f"Requested IDs matched: {len(compiled_ids|skipped_ids)} / {len(requested_ids)}")
    with st.expander("ID Status (requested list)", expanded=False):
        status_rows = []
        for fid in sorted(requested_ids):
            if fid in compiled_ids:
                status_rows.append({"id": fid, "status": "compiled", "reason": ""})
            elif fid in skipped_ids:
                reason = next((r for (sid,_,r,_,_) in skipped_rows if sid.upper()==fid), "")
                status_rows.append({"id": fid, "status": "skipped", "reason": reason})
            else:
                status_rows.append({"id": fid, "status": "missing", "reason": "Not found in CSV"})
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, height=260)
        miss = [r["id"] for r in status_rows if r["status"]=="missing"]
        skip = [r["id"] for r in status_rows if r["status"]=="skipped"]
        st.download_button("Download missing IDs (.txt)", ("\n".join(miss)+"\n").encode("utf-8"),
                           file_name="missing_ids.txt", mime="text/plain", use_container_width=True)
        st.download_button("Download skipped IDs (.txt)", ("\n".join(skip)+"\n").encode("utf-8"),
                           file_name="skipped_ids.txt", mime="text/plain", use_container_width=True)

st.success(f"Compiled {len(compiled_rows)} expressions; Skipped {len(skipped_rows)}.")

with st.expander("Skipped rows (reason + expression)", expanded=True):
    if skipped_rows:
        sk_df = pd.DataFrame(skipped_rows, columns=["id","name","reason","applicable_if","expression"])
        st.dataframe(sk_df, use_container_width=True, height=280)
        reason_counts = (sk_df["reason"].fillna("").str.extract(r"^([^:]+):?")[0]
                         .value_counts().reset_index())
        reason_counts.columns = ["reason_head", "count"]
        st.caption("Grouped reasons:")
        st.dataframe(reason_counts, use_container_width=True, height=160)
        buff = io.StringIO()
        sk_df.to_csv(buff, index=False)
        st.download_button("Download skipped report (.csv)", buff.getvalue().encode("utf-8"),
                           file_name="skipped_filters_report.csv", mime="text/csv",
                           use_container_width=True)
        st.text_area("Skipped IDs (comma-separated)",
                     ", ".join(sk_df["id"].astype(str).tolist()), height=80)
    else:
        st.caption("No skipped rows 🎉")

# ───────────────────────────────────────────────────────────────────────────────
# 7) Metrics

def wilson_ci(k, n, z=1.96):
    if n==0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z**2/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))

def filter_fires(row, ctx) -> bool:
    try:
        if not _eval(row["raw_app"], ctx): return False
        return bool(_eval(row["raw_expr"], ctx))
    except Exception:
        return False

metrics = []
if compiled_rows and triples and pool:
    for r in compiled_rows:
        keep = 0; n = 0
        for prev2, prev1, seed in triples:
            ctx = gen_ctx_for_combo(seed, prev1, prev2, seed)
            fired = filter_fires(r, ctx)
            keep += (not fired)
            n += 1
        p, lb, ub = wilson_ci(keep, n)

        elim = 0
        latest_seed_for_pool = seed_now
        prev1_for_pool = prev_seed
        prev2_for_pool = triples[-1][0] if triples else ""
        for c in pool:
            ctx = gen_ctx_for_combo(latest_seed_for_pool, prev1_for_pool, prev2_for_pool, c)
            if filter_fires(r, ctx): elim += 1
        elim_rate = elim / max(1, len(pool))

        alpha = 1.0
        wpp = p * (elim_rate ** alpha)

        metrics.append({
            "id": r["id"], "name": r["name"],
            "keep%": p, "keep_lb": lb, "keep_ub": ub,
            "elim%": elim_rate, "wpp": wpp, "row": r
        })

st.subheader("Per-filter metrics")
if metrics:
    df = pd.DataFrame(metrics).sort_values(["wpp","keep_lb","elim%"], ascending=[False,False,False])
    fmt = df.copy()
    for col in ["keep%","keep_lb","keep_ub","elim%","wpp"]:
        fmt[col] = (fmt[col]*100).round(2)
    st.dataframe(fmt[["id","name","keep%","keep_lb","keep_ub","elim%","wpp"]],
                 use_container_width=True, height=420)
else:
    st.info("Load CSV + history + pool, then click Compute / Rebuild.")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# 8) Greedy bundle

def apply_filters_bundle(rows: List[Dict[str,Any]], pool: List[str],
                         latest_seed: str, prev1: str, prev2: str) -> Tuple[int, List[str]]:
    survivors = []
    for c in pool:
        ctx = gen_ctx_for_combo(latest_seed, prev1, prev2, c)
        fired = any(filter_fires(r, ctx) for r in rows)
        if not fired: survivors.append(c)
    return len(survivors), survivors

def bundle_keep_est(selected):
    if not selected: return 1.0
    prod_elim = 1.0
    for m in selected:
        prod_elim *= (1.0 - m["keep%"])
    return 1.0 - prod_elim

latest_seed = seed_now
prev1 = prev_seed
prev2 = triples[-1][0] if triples else ""

# precompute pool firing sets
pool_fires: Dict[str,set] = {}
for m in metrics:
    fired = set()
    for c in pool:
        if filter_fires(m["row"], gen_ctx_for_combo(latest_seed, prev1, prev2, c)):
            fired.add(c)
    pool_fires[m["id"]] = fired

selected, selected_rows = [], []
current_survivors = len(pool)
candidates = sorted(metrics, key=lambda x: (x["wpp"], x["keep_lb"]), reverse=True)

while candidates:
    best_delta = 0; best = None
    for m in candidates:
        trial = selected + [m]
        est_keep = bundle_keep_est(trial)
        if est_keep < min_keep:
            continue
        if selected:
            overlap = max(
                (len(pool_fires[m["id"]] & pool_fires[s["id"]]) /
                 max(1,len(pool_fires[m["id"]] | pool_fires[s["id"]]))) for s in selected
            )
        else:
            overlap = 0.0
        pen = 1.0 - redundancy_pen*overlap
        union_fired = set().union(*[pool_fires[s["id"]] for s in trial])
        est_survivors = len(pool) - len(union_fired)
        delta = (current_survivors - est_survivors) * pen
        if delta > best_delta:
            best_delta, best = delta, m
    if not best:
        break
    selected.append(best)
    selected_rows.append(best["row"])
    current_survivors = len(pool) - len(set().union(*[pool_fires[s["id"]] for s in selected]))
    if current_survivors <= target_survivors:
        break
    candidates = [c for c in candidates if c["id"] != best["id"]]

final_n, final_survivors = apply_filters_bundle(selected_rows, pool, latest_seed, prev1, prev2)

st.subheader("Selected bundle")
if selected:
    st.write(f"Filters chosen ({len(selected)}):", ", ".join(m['id'] for m in selected))
    st.write(f"Projected winner survival (product model): {bundle_keep_est(selected):.2%}")
    st.write(f"Projected survivors (pool): {final_n}")
else:
    st.info("No bundle found that satisfies the minimum survival. Lower the threshold or add history.")

colL, colR = st.columns(2)
with colL:
    ids_txt = "\n".join(m["id"] for m in selected)
    st.download_button("Download bundle IDs (.txt)", ids_txt.encode("utf-8"),
                       file_name="bundle_ids.txt", use_container_width=True)
with colR:
    surv_txt = ", ".join(final_survivors)
    st.download_button(f"Download projected survivors ({final_n}).txt",
                       surv_txt.encode("utf-8"),
                       file_name=f"survivors_{final_n}.txt",
                       use_container_width=True)

with st.expander("Survivors (preview)"):
    st.code(", ".join(final_survivors), language="text")
