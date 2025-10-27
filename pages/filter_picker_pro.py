# file: filter_picker_pro.py
import io, csv, ast, math, re, textwrap
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np
import streamlit as st
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# 0) Safe built-ins + utility shims  (mirrors tester page)  ─────────────────────
ALLOWED_BUILTINS = {
    "len": len, "sum": sum, "any": any, "all": all,
    "set": set, "range": range, "sorted": sorted,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "tuple": tuple, "list": list, "dict": dict,
    "zip": zip, "map": map, "enumerate": enumerate,
    "Counter": Counter, "math": math,
}

# ord shim (some CSVs call ord() on non-chars; keep from crashing)
def ord(x):
    try:
        return __builtins__.ord(x)
    except Exception:
        return 0

# legacy 08/09 literal sanitizer, used if an expression fails to eval because of 08, 09, etc.
_leading_zero_int = re.compile(r'(?<![\w])0+(\d+)(?!\s*\.)')
def _sanitize_numeric_literals(code_or_obj):
    if isinstance(code_or_obj, str):
        return _leading_zero_int.sub(r"\1", code_or_obj)
    return code_or_obj

def _eval(code_or_obj, ctx):
    g = {"__builtins__": ALLOWED_BUILTINS}
    try:
        return eval(code_or_obj, g, ctx)
    except SyntaxError:
        return eval(_sanitize_numeric_literals(code_or_obj), g, ctx)

# Sum category + structure helpers (used by CSV)
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

# V-TRAC & mirror maps + aliases used in CSV expressions
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR_PAIRS = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
MIRROR = MIRROR_PAIRS
mirror = MIRROR  # many rows call mirror.get(d, -1)
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS
mirrir = MIRROR  # common historical typo

# ───────────────────────────────────────────────────────────────────────────────
# 1) UI  ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Filter Picker — Universal CSV", layout="wide")
st.title("Filter Picker (Hybrid I/O) — Universal CSV")

with st.expander("Paste current pool — supports continuous digits", expanded=True):
    pool_text = st.text_area(
        "Pool (comma-separated or one per line). Duplicates ignored.",
        height=220,
        placeholder="e.g. 01234, 98765, ...  or  01234\n98765\n..."
    )

c1, c2, c3 = st.columns([1.1,1,1])
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

run_btn = st.button("Compute / Rebuild", type="primary", use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Parse pool  ────────────────────────────────────────────────────────────────
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
# 3) Load universal filter CSV (same as tester)  ────────────────────────────────
def _enabled_value(val: str) -> bool:
    s = (val or "").strip().lower()
    return s in {'"""true"""','"true"','true','1','yes','y'}

def load_filters_csv(file) -> Tuple[List[Dict[str,Any]], List[Tuple[str,str,str,str]]]:
    """
    Returns (rows, syntactic_skips).
    Each row: {id,name,enabled,app_code,expr_code,raw_app,raw_expr}
    Syntactic skips: (id,name,reason,text_that_failed)
    """
    if not file: return [], []
    data = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    reader = csv.DictReader(io.StringIO(data))
    out, bad = [], []
    for raw in reader:
        row = { (k or "").strip().lower(): (v or "").strip() for k,v in raw.items() }
        fid = (row.get('id') or row.get('fid') or "").strip()
        name = (row.get('name') or "").strip()
        applicable = (row.get('applicable_if') or "True").strip().strip("'").strip('"')
        expr = (row.get('expression') or "False").strip().strip("'").strip('"')
        enabled = _enabled_value(row.get('enabled',''))

        # Only **syntax** check and store compiled objects. Do **not** execute here.
        try:
            ast.parse(f"({applicable})", mode='eval'); app_code = compile(applicable, '<app>', 'eval')
        except Exception as e:
            bad.append((fid, name, f"applicable_if parse: {e.__class__.__name__}: {e}", applicable))
            continue

        try:
            ast.parse(f"({expr})", mode='eval'); expr_code = compile(expr, '<expr>', 'eval')
        except Exception as e:
            bad.append((fid, name, f"expression parse: {e.__class__.__name__}: {e}", expr))
            continue

        out.append({"id":fid, "name":name, "enabled":enabled,
                    "app_code":app_code, "expr_code":expr_code,
                    "raw_app":applicable, "raw_expr":expr})
    return out, bad

filters_csv, syntactic_skips = load_filters_csv(filt_file)

# If IDs pasted, restrict to those
applicable_ids = set()
if ids_text.strip():
    applicable_ids = {t.strip().upper() for t in re.split(r"[,\s]+", ids_text) if t.strip()}

if applicable_ids:
    filters_csv = [r for r in filters_csv if r["id"].upper() in applicable_ids]

# ───────────────────────────────────────────────────────────────────────────────
# 4) Parse history + build winner triples  ──────────────────────────────────────
def _read_history(file) -> List[str]:
    if not file: return []
    body = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    # csv or txt: scan all 5-digit tokens
    digs = re.findall(r"\b\d{5}\b", body)
    return digs

raw_hist = _read_history(hist_file)
if raw_hist and chronology == "Latest→Earliest":
    raw_hist = list(reversed(raw_hist))

def _winner_triples(arr: List[str]) -> List[Tuple[str,str,str]]:
    triples = []
    for i in range(2, len(arr)):
        triples.append((arr[i-2], arr[i-1], arr[i]))
    return triples

triples = _winner_triples(raw_hist)

# ───────────────────────────────────────────────────────────────────────────────
# 5) Context generator – mirrors tester page names  ─────────────────────────────
def gen_ctx_for_combo(seed:str, prev:str, prev2:str, combo:str,
                      hot=None, cold=None, due=None):
    seed_digits = [int(x) for x in seed]
    prev_digits = [int(x) for x in prev] if prev else []
    prev2_digits = [int(x) for x in prev2] if prev2 else []
    cdigits = [int(x) for x in combo] if combo else []
    seed_value = int(seed)

    new_seed_digits = set(seed_digits) - set(prev_digits)
    common_to_both = set(seed_digits) & set(prev_digits)
    last2 = set(seed_digits) | set(prev_digits)
    seed_counts = Counter(seed_digits)
    seed_sum = sum(seed_digits)
    seed_vtracs = set(V_TRAC_GROUPS[d] for d in seed_digits)
    combo_sum = sum(cdigits)
    combo_vtracs = set(V_TRAC_GROUPS[d] for d in cdigits)

    ctx = {
        "seed_value": seed_value,
        "seed_sum": seed_sum,
        "seed_sum_last_digit": seed_sum % 10,
        "prev_seed_sum": sum(prev_digits) if prev_digits else None,
        "prev_prev_seed_sum": sum(prev2_digits) if prev2_digits else None,

        "seed_digits": seed_digits,
        "prev_seed_digits": prev_digits,
        "prev_prev_seed_digits": prev2_digits,

        "new_seed_digits": new_seed_digits,
        "common_to_both": common_to_both,
        "last2": last2,
        "seed_counts": seed_counts,
        "seed_vtracs": seed_vtracs,

        "combo_digits": cdigits,
        "combo_sum": combo_sum,
        "combo_vtracs": combo_vtracs,
        "combo_structure": structure_of(cdigits),
        "winner_structure": structure_of(seed_digits),

        "MIRROR": MIRROR, "mirror": MIRROR, "mirrir": MIRROR,
        "MIRROR_PAIRS": MIRROR_PAIRS,
        "V_TRAC_GROUPS": V_TRAC_GROUPS, "VTRAC_GROUPS": V_TRAC_GROUPS,
        "V_TRAC": V_TRAC_GROUPS, "VTRAC_GROUP": V_TRAC_GROUPS, "vtrac": V_TRAC_GROUPS,

        "sum_category": sum_category, "structure_of": structure_of,
        "Counter": Counter,

        # **Defaults** so filters that reference these never crash
        "hot_digits": list(hot) if hot is not None else [],
        "cold_digits": list(cold) if cold is not None else [],
        "due_digits": list(due) if due is not None else [],
    }
    return ctx

# ───────────────────────────────────────────────────────────────────────────────
# 6) Report true compile status (syntax only) + give copy/download of skips  ────
st.subheader("ID Status (requested list)")
req = len(applicable_ids) if applicable_ids else len(filters_csv)
st.caption(f"Compiled (syntax OK): {len(filters_csv)} | Skipped (syntax errors): {len(syntactic_skips)}.")

with st.expander("Skipped rows (reason + expression)", expanded=False):
    if syntactic_skips:
        df_skip = pd.DataFrame(syntactic_skips, columns=["id","name","reason","expr"])
        st.dataframe(df_skip, use_container_width=True, height=360)
        bad_ids = ", ".join(x[0] for x in syntactic_skips if x[0])
        st.code(bad_ids, language="text")
        st.download_button("Download skipped IDs (.txt)", bad_ids.encode("utf-8"),
                           file_name="skipped_ids.txt")
    else:
        st.write("No syntax errors.")

# If nothing to do, stop here
if not (filters_csv and triples and pool):
    st.info("Load CSV + history + pool (and optionally paste applicable IDs), then click Compute / Rebuild.")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# 7) Runtime evaluation helpers  ────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n==0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z**2/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))

def filter_fires(row, ctx) -> bool:
    try:
        if not _eval(row["app_code"], ctx):
            return False
        return bool(_eval(row["expr_code"], ctx))
    except Exception:
        # At runtime we still guard; if anything explodes, treat as NOT firing.
        return False

# ───────────────────────────────────────────────────────────────────────────────
# 8) Per-filter metrics  ────────────────────────────────────────────────────────
metrics = []
latest_seed = triples[-1][2]
prev1 = triples[-1][1]
prev2 = triples[-1][0]

# history (keep%), pool (elim%), WPP
for r in filters_csv:
    keep = 0; n = 0
    for pp, p1, seed in triples:
        ctx = gen_ctx_for_combo(seed, p1, pp, seed)
        fired = filter_fires(r, ctx)
        keep += (not fired)
        n += 1
    p, lb, ub = wilson_ci(keep, n)

    elim = 0
    for c in pool:
        ctx = gen_ctx_for_combo(latest_seed, prev1, prev2, c)
        if filter_fires(r, ctx): elim += 1
    elim_rate = elim / max(1, len(pool))

    alpha = 1.0
    wpp = p * (elim_rate ** alpha)

    metrics.append({
        "id": r["id"], "name": r["name"],
        "keep%": p, "keep_lb": lb, "keep_ub": ub,
        "elim%": elim_rate, "wpp": wpp, "row": r
    })

# Table
st.subheader("Per-filter metrics")
df = pd.DataFrame(metrics).sort_values(["wpp","keep_lb","elim%"], ascending=[False,False,False])
fmt = df.copy()
for col in ["keep%","keep_lb","keep_ub","elim%","wpp"]:
    fmt[col] = (fmt[col]*100).round(2)
st.dataframe(fmt[["id","name","keep%","keep_lb","keep_ub","elim%","wpp"]], use_container_width=True, height=420)

# ───────────────────────────────────────────────────────────────────────────────
# 9) Greedy bundle  ─────────────────────────────────────────────────────────────
def apply_filters_bundle(rows: List[Dict[str,Any]], pool: List[str],
                         latest_seed: str, prev1: str, prev2: str) -> Tuple[int, List[str]]:
    survivors = []
    for c in pool:
        ctx = gen_ctx_for_combo(latest_seed, prev1, prev2, c)
        fired = False
        for r in rows:
            if filter_fires(r, ctx):
                fired = True; break
        if not fired: survivors.append(c)
    return len(survivors), survivors

def bundle_keep_est(selected) -> float:
    if not selected: return 1.0
    prod_elim = 1.0
    for m in selected:
        prod_elim *= (1.0 - m["keep%"])
    return 1.0 - prod_elim

# Precompute pool firing sets to penalize redundancy
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
    best_delta = 0
    best = None
    for m in candidates:
        trial = selected + [m]
        est_keep = bundle_keep_est(trial)
        if est_keep < min_keep:
            continue

        if selected:
            overlap = max((len(pool_fires[m["id"]] & pool_fires[s["id"]]) /
                           max(1,len(pool_fires[m["id"]] | pool_fires[s["id"]]))) for s in selected)
        else:
            overlap = 0.0
        pen = 1.0 - redundancy_pen*overlap

        union_fired = set().union(*[pool_fires[s["id"]] for s in trial]) if trial else set()
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
