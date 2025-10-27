# file: filter_picker_pro.py
import io, csv, ast, math, re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# 0) Safe built-ins + utility shims  (mirrors tester page)
ALLOWED_BUILTINS = {
    "len": len, "sum": sum, "any": any, "all": all,
    "set": set, "range": range, "sorted": sorted,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "tuple": tuple, "list": list, "dict": dict,
    "zip": zip, "map": map, "enumerate": enumerate,
    "Counter": Counter, "math": math,
}

# ord shim
def ord(x):
    try:
        return __builtins__.ord(x)
    except Exception:
        return 0

# sanitize 08/09 etc
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

# Helpers used by CSV
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

# V-TRAC & mirror maps + aliases
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR_PAIRS = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
MIRROR = MIRROR_PAIRS
mirror = MIRROR
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS
mirrir = MIRROR  # typo alias

# ───────────────────────────────────────────────────────────────────────────────
# 1) UI
st.set_page_config(page_title="Filter Picker — Universal", layout="wide")
st.title("Filter Picker (Hybrid I/O) — Universal CSV")

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

ids_text = st.text_input(
    "Optional: paste applicable filter IDs (comma/space separated). Leave blank to consider all compiled rows."
)

cA, cB, cC = st.columns([1,1,1])
with cA:
    min_keep = st.slider("Min winner survival (bundle)", 0.50, 0.99, 0.75, 0.01)
with cB:
    target_survivors = st.number_input("Target survivors (bundle)", 10, 2000, 50, 10)
with cC:
    redundancy_pen = st.slider("Redundancy penalty (diversity)", 0.0, 1.0, 0.2, 0.05)

run_btn = st.button("Compute / Rebuild", type="primary", use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Parse pool
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
# 3) Load universal filter CSV (same as tester)
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
        applicable = (row.get('applicable_if') or "True").strip().strip("'").strip('"')
        expr = (row.get('expression') or "False").strip().strip("'").strip('"')
        enabled = _enabled_value(row.get('enabled',''))

        # compile if possible, else keep str (we'll eval with sanitizer)
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

# If IDs pasted, restrict to those (but we also need to know what was missing)
requested_ids: set[str] = set()
if ids_text.strip():
    requested_ids = {t.strip().upper() for t in re.split(r"[,\s]+", ids_text) if t.strip()}

all_csv_ids = {r["id"].upper() for r in filters_csv}
missing_ids = sorted(list(requested_ids - all_csv_ids)) if requested_ids else []

# Apply restriction for compile if user pasted IDs
rows_for_compile = (
    [r for r in filters_csv if (not requested_ids or r["id"].upper() in requested_ids)]
)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Parse history + build winner triples
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
    triples = []
    for i in range(2, len(arr)):
        triples.append((arr[i-2], arr[i-1], arr[i]))
    return triples

triples = _winner_triples(raw_hist)

# ───────────────────────────────────────────────────────────────────────────────
# 5) Context generator (names mirrored to tester)
def gen_ctx_for_combo(seed:str, prev:str, prev2:str, combo:str, hot=None, cold=None, due=None):
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
        "Counter": Counter
    }
    if hot is not None:  ctx["hot_digits"]  = list(hot)
    if cold is not None: ctx["cold_digits"] = list(cold)
    if due is not None:  ctx["due_digits"]  = list(due)
    return ctx

# ───────────────────────────────────────────────────────────────────────────────
# 6) Compile rows (report skips), compute metrics
def _compiles(rows):
    compiled, skipped = [], []
    for r in rows:
        try:
            dummy = gen_ctx_for_combo("00000","00000","00000","00000")
            _eval(r["app_code"], dummy)
            _eval(r["expr_code"], dummy)
            compiled.append(r)
        except Exception as e:
            skipped.append((r["id"], r["name"], str(e), r.get("raw_app",""), r["raw_expr"]))
    return compiled, skipped

compiled_rows, skipped_rows = _compiles(rows_for_compile)

# ID status panel — compiled / skipped / missing (for pasted IDs)
if requested_ids:
    compiled_ids = {r["id"].upper() for r in compiled_rows}
    skipped_ids  = {sid.upper() for sid,_,_,_,_ in skipped_rows}
    missing_ids  = sorted(list(requested_ids - (compiled_ids | skipped_ids)))

    st.info(f"Requested IDs matched: {len(compiled_ids|skipped_ids)} / {len(requested_ids)}")
    with st.expander("ID Status (requested list)"):
        status_rows = []
        for fid in sorted(requested_ids):
            if fid in compiled_ids:
                status_rows.append({"id": fid, "status": "compiled", "reason": ""})
            elif fid in skipped_ids:
                # find reason
                reason = next((r for (sid,_,r,_,_) in skipped_rows if sid.upper()==fid), "")
                status_rows.append({"id": fid, "status": "skipped", "reason": reason})
            else:
                status_rows.append({"id": fid, "status": "missing", "reason": "Not found in CSV"})
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, height=260)

        # Download missing/skipped id lists
        miss = [r["id"] for r in status_rows if r["status"]=="missing"]
        skip = [r["id"] for r in status_rows if r["status"]=="skipped"]
        st.download_button(
            "Download missing IDs (.txt)",
            ("\n".join(miss)+"\n").encode("utf-8"),
            file_name="missing_ids.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.download_button(
            "Download skipped IDs (.txt)",
            ("\n".join(skip)+"\n").encode("utf-8"),
            file_name="skipped_ids.txt",
            mime="text/plain",
            use_container_width=True
        )

# Status + skipped report with CSV download
st.success(f"Compiled {len(compiled_rows)} expressions; Skipped {len(skipped_rows)}.")

with st.expander("Skipped rows (reason + expression)", expanded=False):
    if skipped_rows:
        sk_df = pd.DataFrame(skipped_rows, columns=["id","name","reason","applicable_if","expression"])
        st.dataframe(sk_df, use_container_width=True, height=320)
        buff = io.StringIO()
        sk_df.to_csv(buff, index=False)
        st.download_button(
            "Download skipped report (.csv)",
            buff.getvalue().encode("utf-8"),
            file_name="skipped_filters_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        # Also quick copyable list
        st.text_area("Skipped IDs (comma-separated)", ", ".join(sk_df["id"].astype(str).tolist()), height=80)
    else:
        st.caption("No skipped rows 🎉")

# ───────────────────────────────────────────────────────────────────────────────
# Keep% (winner survival on history) and Elim% (thinning on current pool)
def wilson_ci(k, n, z=1.96):
    if n==0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z**2/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))

def filter_fires(row, ctx) -> bool:
    try:
        if not _eval(row["app_code"], ctx): return False
        return bool(_eval(row["expr_code"], ctx))
    except Exception:
        return False

# Compute per-filter metrics
metrics = []
if compiled_rows and triples and pool:
    for r in compiled_rows:
        keep = 0; n = 0
        for prev2, prev1, seed in triples:
            ctx = gen_ctx_for_combo(seed, prev1, prev2, seed)  # winner as combo
            fired = filter_fires(r, ctx)
            keep += (not fired)
            n += 1
        p, lb, ub = wilson_ci(keep, n)

        elim = 0
        latest_seed_for_pool = triples[-1][2] if triples else (pool[0] if pool else "00000")
        prev1_for_pool = triples[-1][1] if triples else ""
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
# 7) Greedy bundle
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

latest_seed = triples[-1][2] if triples else (pool[0] if pool else "00000")
prev1 = triples[-1][1] if triples else ""
prev2 = triples[-1][0] if triples else ""

# Precompute pool firing sets for diversity
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
    st.write(f"Filters chosen ({len(selected)}):",
             ", ".join(m['id'] for m in selected))
    st.write(f"Projected winner survival (product model): "
             f"{bundle_keep_est(selected):.2%}")
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
