# file: filter_picker_pro.py
import io, csv, ast, math, re, textwrap
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
import streamlit as st

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

# legacy 08/09 literal sanitizer, used if an expression fails to parse/eval
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
mirror = MIRROR
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS
mirrir = MIRROR  # common typo seen in old rows

# ───────────────────────────────────────────────────────────────────────────────
# 1) UI  ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Filter Picker — Universal", layout="wide")
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
    # csv: look for a 5-digit token per row; txt: scan all 5-digit tokens
    digs = re.findall(r"\b\d{5}\b", body)
    return digs

raw_hist = _read_history(hist_file)
if raw_hist and chronology == "Latest→Earliest":
    raw_hist = list(reversed(raw_hist))

# For keep% we need windows (prev_prev, prev, seed(current))
def _winner_triples(arr: List[str]) -> List[Tuple[str,str,str]]:
    triples = []
    for i in range(2, len(arr)):
        triples.append((arr[i-2], arr[i-1], arr[i]))
    return triples

triples = _winner_triples(raw_hist)

# ───────────────────────────────────────────────────────────────────────────────
# 5) Context generator – mirrors tester page names  ─────────────────────────────
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
# 6) Compile rows (report skips), compute metrics  ──────────────────────────────
def _compiles(rows):
    compiled, skipped = [], []
    for r in rows:
        try:
            # quick eval against a harmless context to ensure both sides execute
            dummy = gen_ctx_for_combo("00000","00000","00000","00000")
            _eval(r["app_code"], dummy)
            _eval(r["expr_code"], dummy)
            compiled.append(r)
        except Exception as e:
            skipped.append((r["id"], r["name"], str(e), r.get("raw_app",""), r["raw_expr"]))
    return compiled, skipped

compiled_rows, skipped_rows = _compiles(filters_csv)

# Applicable IDs summary (only if user pasted ids)
if ids_text.strip():
    requested = {t.strip().upper() for t in re.split(r"[,\s]+", ids_text) if t.strip()}
    compiled_ids = {r["id"].upper() for r in compiled_rows}
    matched = len(requested & compiled_ids)
    st.info(f"Matched Applicable IDs: {matched} / {len(requested)}")
    missing = sorted(requested - compiled_ids)
    if missing:
        with st.expander("Missing requested IDs (not found or failed to compile)", expanded=False):
            st.code(", ".join(missing), language="text")

# Status + skipped report with CSV download
st.success(f"Compiled {len(compiled_rows)} expressions; Skipped {len(skipped_rows)}.")

with st.expander("Skipped rows (reason + expression)", expanded=False):
    # table preview
    if skipped_rows:
        import pandas as pd
        sk_df = pd.DataFrame(skipped_rows, columns=["id","name","reason","applicable_if","expression"])
        st.dataframe(sk_df, use_container_width=True, height=320)
        # downloadable CSV
        buff = io.StringIO()
        sk_df.to_csv(buff, index=False)
        st.download_button(
            "Download skipped report (.csv)",
            buff.getvalue().encode("utf-8"),
            file_name="skipped_filters_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("No skipped rows 🎉")

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
    # history
    for r in compiled_rows:
        keep = 0; n = 0
        for prev2, prev1, seed in triples:
            # "Does this filter eliminate NEXT winner when applied with this seed?"
            ctx = gen_ctx_for_combo(seed, prev1, prev2, seed)  # combo is winner == seed
            fired = filter_fires(r, ctx)
            # keep == winner survives filter ⇒ not fired
            keep += (not fired)
            n += 1
        p, lb, ub = wilson_ci(keep, n)

        # elim on pool
        elim = 0
        latest_seed_for_pool = triples[-1][2] if triples else (pool[0] if pool else "00000")
        prev1_for_pool = triples[-1][1] if triples else ""
        prev2_for_pool = triples[-1][0] if triples else ""
        for c in pool:
            ctx = gen_ctx_for_combo(latest_seed_for_pool, prev1_for_pool, prev2_for_pool, c)
            if filter_fires(r, ctx): elim += 1
        elim_rate = elim / max(1, len(pool))

        # WPP (alpha=1.0 by default)
        alpha = 1.0
        wpp = p * (elim_rate ** alpha)

        metrics.append({
            "id": r["id"], "name": r["name"],
            "keep%": p, "keep_lb": lb, "keep_ub": ub,
            "elim%": elim_rate, "wpp": wpp, "row": r
        })

# Table
st.subheader("Per-filter metrics")
if metrics:
    import pandas as pd
    df = pd.DataFrame(metrics).sort_values(["wpp","keep_lb","elim%"], ascending=[False,False,False])
    fmt = df.copy()
    for col in ["keep%","keep_lb","keep_ub","elim%","wpp"]:
        fmt[col] = (fmt[col]*100).round(2)
    st.dataframe(fmt[["id","name","keep%","keep_lb","keep_ub","elim%","wpp"]], use_container_width=True, height=420)
else:
    st.info("Load CSV + history + pool, then click Compute / Rebuild.")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# 7) Greedy bundle  ─────────────────────────────────────────────────────────────
# Helper: apply a set of filters to pool, return survivors count and ids kept
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

# Helper: estimate bundle keep% from individual keep% (assumes weak dependence)
def bundle_keep_est(selected):
    if not selected: return 1.0
    prod_elim = 1.0
    for m in selected:
        prod_elim *= (1.0 - m["keep%"])
    return 1.0 - prod_elim

latest_seed = triples[-1][2] if triples else (pool[0] if pool else "00000")
prev1 = triples[-1][1] if triples else ""
prev2 = triples[-1][0] if triples else ""

# Greedy forward selection with redundancy penalty via Jaccard on pool firing
# Precompute firing sets on pool for diversity
pool_fires: Dict[str,set] = {}
for m in metrics:
    fired = set()
    for c in pool:
        if filter_fires(m["row"], gen_ctx_for_combo(latest_seed, prev1, prev2, c)):
            fired.add(c)
    pool_fires[m["id"]] = fired

selected, selected_rows = [], []
current_survivors = len(pool)

# Sort candidates by wpp descending to start
candidates = sorted(metrics, key=lambda x: (x["wpp"], x["keep_lb"]), reverse=True)

while candidates:
    best_delta = 0
    best = None
    # try each candidate and pick the one that reduces survivors the most,
    # while maintaining bundle keep >= min_keep
    for m in candidates:
        trial = selected + [m]
        # rough bundle keep (product model)
        est_keep = bundle_keep_est(trial)
        if est_keep < min_keep:  # respect safety constraint
            continue
        # redundancy penalty (prefer low overlap on fired sets)
        if selected:
            overlap = max((len(pool_fires[m["id"]] & pool_fires[s["id"]]) /
                          max(1,len(pool_fires[m["id"]] | pool_fires[s["id"]]))) for s in selected)
        else:
            overlap = 0.0
        pen = 1.0 - redundancy_pen*overlap

        # estimate survivors by counting unique fired union
        union_fired = set().union(*[pool_fires[s["id"]] for s in trial])
        est_survivors = len(pool) - len(union_fired)
        delta = (current_survivors - est_survivors) * pen
        if delta > best_delta:
            best_delta, best = delta, m

    if not best:
        break

    selected.append(best)
    selected_rows.append(best["row"])
    # update survivors estimate quickly
    current_survivors = len(pool) - len(set().union(*[pool_fires[s["id"]] for s in selected]))
    # stop if target reached
    if current_survivors <= target_survivors:
        break
    # remove chosen from candidates
    candidates = [c for c in candidates if c["id"] != best["id"]]

# Final, exact survivors for the bundle
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

# Downloads (no reset)
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

# Preview survivors on screen
with st.expander("Survivors (preview)"):
    st.code(", ".join(final_survivors), language="text")
