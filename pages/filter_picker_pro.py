# pages/filter_picker_parity.py
# Streamlit ≥ 1.28
# Full page: tester-parity filter picker with safe AST evaluator + bundle builder

import io, re, time, math, ast
from typing import List, Dict, Tuple, Set, Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Filter Picker — Tester Parity", layout="wide")

# -----------------------------
# Session defaults
# -----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("filters_df", None)
    ss.setdefault("history_winners", [])
    ss.setdefault("pool_combos", [])
    ss.setdefault("seed_str", "")
    ss.setdefault("chronology", "Newest → Oldest")
    ss.setdefault("alpha_wpp", 1.5)
    ss.setdefault("gamma_decay", 0.03)
    ss.setdefault("result_df", None)
    ss.setdefault("fired_on_winners", {})
    ss.setdefault("fired_on_pool", {})
    ss.setdefault("bundle_result", None)
    ss.setdefault("applicable_ids_raw", "")
    ss.setdefault("compile_log", [])         # list of (id, ok, msg)
    ss.setdefault("compiled_funcs", {})      # id -> callable(ctx)->bool
    ss.setdefault("var_usage", {})           # id -> set(varnames)

init_state()

# -----------------------------
# Utilities
# -----------------------------
def parse_pool_text(text: str) -> List[str]:
    if not text: return []
    if "," in text:
        parts = [re.sub(r"\D", "", p) for p in text.split(",")]
        return [p for p in parts if len(p) == 5 and p.isdigit()]
    return re.findall(r"(?<!\d)(\d{5})(?!\d)", text)

def parse_history_file(file, chronology_label: str) -> List[str]:
    raw = file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    winners = re.findall(r"(?<!\d)(\d{5})(?!\d)", text)
    return winners if chronology_label == "Newest → Oldest" else winners[::-1]

def load_filters_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}
    # Most robust mapping
    idc  = cols.get("id") or list(df.columns)[0]
    namec= cols.get("name") or (list(df.columns)[1] if len(df.columns)>1 else idc)
    exprc= cols.get("expression") or (list(df.columns)[-1])
    out = pd.DataFrame({
        "id": df[idc].astype(str),
        "name": df[namec].astype(str),
        "expression": df[exprc].astype(str)
    })
    # Normalize smart quotes/dashes
    out["expression"] = (out["expression"]
        .str.replace("“","\"", regex=False)
        .str.replace("”","\"", regex=False)
        .str.replace("’","'", regex=False)
        .str.replace("—","-", regex=False)
    )
    return out

def parse_applicable_ids(text: str) -> Set[str]:
    if not text.strip(): return set()
    return {t.strip() for t in re.split(r"[,\s]+", text.strip()) if t.strip()}

# -----------------------------
# Tester-style evaluation context
# (string digits, common helpers)
# -----------------------------
MIRROR = {"0":"5","1":"6","2":"7","3":"8","4":"9","5":"0","6":"1","7":"2","8":"3","9":"4"}
PRIMES = {'2','3','5','7'}
LOSER_7_9 = {'7','8','9'}

SAFE_FUNCS = {
    "sum": sum, "any": any, "all": all, "len": len, "set": set, "min": min, "max": max,
    "range": range, "ord": ord, "int": int, "float": float, "abs": abs
}

def build_ctx(combo: str, seed: str) -> Dict:
    combo_digits = list(combo)              # ['1','2','3','4','5']
    seed_digits  = list(seed) if seed else []
    ctx = {
        "combo_digits": combo_digits,
        "seed_digits": seed_digits,
        "loser_7_9": list(LOSER_7_9),
        "primes": list(PRIMES),
        "mirror": MIRROR,
    }
    ctx.update(SAFE_FUNCS)
    return ctx

# -----------------------------
# Safe AST compiler (tester parity)
# -----------------------------
ALLOWED_NAMES = set([
    "combo_digits","seed_digits","loser_7_9","primes","mirror",
    "sum","any","all","len","set","min","max","range","ord","int","float","abs"
])
ALLOWED_CALLS = set(["sum","any","all","len","set","min","max","range","ord","int","float","abs"])

ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.Subscript, ast.Slice, ast.Index, ast.Call, ast.comprehension,
    ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.IfExp
)

def _collect_names(node: ast.AST, out: Set[str]):
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)

def _validate_ast(tree: ast.AST) -> Optional[str]:
    for n in ast.walk(tree):
        if not isinstance(n, ALLOWED_NODES):
            return f"Disallowed syntax: {type(n).__name__}"
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in ALLOWED_CALLS:
                return f"Disallowed call: {ast.unparse(n) if hasattr(ast,'unparse') else 'call'}"
    return None

def compile_expression(expr_text: str) -> Tuple[Optional[Callable[[Dict], bool]], Set[str], str]:
    """
    Return (callable, used_names, message). callable(ctx)->bool or None if failed.
    """
    try:
        tree = ast.parse(expr_text, mode="eval")
    except Exception as e:
        return None, set(), f"Syntax error: {e}"

    err = _validate_ast(tree)
    if err:
        return None, set(), err

    used = set(); _collect_names(tree, used)
    unknown = [n for n in used if n not in ALLOWED_NAMES]
    if unknown:
        return None, used, f"Unknown names: {', '.join(sorted(set(unknown)))}"

    code = compile(tree, "<expr>", "eval")
    def _fn(ctx: Dict) -> bool:
        try:
            return bool(eval(code, {"__builtins__": {}}, ctx))
        except Exception:
            # runtime guard: treat failure as "does not fire"
            return False
    return _fn, used, "ok"

def compile_filters(df: pd.DataFrame) -> Tuple[Dict[str,Callable], Dict[str,Set[str]], List[Tuple[str,bool,str]]]:
    compiled = {}
    usage = {}
    log = []
    for _, r in df.iterrows():
        fid = str(r["id"]).strip()
        expr = str(r["expression"]).strip()
        fn, used, msg = compile_expression(expr)
        ok = fn is not None
        log.append((fid, ok, msg))
        if ok:
            compiled[fid] = fn
            usage[fid] = used
    return compiled, usage, log

# -----------------------------
# Stats & scoring
# -----------------------------
def wilson_ci(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0: return (0.0, 0.0)
    p = k/n
    denom = 1 + (z*z)/n
    centre = p + (z*z)/(2*n)
    margin = z * math.sqrt((p*(1-p) + (z*z)/(4*n))/n)
    lo = (centre - margin)/denom
    hi = (centre + margin)/denom
    return max(0.0, lo), min(1.0, hi)

def time_weights(n: int, gamma: float) -> np.ndarray:
    if n <= 0: return np.array([])
    idx = np.arange(n)
    w = np.exp(-gamma * idx)
    return w / w.sum()

def compute_scores(compiled: Dict[str,Callable],
                   filt_df: pd.DataFrame,
                   pool: List[str],
                   winners: List[str],
                   seed: str,
                   alpha: float,
                   gamma: float):
    Nw, Np = len(winners), len(pool)
    w_w = time_weights(Nw, gamma) if Nw>0 else np.array([])

    keep_rate, keep_lo, keep_hi, elim_rate = [], [], [], []
    fired_w, fired_p = {}, {}

    for _, row in filt_df.iterrows():
        fid = row["id"]; fn = compiled.get(fid)
        if fn is None:
            # Filter not compiled; treat as no-op
            keep_rate += [0.0]; keep_lo += [0.0]; keep_hi += [0.0]; elim_rate += [0.0]
            fired_w[fid] = set(); fired_p[fid] = set()
            continue

        # winners
        fw = set(); kept_mass = 0.0
        for i, w in enumerate(winners):
            fires = fn(build_ctx(w, seed))
            if fires: fw.add(i)
            else: kept_mass += (w_w[i] if Nw>0 else 1.0)
        fired_w[fid] = fw
        if Nw>0:
            kr = kept_mass if w_w.size>0 else (Nw - len(fw)) / Nw
            keep_rate.append(kr)
            lo, hi = wilson_ci(Nw-len(fw), Nw)
            keep_lo.append(lo); keep_hi.append(hi)
        else:
            keep_rate.append(0.0); keep_lo.append(0.0); keep_hi.append(0.0)

        # pool
        fp = set()
        for j, c in enumerate(pool):
            if fn(build_ctx(c, seed)): fp.add(j)
        fired_p[fid] = fp
        elim_rate.append(len(fp)/Np if Np>0 else 0.0)

    out = filt_df.copy()
    out["keep_rate"] = keep_rate
    out["keep_lo"]   = keep_lo
    out["keep_hi"]   = keep_hi
    out["elim_rate"] = elim_rate
    out["WPP"] = out["keep_rate"] * (out["elim_rate"] ** float(alpha))
    out = out.sort_values(["WPP","keep_rate","elim_rate"], ascending=[False,False,False]).reset_index(drop=True)
    return out, fired_w, fired_p

def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def build_bundle(scored: pd.DataFrame,
                 fired_w: Dict[str,Set[int]],
                 fired_p: Dict[str,Set[int]],
                 Nw: int,
                 w_weights: np.ndarray,
                 min_survival: float,
                 target_survivors: Optional[int],
                 redundancy_penalty: float = 0.2):
    selected = []
    U_w, U_p = set(), set()
    hist = []

    def survival(union_w: Set[int]) -> float:
        if Nw == 0: return 0.0
        if w_weights.size == 0:
            return 1.0 - (len(union_w)/Nw)
        mask = np.ones(Nw, dtype=bool)
        if union_w:
            idx = np.fromiter(union_w, dtype=int, count=len(union_w))
            mask[idx] = False
        return w_weights[mask].sum()

    pool_size = max(0, max([max(s) if s else -1 for s in fired_p.values()] + [-1]) + 1)
    cur_surv = survival(U_w)

    ids = scored["id"].tolist()
    while True:
        best = None; best_score = -1e9; best_Uw = U_w; best_Up = U_p; best_surv = cur_surv
        for fid in ids:
            if fid in selected: continue
            candUw = U_w | fired_w.get(fid,set())
            candUp = U_p | fired_p.get(fid,set())
            candSurv = survival(candUw)
            if candSurv < min_survival: continue

            # penalize redundancy on winners overlap
            if selected:
                S = fired_w.get(fid,set())
                max_j = max((jaccard(S, fired_w.get(sid,set())) for sid in selected), default=0.0)
            else:
                max_j = 0.0
            delta = len(candUp) - len(U_p)
            score = delta - redundancy_penalty * max_j * max(1, len(candUp))
            if score > best_score:
                best_score = score; best = fid; best_Uw = candUw; best_Up = candUp; best_surv = candSurv

        if best is None: break
        selected.append(best); U_w = best_Uw; U_p = best_Up; cur_surv = best_surv
        survivors = pool_size - len(U_p)
        hist.append({"added": best, "survival": cur_surv, "survivors": survivors})
        if target_survivors is not None and survivors <= target_survivors:
            break

    return {
        "selected": selected,
        "survival": cur_surv,
        "survivors": (pool_size - len(U_p)),
        "history": hist
    }

# -----------------------------
# UI
# -----------------------------
st.title("Filter Picker — Tester Parity (Safe)")

with st.expander("How to use"):
    st.markdown("""
1) **Upload tester filter CSV** (same one the tester uses).  
2) **Paste your pool** (commas or continuous digits).  
3) **Upload history** (CSV/TXT), set **chronology**, and enter **seed**.  
4) (Optional) **Paste Applicable IDs** to limit the candidate set.  
5) Click **RUN**. If you change files/IDs/seed, click RUN again.  
6) Use **Build greedy bundle** with your survival floor and target survivors (e.g., 50).  
   Try **Feasibility Probe** or **Push-to-50** if you stall.
""")

# Inputs
c1, c2 = st.columns([1.7, 1.3])
with c1:
    pool_text = st.text_area("Paste current pool (commas or continuous)", height=160,
                             placeholder="88001,87055,04510,…  or a single blob like 8800187055…")
    applicable_ids_raw = st.text_area("Applicable filter IDs (optional — comma/space/newline-separated)",
                                      value=st.session_state.applicable_ids_raw, height=120)
with c2:
    chronology = st.radio("History chronology", ["Newest → Oldest","Oldest → Newest"])
    seed_str = st.text_input("Seed (last 5 digits)", value=st.session_state.seed_str, max_chars=5)
    alpha = st.slider("WPP α (thin weight)", 0.2, 2.0, float(st.session_state.alpha_wpp), 0.1)
    gamma = st.slider("Time-decay γ (history)", 0.0, 0.10, float(st.session_state.gamma_decay), 0.01)

u1, u2, u3 = st.columns([1,1,1])
with u1:
    filters_file = st.file_uploader("Upload tester filter CSV", type=["csv"])
with u2:
    history_file = st.file_uploader("Upload winners history (CSV or TXT)", type=["csv","txt"])
with u3:
    run_clicked = st.button("🚀 RUN (compile + score)", type="primary", use_container_width=True)

# Persist
st.session_state.chronology = chronology
st.session_state.seed_str = re.sub(r"\D","",seed_str)[:5]
st.session_state.alpha_wpp = alpha
st.session_state.gamma_decay = gamma

if pool_text:
    st.session_state.pool_combos = parse_pool_text(pool_text)
if filters_file is not None:
    st.session_state.filters_df = load_filters_csv(filters_file)
if history_file is not None:
    st.session_state.history_winners = parse_history_file(history_file, st.session_state.chronology)
if applicable_ids_raw is not None:
    st.session_state.applicable_ids_raw = applicable_ids_raw

# Metrics
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Parsed pool size", len(st.session_state.pool_combos))
m2.metric("Filters (CSV)", 0 if st.session_state.filters_df is None else len(st.session_state.filters_df))
m3.metric("History rows", len(st.session_state.history_winners))
m4.metric("Seed", st.session_state.seed_str if st.session_state.seed_str else "—")

# RUN
if run_clicked:
    if not st.session_state.pool_combos:
        st.warning("Please paste the current pool."); st.stop()
    if st.session_state.filters_df is None or st.session_state.filters_df.empty:
        st.warning("Please upload the tester filter CSV."); st.stop()
    if not st.session_state.history_winners:
        st.warning("Please upload a winners history file."); st.stop()
    if not st.session_state.seed_str or len(st.session_state.seed_str) != 5:
        st.warning("Please enter a 5-digit seed."); st.stop()

    use_df = st.session_state.filters_df.copy()
    subset = parse_applicable_ids(st.session_state.applicable_ids_raw)
    if subset:
        use_df = use_df[use_df["id"].astype(str).isin(subset)].reset_index(drop=True)
        if use_df.empty:
            st.error("None of the pasted IDs matched the CSV."); st.stop()

    with st.spinner("Compiling expressions safely…"):
        compiled, usage, clog = compile_filters(use_df)
        st.session_state.compiled_funcs = compiled
        st.session_state.var_usage = usage
        st.session_state.compile_log = clog

    ok_cnt = sum(1 for _,ok,_ in st.session_state.compile_log if ok)
    fail_cnt = len(st.session_state.compile_log) - ok_cnt
    st.success(f"Compiled {ok_cnt} expressions; {fail_cnt} skipped.")

    if fail_cnt:
        with st.expander("Compile report (skipped filters)"):
            fail_rows = [(fid,msg) for (fid,ok,msg) in st.session_state.compile_log if not ok]
            fr = pd.DataFrame(fail_rows, columns=["id","reason"])
            st.dataframe(fr, hide_index=True, use_container_width=True)

    with st.spinner("Scoring filters on history + applying to pool…"):
        res, fw, fp = compute_scores(
            st.session_state.compiled_funcs,
            use_df,
            st.session_state.pool_combos,
            st.session_state.history_winners,
            st.session_state.seed_str,
            alpha=st.session_state.alpha_wpp,
            gamma=st.session_state.gamma_decay
        )
        st.session_state.result_df = res
        st.session_state.fired_on_winners = fw
        st.session_state.fired_on_pool = fp

# Results
st.markdown("### Per-filter metrics")
if st.session_state.result_df is None or st.session_state.result_df.empty:
    st.info("Click RUN after providing inputs.")
else:
    df = st.session_state.result_df.copy()
    df["keep%"] = (df["keep_rate"].astype(float)*100).round(2)
    df["elim%"] = (df["elim_rate"].astype(float)*100).round(2)
    df["WPP"]   = df["WPP"].astype(float).round(6)
    st.dataframe(df[["id","name","keep%","elim%","WPP","expression"]], hide_index=True, use_container_width=True)

    ch = alt.Chart(df.assign(keep_pct=df["keep_rate"]*100, elim_pct=df["elim_rate"]*100)).mark_circle(size=80).encode(
        x=alt.X("elim_pct:Q", title="Elimination on Pool (%)"),
        y=alt.Y("keep_pct:Q", title="Winner Survival (Keep %)"),
        tooltip=["id","name","keep_pct","elim_pct","WPP"]
    ).interactive()
    st.altair_chart(ch.properties(height=360), use_container_width=True)

    # Copy/download
    ids_sorted = "\n".join(df.sort_values("WPP", ascending=False)["id"].astype(str).tolist())
    st.markdown("#### Copy / Download")
    st.code(ids_sorted, language="text")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("Download IDs (.txt)", ids_sorted.encode("utf-8"), file_name="filter_ids_ranked.txt", use_container_width=True)
    with col_b:
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download scored table (.csv)", buf.getvalue().encode("utf-8"), file_name="filters_scored.csv", use_container_width=True)

# Advanced: Greedy bundle
st.markdown("---")
st.markdown("## Build greedy bundle")

if st.session_state.result_df is None or st.session_state.result_df.empty:
    st.info("Run scoring first.")
else:
    a1, a2, a3 = st.columns(3)
    with a1:
        min_survival = st.slider("Min winner survival", 0.50, 0.99, 0.72, 0.01)
    with a2:
        target_survivors = st.number_input("Target survivors", min_value=0, value=50, step=5)
    with a3:
        red_pen = st.slider("Redundancy penalty", 0.0, 1.0, 0.20, 0.05)

    col_g1, col_g2, col_g3 = st.columns([1,1,1])
    go_bundle = col_g1.button("Build greedy bundle", type="primary", use_container_width=True)
    probe = col_g2.button("Feasibility probe (min survival = 0)", use_container_width=True)
    push50 = col_g3.button("Push-to-50 (relax survival in tiny steps)", use_container_width=True)

    winners = st.session_state.history_winners
    Nw = len(winners)
    w_weights = time_weights(Nw, st.session_state.gamma_decay) if Nw>0 else np.array([])

    def _run_bundle(ms: float):
        return build_bundle(
            st.session_state.result_df,
            st.session_state.fired_on_winners,
            st.session_state.fired_on_pool,
            Nw,
            w_weights,
            min_survival=ms,
            target_survivors=int(target_survivors),
            redundancy_penalty=float(red_pen)
        )

    if probe:
        with st.spinner("Probing best-possible thinning…"):
            b = _run_bundle(0.0)
            st.session_state.bundle_result = b
    elif push50:
        with st.spinner("Exploring minimal survival drop to reach ≤ target…"):
            found = None
            for ms in np.linspace(float(min_survival), 0.55, num=9):  # try a few gentle steps
                b = _run_bundle(ms)
                if b["survivors"] <= int(target_survivors):
                    found = (ms, b); break
            st.session_state.bundle_result = found[1] if found else _run_bundle(float(min_survival))
            if found:
                st.success(f"Reached target survivors with min survival ≈ {found[0]:.2f}")
            else:
                st.warning("Could not reach target with the tested survival steps.")
    elif go_bundle:
        with st.spinner("Building bundle…"):
            st.session_state.bundle_result = _run_bundle(float(min_survival))

    if st.session_state.bundle_result:
        b = st.session_state.bundle_result
        st.markdown("### Selected bundle")
        st.write(f"**Filters chosen ({len(b['selected'])})**: {', '.join(b['selected']) or '—'}")
        st.write(f"**Projected winner survival**: {b['survival']*100:.2f}%")
        st.write(f"**Projected survivors (pool)**: {b['survivors']:,}")

        if b["history"]:
            st.dataframe(pd.DataFrame(b["history"]), hide_index=True, use_container_width=True)

        # Survivors for download
        union = set()
        for fid in b["selected"]:
            union |= st.session_state.fired_on_pool.get(fid, set())
        pool = st.session_state.pool_combos
        survivors_idx = sorted(set(range(len(pool))) - union)
        survivors_list = [pool[i] for i in survivors_idx]
        blob = ",".join(survivors_list)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download bundle IDs (.txt)",
                               "\n".join(b["selected"]).encode("utf-8"),
                               file_name="bundle_ids.txt", use_container_width=True)
        with d2:
            st.download_button(f"Download survivors ({len(survivors_list)}).txt",
                               blob.encode("utf-8"),
                               file_name="projected_survivors.txt", use_container_width=True)

        st.markdown("#### Survivors (preview)")
        st.code(", ".join(survivors_list[:2000]), language="text")
