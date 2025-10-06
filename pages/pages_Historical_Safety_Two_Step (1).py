
import re
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime

st.set_page_config(page_title="Historical Safety — Two-Step", layout="wide")

# ----------------- Helpers -----------------
def digits_from_str(s: str):
    s = ''.join(ch for ch in str(s) if ch.isdigit())
    return [int(ch) for ch in s]

def sum_of_digits(digs): return sum(digs)
def parity_counts(digs):
    ev = sum(1 for d in digs if d % 2 == 0); return ev, len(digs) - ev
def high_low_counts(digs):
    lo = sum(1 for d in digs if d in {0,1,2,3,4}); hi = len(digs) - lo; return hi, lo
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
def vtrac_groups(digs): return [V_TRAC_GROUPS[d] for d in digs]
def spread(digs): return max(digs) - min(digs) if digs else 0
def sum_category(total: int) -> str:
    if total <= 15: return 'Very Low'
    if total <= 24: return 'Low'
    if total <= 33: return 'Mid'
    return 'High'

def seed_profile(digs):
    return {
        "sum": sum_of_digits(digs),
        "sum_category": sum_category(sum_of_digits(digs)),
        "even": parity_counts(digs)[0],
        "odd": parity_counts(digs)[1],
        "high": high_low_counts(digs)[0],
        "low": high_low_counts(digs)[1],
        "vtrac": vtrac_groups(digs),
        "spread": spread(digs),
    }

def similarity_score(p_now: dict, p_hist: dict, weights=None):
    if weights is None:
        weights = {"sum":1.0,"sum_category":1.5,"even":0.6,"odd":0.6,"high":0.8,"low":0.8,"spread":0.8,"vtrac":1.2}
    s = 0.0
    s += weights["sum"]*(1 - min(abs(p_now["sum"]-p_hist["sum"]) / 45.0, 1))
    s += weights["spread"]*(1 - min(abs(p_now["spread"]-p_hist["spread"]) / 9.0, 1))
    for k,cap in [("even",5.0),("odd",5.0),("high",5.0),("low",5.0)]:
        s += weights[k]*(1 - min(abs(p_now[k]-p_hist[k]) / cap, 1))
    s += weights["sum_category"]*(1.0 if p_now["sum_category"]==p_hist["sum_category"] else 0.0)
    now_v, hist_v = Counter(p_now["vtrac"]), Counter(p_hist["vtrac"])
    overlap = sum(min(now_v[k], hist_v[k]) for k in now_v.keys() & hist_v.keys())
    s += weights["vtrac"]*(overlap/5.0)
    return s

SAFE_BUILTINS = {
    'len': len, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
    'any': any, 'all': all, 'sorted': sorted, 'set': set, 'tuple': tuple, 'list': list,
    'range': range, 'round': round
}

def build_combo_context(combo_digits, seed_digits):
    d0,d1,d2,d3,d4 = (combo_digits + [None]*5)[:5]
    even_count = parity_counts(combo_digits)[0]
    odd_count  = parity_counts(combo_digits)[1]
    high_count = high_low_counts(combo_digits)[0]
    low_count  = high_low_counts(combo_digits)[1]
    return {
        "combo_digits": combo_digits, "seed_digits": seed_digits,
        "d0": d0, "d1": d1, "d2": d2, "d3": d3, "d4": d4,
        "combo_sum": sum_of_digits(combo_digits),
        "combo_spread": spread(combo_digits),
        "even_count": even_count, "odd_count": odd_count,
        "high_count": high_count, "low_count": low_count,
        "vtrac": vtrac_groups(combo_digits),
        # common aliases
        "digits": combo_digits, "draw_digits": combo_digits, "result": combo_digits,
        "winner": combo_digits, "win_digits": combo_digits, "current_digits": combo_digits,
        "prev_digits": seed_digits, "seed": seed_digits
    }

def eval_strict_bool(expr_text, combo_digits, seed_digits):
    try:
        compiled = compile(expr_text, "<filter_expr>", "eval")
        val = eval(compiled, {"__builtins__": {}}, {**build_combo_context(combo_digits, seed_digits), **SAFE_BUILTINS})
        if isinstance(val, bool):
            return True, val, None
        return False, False, "non_boolean"
    except Exception:
        return False, False, "exception"

# ----------------- UI -----------------
st.title("Historical Safety — Two-Step Pipeline")

# Step 1
st.subheader("Step 1 · Find similar seeds")
col1, col2 = st.columns(2)
with col1:
    hist_file = st.file_uploader("Upload history CSV/TXT", type=["csv","txt"], key="p_hist")
    reverse_hint = st.checkbox("History is reverse chronological (newest first)", value=True, key="p_rev")
with col2:
    seed_input = st.text_input("Current seed (5 digits)", "27500", key="p_seed")

seed_now = digits_from_str(seed_input)
if len(seed_now) != 5:
    st.error("Seed must be 5 digits.")
else:
    with st.expander("Similarity settings", expanded=False):
        w_sum = st.slider("Weight: Sum", 0.0, 3.0, 1.0, 0.1, key="p_wsum")
        w_sumcat = st.slider("Weight: Sum Category", 0.0, 3.0, 1.5, 0.1, key="p_wsumcat")
        w_even = st.slider("Weight: Even Count", 0.0, 3.0, 0.6, 0.1, key="p_weven")
        w_odd  = st.slider("Weight: Odd Count", 0.0, 3.0, 0.6, 0.1, key="p_wodd")
        w_high = st.slider("Weight: High Count", 0.0, 3.0, 0.8, 0.1, key="p_whigh")
        w_low  = st.slider("Weight: Low Count", 0.0, 3.0, 0.8, 0.1, key="p_wlow")
        w_spread = st.slider("Weight: Spread", 0.0, 3.0, 0.8, 0.1, key="p_wspread")
        w_vtrac = st.slider("Weight: V-Trac Overlap", 0.0, 3.0, 1.2, 0.1, key="p_wvtrac")
        weights = {"sum":w_sum,"sum_category":w_sumcat,"even":w_even,"odd":w_odd,"high":w_high,"low":w_low,"spread":w_spread,"vtrac":w_vtrac}
        K  = st.slider("Top-K neighbors", 10, 500, 200, 5, key="p_k")
        thr = st.slider("Minimum similarity", 0.0, 8.0, 3.0, 0.1, key="p_thr")

    if st.button("Compute similar seeds", key="p_step1"):
        if not hist_file:
            st.error("Upload a history file first.")
        else:
            try:
                if hist_file.name.lower().endswith(".txt"):
                    raw = hist_file.read().decode("utf-8", errors="ignore").strip().splitlines()
                    rows = []
                    for line in raw:
                        parts = [p.strip() for p in (line.split(",") if "," in line else line.split())]
                        if parts: rows.append(parts)
                    dfh = pd.DataFrame(rows)
                else:
                    dfh = pd.read_csv(hist_file)
            except Exception as e:
                st.error(f"History read error: {e}")
                dfh = None

            if dfh is not None:
                lowmap = {c.lower(): c for c in dfh.columns}
                seed_col = next((lowmap[k] for k in ["seed","prev","previous","prev_seed","seed_value"] if k in lowmap), None)
                win_col  = next((lowmap[k] for k in ["winner","result","current","draw","win_value","result_value"] if k in lowmap), None)
                if seed_col is None and win_col is None:
                    if len(dfh.columns) == 1:
                        win_col = list(dfh.columns)[0]; dfh.rename(columns={win_col:"Result"}, inplace=True); win_col = "Result"
                    else:
                        win_col = lowmap.get("result") or lowmap.get("draw")
                if win_col is None:
                    st.error("History must include a results column (e.g., 'Result').")
                else:
                    if seed_col is None:
                        series = dfh[win_col].astype(str).tolist()
                        if len(series) < 2:
                            st.error("History needs at least 2 rows.")
                        else:
                            if reverse_hint:
                                seeds = series[1:]; wins = series[:-1]
                            else:
                                seeds = series[:-1]; wins = series[1:]
                            dfh = pd.DataFrame({"seed":seeds,"winner":wins})
                    else:
                        dfh = dfh[[seed_col, win_col]].rename(columns={seed_col:"seed", win_col:"winner"})

                    dfh["seed_digits"] = dfh["seed"].apply(digits_from_str)
                    dfh["winner_digits"] = dfh["winner"].apply(digits_from_str)
                    dfh = dfh[dfh["seed_digits"].apply(len)==5]
                    dfh = dfh[dfh["winner_digits"].apply(len)==5].reset_index(drop=True)

                    dfh["p_sum"] = dfh["seed_digits"].apply(sum_of_digits)
                    dfh["p_sumcat"] = dfh["p_sum"].apply(sum_category)
                    dfh["p_even"] = dfh["seed_digits"].apply(lambda d: parity_counts(d)[0])
                    dfh["p_odd"]  = dfh["seed_digits"].apply(lambda d: parity_counts(d)[1])
                    dfh["p_high"] = dfh["seed_digits"].apply(lambda d: high_low_counts(d)[0])
                    dfh["p_low"]  = dfh["seed_digits"].apply(lambda d: high_low_counts(d)[1])
                    dfh["p_vtr"]  = dfh["seed_digits"].apply(vtrac_groups)
                    dfh["p_spr"]  = dfh["seed_digits"].apply(spread)

                    prof_now = {"sum":sum_of_digits(seed_now),"sum_category":sum_category(sum_of_digits(seed_now)),
                                "even":parity_counts(seed_now)[0],"odd":parity_counts(seed_now)[1],
                                "high":high_low_counts(seed_now)[0],"low":high_low_counts(seed_now)[1],
                                "vtrac":vtrac_groups(seed_now),"spread":spread(seed_now)}

                    dfh["similarity"] = [
                        similarity_score(prof_now, {"sum":r.p_sum,"sum_category":r.p_sumcat,"even":r.p_even,"odd":r.p_odd,
                                                    "high":r.p_high,"low":r.p_low,"vtrac":r.p_vtr,"spread":r.p_spr}, weights)
                        for _, r in dfh.iterrows()
                    ]

                    nbrs = dfh[dfh["similarity"]>=thr].sort_values("similarity", ascending=False).head(K)
                    if nbrs.empty:
                        st.warning("No neighbors — relax similarity or increase Top-K.")
                    else:
                        st.success(f"Locked {len(nbrs)} similar seeds.")
                        st.session_state["neighbors_locked"] = nbrs[["seed","winner","seed_digits","winner_digits","similarity"]].copy()
                        st.dataframe(st.session_state["neighbors_locked"].head(50), hide_index=True, use_container_width=True)
                        wins_df = st.session_state["neighbors_locked"][["winner"]].rename(columns={"winner":"winner_draw"}).copy()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button("Download winners from similar seeds (CSV)",
                                           wins_df.to_csv(index=False).encode("utf-8"),
                                           file_name=f"similar_seed_winners_{ts}.csv", mime="text/csv")

st.subheader("Step 2 · Evaluate filters on locked winners")
filt_file = st.file_uploader("Upload Filters CSV (id + expression required)", type=["csv"], key="p_filters")
use_enabled = st.checkbox("Use only enabled==True (if present)", value=False, key="p_enabled")
limit_ids = st.checkbox("Limit to pasted Filter IDs", value=False, key="p_limit_ids")
ids_text = st.text_area("Paste IDs (comma/space/newline separated)", "", height=100, key="p_ids_text")

if st.button("Evaluate", key="p_step2"):
    if "neighbors_locked" not in st.session_state:
        st.error("Run Step 1 to lock neighbors first.")
    elif not filt_file:
        st.error("Upload a filters CSV.")
    else:
        nbrs = st.session_state["neighbors_locked"].copy()
        try:
            dff = pd.read_csv(filt_file)
        except Exception as e:
            st.error(f"Filters read error: {e}")
            dff = None
        if dff is not None:
            low = {c.lower(): c for c in dff.columns}
            id_col = next((low[k] for k in ["id","filter_id","fid"] if k in low), None)
            expr_col = next((low[k] for k in ["expression","expr","rule"] if k in low), None)
            name_col = next((low[k] for k in ["name","layman_explanation","layman","description","title"] if k in low), None) or id_col
            if id_col is None or expr_col is None:
                st.error("Filters CSV must include 'id' and 'expression'.")
            else:
                dff = dff[[id_col,name_col,expr_col] + [c for c in dff.columns if c not in (id_col,name_col,expr_col)]].rename(columns={id_col:"id",name_col:"name",expr_col:"expression"})
                if use_enabled and ("enabled" in dff.columns):
                    dff = dff[dff["enabled"].astype(str).str.lower().isin(["true","1","yes","y","t"])].copy()
                if limit_ids and ids_text.strip():
                    tokens = re.split(r"[,\s]+", ids_text.strip())
                    include_ids = [t.strip() for t in tokens if t.strip()]
                    dff = dff[dff["id"].astype(str).isin(include_ids)].copy()

                results = []
                for _, row in dff.iterrows():
                    fid, fname, expr = str(row["id"]), str(row["name"]), str(row["expression"]).strip()
                    if not expr: continue
                    valid = err_nonbool = err_exc = elim = 0
                    for _, n in nbrs.iterrows():
                        seed_d = n["seed_digits"]; combo_d = n["winner_digits"]
                        is_bool, val, err = eval_strict_bool(expr, combo_d, seed_d)
                        if err is None:
                            valid += 1
                            if val is True: elim += 1
                        elif err == "non_boolean":
                            err_nonbool += 1
                        else:
                            err_exc += 1
                    elim_rate = (elim/valid) if valid else 0.0
                    safety = 1.0 - elim_rate if valid else 0.0
                    results.append({
                        "id": fid, "name": fname,
                        "valid_cases": valid,
                        "error_cases_nonbool": err_nonbool,
                        "error_cases_exceptions": err_exc,
                        "historical_safety_%": round(safety*100.0, 2)
                    })
                if not results:
                    st.error("No evaluable filters found.")
                else:
                    out = pd.DataFrame(results).sort_values(["historical_safety_%","valid_cases"], ascending=[False,False]).reset_index(drop=True)
                    st.success(f"Evaluated {len(out)} filters against {len(nbrs)} winners.")
                    st.dataframe(out, hide_index=True, use_container_width=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button("Download recommendations CSV",
                                       out.to_csv(index=False).encode("utf-8"),
                                       file_name=f"historical_recommendations_{ts}.csv", mime="text/csv")
