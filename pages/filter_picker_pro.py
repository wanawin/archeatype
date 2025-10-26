import io, math, random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Safe, tester-only context
# =========================
DIGITS = [str(i) for i in range(10)]

def to_digits(x) -> List[str]:
    if isinstance(x, str):
        return list(x.strip().replace(" ", ""))
    return [str(v) for v in x]

def build_ctx(combo: str, seed: str) -> Dict[str, Any]:
    combo_digits = to_digits(combo)
    seed_digits  = to_digits(seed)
    return dict(
        combo_digits=combo_digits,
        seed_digits=seed_digits,
        combo_sum=sum(int(d) for d in combo_digits),
        seed_sum=sum(int(d) for d in seed_digits),
        any=any, all=all, sum=sum, len=len, max=max, min=min, set=set, range=range, ord=ord
    )

def safe_eval(expr: str, ctx: Dict[str, Any]) -> bool:
    return bool(eval(expr, {}, ctx))

@dataclass
class FilterRow:
    fid: str
    name: str
    expr: str
    applicable_if: Optional[str] = None

# ===============
# Wilson interval
# ===============
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z*z/(2*n)) / denom
    radius = z * math.sqrt( (p*(1-p) + z*z/(4*n)) / n ) / denom
    lo, hi = max(0.0, center - radius), min(1.0, center + radius)
    return lo, hi

# ==========
# UI set-up
# ==========
st.set_page_config(page_title="Filter Picker Pro (Tester-only)", layout="wide")
st.title("Filter Picker Pro — Tester-only variables")
st.caption("Ranks filters by keep % (with CIs) + thinning, controls redundancy, and builds a high-survival bundle. No LL variables used.")

with st.sidebar:
    st.subheader("1) Upload data")
    filters_file = st.file_uploader("Filters CSV (tester-friendly)", type=["csv"])
    st.caption("Columns: id or name, expression, optional applicable_if.")
    hist_file = st.file_uploader("Historical winners (CSV/TXT)", type=["csv","txt"])
    hist_col  = st.text_input("Column name for winners (blank if single col)", "")
    pool_file = st.file_uploader("Current pool (CSV/TXT)", type=["csv","txt"])
    pool_col  = st.text_input("Column name for pool (blank if single col)", "")

    st.subheader("2) Current seed")
    seed_text = st.text_input("Seed (5 digits)", "88001")

    st.subheader("3) Scoring knobs")
    bt_n = st.number_input("Backtest sample size", 50, 5000, 500, 50,
                           help="Winners randomly sampled from your history for keep-rate.")
    alpha = st.slider("α in WPP = K × E^α", 0.1, 2.0, 1.0, 0.1)
    red_penalty = st.slider("Redundancy penalty (Jaccard weight)", 0.0, 1.0, 0.3, 0.05,
                            help="Higher penalizes filters that eliminate the same winners as already-picked filters.")

    st.subheader("4) Bundle builder")
    min_keep_lo = st.slider("Min lower-CI Keep % (bundle)", 50, 100, 80, 1)
    max_filters = st.number_input("Max filters to pick", 1, 200, 20)

# ========
# Loaders
# ========
def load_list_like(file, colname: str) -> List[str]:
    if file is None:
        return []
    raw = file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
        if colname and colname in df.columns:
            vals = df[colname].astype(str).tolist()
        else:
            vals = df.iloc[:,0].astype(str).tolist()
    except Exception:
        vals = io.BytesIO(raw).read().decode("utf-8").strip().splitlines()
    vals = [v.strip().replace(",", "").replace(" ", "") for v in vals if v.strip()]
    return vals

def load_filters(file) -> List[FilterRow]:
    if file is None:
        return []
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    fid_col = "id" if "id" in df.columns else ("name" if "name" in df.columns else None)
    if fid_col is None or "expression" not in df.columns:
        st.error("Filters CSV must include 'id' or 'name' and 'expression'.")
        return []
    out = []
    for _, r in df.iterrows():
        fid = str(r.get(fid_col,"")).strip()
        name = str(r.get("name", fid)).strip() or fid
        expr = str(r.get("expression","")).strip()
        app  = str(r.get("applicable_if","")).strip() if "applicable_if" in df.columns else None
        if fid and expr:
            out.append(FilterRow(fid=fid, name=name, expr=expr, applicable_if=app or None))
    return out

filters = load_filters(filters_file)
history = load_list_like(hist_file, hist_col)
pool    = load_list_like(pool_file, pool_col)

cA,cB,cC = st.columns(3)
cA.metric("Filters", len(filters))
cB.metric("History", len(history))
cC.metric("Pool", len(pool))

if not filters or not pool:
    st.info("Upload filters and a pool to continue.")
    st.stop()

# Sample history for speed
hist_sample = history if len(history) <= bt_n else random.sample(history, bt_n)

# ==================================================
# Evaluate each filter: K (with CI), E, masks & WPP
# ==================================================
@st.cache_data(show_spinner=False)
def score_filters(filters: List[FilterRow], hist: List[str], pool: List[str], seed_text: str) -> pd.DataFrame:
    rows = []
    # pre-build contexts for speed
    pool_ctxs = [build_ctx(c, seed_text) for c in pool]
    hist_ctxs = [build_ctx(w, seed_text) for w in hist]

    for fr in filters:
        # --- pool elimination mask (current pool)
        elim_pool = []
        for ctx in pool_ctxs:
            try:
                if fr.applicable_if and not safe_eval(fr.applicable_if, ctx):
                    elim_pool.append(False)
                else:
                    elim_pool.append(bool(safe_eval(fr.expr, ctx)))
            except Exception:
                elim_pool.append(False)
        E = sum(elim_pool) / max(1, len(pool))

        # --- winner keep mask on history
        kept = []
        elim_hist = []  # for Jaccard redundancy (who got eliminated)
        for ctx in hist_ctxs:
            try:
                if fr.applicable_if and not safe_eval(fr.applicable_if, ctx):
                    kept.append(True)      # not applicable → kept
                    elim_hist.append(False)
                else:
                    fired = bool(safe_eval(fr.expr, ctx))
                    kept.append(not fired)
                    elim_hist.append(fired)
            except Exception:
                kept.append(True)
                elim_hist.append(False)
        k = sum(kept)
        n = len(kept)
        K = k / max(1, n)
        K_lo, K_hi = wilson_ci(k, n)

        rows.append(dict(
            id=fr.fid, name=fr.name, expression=fr.expr, applicable_if=fr.applicable_if or "",
            keep_rate=K, keep_lo=K_lo, keep_hi=K_hi,
            elim_rate=E,
            wpp=K * (E ** 1.0),   # temporary α=1, will rescore below
            elim_hist_mask=elim_hist,  # list[bool]
        ))
    df = pd.DataFrame(rows)
    return df

with st.spinner("Scoring filters (CIs, masks, thinning)…"):
    df = score_filters(filters, hist_sample, pool, seed_text)

# Recompute WPP with chosen α
df["wpp"] = df["keep_rate"] * (df["elim_rate"] ** alpha)

st.subheader("Per-filter metrics")
st.dataframe(
    df.sort_values("wpp", ascending=False)[["id","keep_rate","keep_lo","keep_hi","elim_rate","wpp"]],
    use_container_width=True, height=420
)

# ----------------------------
# Pareto plot (E vs K)
# ----------------------------
st.subheader("Pareto — thin vs keep")
chart = (
    alt.Chart(df)
    .mark_circle(size=80, opacity=0.7)
    .encode(
        x=alt.X("elim_rate:Q", title="Elimination rate (E) on pool"),
        y=alt.Y("keep_rate:Q", title="Winner keep rate (K)"),
        color=alt.value("#4F46E5"),
        tooltip=["id","name","keep_rate","elim_rate","wpp"]
    )
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

# =========================================
# Greedy bundle with redundancy penalty
# =========================================
def jaccard(a: List[bool], b: List[bool]) -> float:
    # Treat True as "eliminated this winner"
    inter = sum(1 for x,y in zip(a,b) if x and y)
    union = sum(1 for x,y in zip(a,b) if x or y)
    return 0.0 if union == 0 else inter/union

def compose_keep_lo(current_lo: float, added_lo: float) -> float:
    # independence lower-bound approximation
    return current_lo * added_lo

def bundle_select(df: pd.DataFrame, min_keep_lo: float, max_k: int, alpha: float, red_penalty: float):
    remaining = df.copy()
    chosen = []
    agg_keep_lo = 1.0
    agg_elim = 0.0  # on pool

    while len(chosen) < max_k and not remaining.empty:
        best_idx = None
        best_gain = -1e9

        for idx, r in remaining.iterrows():
            # redundancy penalty: max Jaccard with chosen
            if chosen:
                jac = max(jaccard(r.elim_hist_mask, c.elim_hist_mask) for c in chosen)
            else:
                jac = 0.0

            # candidate new bundle metrics
            next_keep_lo = compose_keep_lo(agg_keep_lo, r.keep_lo)
            if next_keep_lo < min_keep_lo/100:
                continue

            new_elim = 1 - (1-agg_elim)*(1-r.elim_rate)
            marginal_elim = new_elim - agg_elim

            # value: WPP-like marginal – redundancy penalty
            gain = (r.keep_rate * (r.elim_rate ** alpha)) + marginal_elim - red_penalty * jac

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break

        pick = remaining.loc[best_idx]
        chosen.append(pick)
        agg_keep_lo = compose_keep_lo(agg_keep_lo, pick.keep_lo)
        agg_elim = 1 - (1-agg_elim)*(1-pick.elim_rate)
        remaining = remaining.drop(index=best_idx)

    return pd.DataFrame(chosen), agg_keep_lo, agg_elim

with st.spinner("Building bundle…"):
    chosen, bundle_keep_lo, bundle_elim = bundle_select(
        df, min_keep_lo=min_keep_lo, max_k=max_filters, alpha=alpha, red_penalty=red_penalty
    )

st.subheader("Recommended bundle")
if chosen.empty:
    st.warning("No bundle met the lower-CI keep target. Try reducing the target, α, or redundancy penalty.")
else:
    ids = chosen["id"].tolist()
    c1,c2,c3 = st.columns(3)
    c1.metric("Picked filters", len(ids))
    c2.metric("Bundle lower-CI keep", f"{bundle_keep_lo*100:.1f}%")
    c3.metric("Projected pool reduction", f"{bundle_elim*100:.1f}%")
    st.code(", ".join(ids), language="text")
    st.dataframe(
        chosen[["id","keep_rate","keep_lo","keep_hi","elim_rate","wpp"]],
        use_container_width=True, height=360
    )

st.divider()
st.markdown("""
**What’s inside**
- All metrics use **tester-only** variables (`combo_digits`, `seed_digits`, sums, etc.).
- **K** = winner keep-rate on your sampled history; we show a **Wilson 95% CI** for safety.
- **E** = elimination-rate on your **current pool**.
- **WPP = K × E^α** balances safety vs thinning (α slider).
- **Bundle builder**: greedy, uses **lower-CI K** and penalizes **redundancy** via Jaccard on eliminated winners.
- **Pareto chart** lets you visually pick thin vs keep.
""")
