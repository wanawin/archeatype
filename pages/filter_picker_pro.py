# filter_picker_hybrid.py
# -----------------------------------------------------------
# Paste applicable filter IDs + paste pool (supports continuous digits),
# upload master filter CSV and historical winners.
# Tester-only expressions (no LL variables). Full app, no patches.
# -----------------------------------------------------------

import io, math, random, re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Minimal, tester-only context
# ----------------------------
DIGITS = [str(i) for i in range(10)]

def to_digits(x) -> List[str]:
    if isinstance(x, str):
        return list(re.sub(r"\s+", "", x))
    return [str(v) for v in x]

def build_ctx(combo: str, seed: str) -> Dict[str, Any]:
    combo_digits = to_digits(combo)
    seed_digits  = to_digits(seed)
    return dict(
        combo_digits=combo_digits,
        seed_digits=seed_digits,
        combo_sum=sum(int(d) for d in combo_digits),
        seed_sum=sum(int(d) for d in seed_digits),
        # safe builtins allowed in tester expressions
        any=any, all=all, sum=sum, len=len, max=max, min=min, set=set, range=range, ord=ord
    )

def safe_eval(expr: str, ctx: Dict[str, Any]) -> bool:
    # NOTE: expressions are assumed tester-safe (no LL vars).
    return bool(eval(expr, {}, ctx))

@dataclass
class FilterRow:
    fid: str
    name: str
    expr: str
    applicable_if: Optional[str] = None
    enabled: Optional[bool] = None

# ----------------------------
# Wilson 95% CI for proportion
# ----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    radius = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    lo, hi = max(0.0, center - radius), min(1.0, center + radius)
    return lo, hi

# ----------------------------
# Robust parsers (incl. continuous)
# ----------------------------
def _chunk_continuous(s: str, size: int = 5) -> List[str]:
    s = re.sub(r"\D", "", s)  # keep digits only
    return [s[i:i+size] for i in range(0, len(s) - len(s)%size, size) if len(s[i:i+size]) == size]

def parse_ids(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    # split on newline, comma, semicolon or whitespace
    parts = re.split(r"[,\s;]+", text.strip())
    return [p for p in parts if p]

def parse_pool_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    raw = text.strip()
    # If it contains any non-digit separators, split first
    if re.search(r"[^\d]", raw):
        # pull out 5-digit tokens anywhere
        tokens = re.findall(r"\d{5}", raw.replace(",", " ").replace(";", " "))
        if tokens:
            return tokens
        # otherwise, strip non-digits and chunk
        return _chunk_continuous(raw)
    # pure digits → chunk into 5s
    return _chunk_continuous(raw)

def load_list_like(file, colname: str) -> List[str]:
    """Load winners (or pool) from CSV/TXT. Supports:
       - single column CSV with 5-digit strings
       - single cell with continuous digits
       - TXT with lines or a continuous digit block
    """
    if file is None:
        return []
    raw = file.read()
    # try CSV first
    try:
        df = pd.read_csv(io.BytesIO(raw))
        if colname and colname in df.columns:
            vals = df[colname].astype(str).tolist()
        else:
            # if first column looks like one big continuous digit block, handle it
            series = df.iloc[:,0].astype(str)
            if len(series) == 1 and re.fullmatch(r"\D*\d+\D*", series.iloc[0] or ""):
                return _chunk_continuous(series.iloc[0])
            vals = series.tolist()
        # extract any 5-digit tokens per cell
        out: List[str] = []
        for v in vals:
            toks = re.findall(r"\d{5}", str(v))
            if toks:
                out.extend(toks)
        if out:
            return out
        # fallback: treat concatenation as continuous
        return _chunk_continuous("".join(vals))
    except Exception:
        # not a CSV → plain text
        txt = io.BytesIO(raw).read().decode("utf-8", errors="ignore").strip()
        # prefer 5-digit tokens if present
        tokens = re.findall(r"\d{5}", txt)
        if tokens:
            return tokens
        return _chunk_continuous(txt)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Filter Picker (Hybrid I/O)", layout="wide")
st.title("Filter Picker — Hybrid I/O (paste pool & IDs, upload filters & history)")
st.caption("Tester-only evaluation (no LL variables). Builds a bundle that thins your pool while preserving the winner on uploaded history.")

with st.sidebar:
    st.subheader("1) Upload master filter CSV")
    filters_file = st.file_uploader("Master filters CSV", type=["csv"])
    st.caption("Accepted schemas:\n- Wide: id,name,enabled,applicable_if,expression,...\n- Slim: name,description,expression (id derived from name)")

    st.subheader("2) Upload historical winners")
    hist_file = st.file_uploader("Historical winners (CSV/TXT)", type=["csv","txt"])
    hist_col  = st.text_input("Column name for winners (blank if single column)", "")

    st.subheader("3) Current seed")
    seed_input = st.text_input("Seed input (either 5 digits or the 13-draw line; we take the last 5 digits)",
                               "88001,87055,04510,43880,99472,21693,96549,44281,78170,83337,77692,75003,61795")
    seed_5 = re.sub(r"\D", "", seed_input)[-5:]

    st.subheader("4) Scoring & selection knobs")
    bt_n = st.number_input("Backtest sample size", 50, 5000, 500, 50)
    alpha = st.slider("α in WPP = K × E^α", 0.1, 2.0, 1.0, 0.1)
    min_keep_lo = st.slider("Min lower-CI Keep % (bundle)", 50, 100, 80, 1)
    max_filters = st.number_input("Max filters to pick", 1, 200, 20)
    red_penalty = st.slider("Redundancy penalty weight (Jaccard on eliminated winners)", 0.0, 1.0, 0.3, 0.05)

st.subheader("Paste applicable filter IDs (subset you want to consider)")
ids_text = st.text_area("IDs (comma/space/newline). Leave blank to consider all enabled filters.",
                        height=100, placeholder="LL002f, LL003b, LL006i, ...")

st.subheader("Paste current pool — supports continuous digits")
pool_text = st.text_area("Examples:\n01234\n98765\n—or—\n0123498765...\n—or—\n01234, 98765, ...",
                         height=200)

# ----------------------------
# Loaders for filters/history
# ----------------------------
def load_filters(file) -> List[FilterRow]:
    if file is None:
        return []
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    fid_col = "id" if "id" in df.columns else ("name" if "name" in df.columns else None)
    if fid_col is None or "expression" not in df.columns:
        st.error("Master CSV must include 'id' or 'name' and 'expression'.")
        return []

    rows: List[FilterRow] = []
    for _, r in df.iterrows():
        fid = str(r.get(fid_col, "")).strip()
        name = str(r.get("name", fid)).strip() or fid
        expr = str(r.get("expression", "")).strip()
        app  = str(r.get("applicable_if", "")).strip() if "applicable_if" in df.columns else None
        enabled = r.get("enabled", None)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1","true","yes","y")
        if fid and expr:
            rows.append(FilterRow(fid=fid, name=name, expr=expr, applicable_if=(app or None), enabled=enabled))
    return rows

filters_all = load_filters(filters_file)
history = load_list_like(hist_file, hist_col)
pool = parse_pool_text(pool_text)

cA, cB, cC, cD = st.columns(4)
cA.metric("Master filters", len(filters_all))
cB.metric("History rows", len(history))
cC.metric("Parsed pool size", len(pool))
cD.metric("Seed (last 5)", seed_5 or 0)

if not filters_all or not pool or not seed_5 or not history:
    st.info("Upload master filter CSV, upload history, paste pool (continuous supported), and provide a 5-digit seed.")
    st.stop()

# Apply applicable IDs subset if provided + enabled flag
applicable_ids = set(parse_ids(ids_text))
if applicable_ids:
    filters = [f for f in filters_all if f.fid in applicable_ids]
else:
    filters = [f for f in filters_all if (f.enabled is None or f.enabled is True)]

st.success(f"Filters considered: {len(filters)}")

# Sample history for speed
hist_sample = history if len(history) <= bt_n else random.sample(history, bt_n)

# ----------------------------
# Scoring per filter
# ----------------------------
@st.cache_data(show_spinner=False)
def score_filters(filters: List[FilterRow], hist: List[str], pool: List[str], seed_5: str) -> pd.DataFrame:
    rows = []
    pool_ctxs = [build_ctx(c, seed_5) for c in pool]
    hist_ctxs = [build_ctx(w, seed_5) for w in hist]

    for fr in filters:
        # Pool elimination
        elim_pool_mask = []
        for ctx in pool_ctxs:
            try:
                if fr.applicable_if and not safe_eval(fr.applicable_if, ctx):
                    elim_pool_mask.append(False)   # not applied → keep
                else:
                    elim_pool_mask.append(bool(safe_eval(fr.expr, ctx)))
            except Exception:
                elim_pool_mask.append(False)
        E = sum(elim_pool_mask) / max(1, len(pool))

        # History keep (winner survival)
        keep_mask = []
        elim_hist_mask = []
        for ctx in hist_ctxs:
            try:
                if fr.applicable_if and not safe_eval(fr.applicable_if, ctx):
                    keep_mask.append(True)
                    elim_hist_mask.append(False)
                else:
                    fired = bool(safe_eval(fr.expr, ctx))
                    keep_mask.append(not fired)
                    elim_hist_mask.append(fired)
            except Exception:
                keep_mask.append(True)
                elim_hist_mask.append(False)
        k = sum(keep_mask)
        n = len(keep_mask)
        K = k / max(1, n)
        K_lo, K_hi = wilson_ci(k, n)

        rows.append(dict(
            id=fr.fid, name=fr.name, expression=fr.expr, applicable_if=fr.applicable_if or "",
            keep_rate=K, keep_lo=K_lo, keep_hi=K_hi, elim_rate=E,
            elim_hist_mask=elim_hist_mask
        ))
    return pd.DataFrame(rows)

with st.spinner("Scoring filters…"):
    df = score_filters(filters, hist_sample, pool, seed_5)

# WPP with current alpha
df["wpp"] = df["keep_rate"] * (df["elim_rate"] ** alpha)

st.subheader("Per-filter metrics")
st.dataframe(
    df.sort_values("wpp", ascending=False)[["id","keep_rate","keep_lo","keep_hi","elim_rate","wpp"]],
    use_container_width=True, height=420
)

# Pareto: thin vs keep
st.subheader("Pareto — Pool Elimination (E) vs Winner Keep (K)")
chart = (
    alt.Chart(df)
    .mark_circle(size=80, opacity=0.75)
    .encode(
        x=alt.X("elim_rate:Q", title="Elimination rate on pool (E)"),
        y=alt.Y("keep_rate:Q", title="Winner keep rate on history (K)"),
        tooltip=["id","name","keep_rate","elim_rate","wpp"]
    ).interactive()
)
st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Bundle builder (greedy, CI)
# ----------------------------
def jaccard(a: List[bool], b: List[bool]) -> float:
    inter = sum(1 for x,y in zip(a,b) if x and y)
    uni   = sum(1 for x,y in zip(a,b) if x or y)
    return 0.0 if uni == 0 else inter/uni

def compose_keep_lo(cur_lo: float, add_lo: float) -> float:
    # independence lower bound
    return cur_lo * add_lo

def build_bundle(df: pd.DataFrame, min_keep_lo: float, max_k: int, alpha: float, red_penalty: float):
    remaining = df.copy()
    chosen = []
    agg_keep_lo = 1.0
    agg_elim = 0.0

    while len(chosen) < max_k and not remaining.empty:
        best_idx = None
        best_gain = -1e9

        for idx, r in remaining.iterrows():
            next_lo = compose_keep_lo(agg_keep_lo, r.keep_lo)
            if next_lo < (min_keep_lo/100.0):
                continue
            # diminishing returns on elimination
            new_agg_elim = 1 - (1-agg_elim)*(1-r.elim_rate)
            marginal_elim = new_agg_elim - agg_elim
            # redundancy penalty: similarity of eliminated-winners mask
            jac = max((jaccard(r.elim_hist_mask, c.elim_hist_mask) for c in chosen), default=0.0)
            # value = WPP-like + marginal_elim - redundancy
            gain = (r.keep_rate * (r.elim_rate ** alpha)) + marginal_elim - red_penalty * jac
            if gain > best_gain:
                best_gain, best_idx = gain, idx

        if best_idx is None:
            break
        pick = remaining.loc[best_idx]
        chosen.append(pick)
        agg_keep_lo = compose_keep_lo(agg_keep_lo, pick.keep_lo)
        agg_elim = 1 - (1-agg_elim)*(1-pick.elim_rate)
        remaining = remaining.drop(index=best_idx)

    return pd.DataFrame(chosen), agg_keep_lo, agg_elim

with st.spinner("Building bundle…"):
    chosen, bundle_keep_lo, bundle_elim = build_bundle(df, min_keep_lo, max_filters, alpha, red_penalty)

st.subheader("Recommended bundle (IDs)")
if chosen.empty:
    st.warning("No bundle met the lower-CI keep target. Try lowering the target, α, or redundancy penalty.")
else:
    id_list = ", ".join(chosen["id"].tolist())
    c1,c2,c3 = st.columns(3)
    c1.metric("Picked filters", len(chosen))
    c2.metric("Bundle lower-CI Keep", f"{bundle_keep_lo*100:.1f}%")
    c3.metric("Projected pool reduction", f"{bundle_elim*100:.1f}%")
    st.code(id_list, language="text")
    st.markdown("**Chosen details**")
    st.dataframe(chosen[["id","keep_rate","keep_lo","keep_hi","elim_rate","wpp"]], use_container_width=True, height=360)

# Export chosen subset as a CSV (id + expression) for running elsewhere
st.subheader("Export chosen subset (for runner)")
export_df = chosen[["id"]].merge(
    df[["id","name","expression","applicable_if"]],
    on="id", how="left"
).drop_duplicates("id")
st.code(export_df.to_csv(index=False), language="csv")
st.download_button("Download chosen filters CSV",
                   data=export_df.to_csv(index=False).encode("utf-8"),
                   file_name="chosen_filters.csv",
                   mime="text/csv")

st.divider()
st.markdown("""
**Paste formats supported for pool & history**
- One per line (classic)
- Comma/space/semicolon separated
- **Continuous digits:** we auto-chunk into 5s and ignore leftover <5 digits
""")
