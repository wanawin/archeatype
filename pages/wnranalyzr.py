# archetype/pages/wnranalyzr.py
from __future__ import annotations

import io, re, math, hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- UI CONFIG --------------------
st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")
APP_VERSION = "v0.4 (Run button + hot/cold guards)"
st.caption(f"DC5 Hot/Cold/Due Analyzer — {APP_VERSION}")

# -------------------- UTILITIES --------------------
def _hash_params(d: Dict[str, Any]) -> str:
    m = hashlib.md5(); m.update(repr(sorted(d.items())).encode()); return m.hexdigest()

def _clean_combo_to_digits(s: str) -> Optional[List[int]]:
    if s is None: return None
    m = re.search(r"(\d{5})", str(s))
    return [int(ch) for ch in m.group(1)] if m else None

def _ensure_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Guarantee list-like columns exist (prevents KeyError: 'hot', etc.)
    for col in ("hot", "cold", "due", "neutral"):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    return df

def composition_counts(df: pd.DataFrame, tag_col: str, digits_col: str = "digits") -> pd.Series:
    # Robust counting even if column missing or contains non-lists
    if tag_col not in df.columns:
        return pd.Series([0]*len(df), index=df.index)
    def _count(row):
        tags = row.get(tag_col, [])
        tagset = set(tags) if isinstance(tags, (list, tuple, set)) else set()
        digs = row.get(digits_col, [])
        return sum(int(d) in tagset for d in digs)
    return df.apply(_count, axis=1)

def _freq_of_window(rows: List[List[int]]) -> np.ndarray:
    counts = np.zeros(10, dtype=int)
    for digs in rows:
        for d in digs: counts[d] += 1
    return counts

def _rank_hot_cold(freq: np.ndarray, hot_pct: int, cold_pct: int) -> Tuple[List[int], List[int]]:
    order = np.argsort(-freq)
    k_hot  = max(0, min(10, math.ceil(hot_pct  * 10 / 100)))
    k_cold = max(0, min(10, math.ceil(cold_pct * 10 / 100)))
    hot  = list(order[:k_hot])
    cold = list(order[-k_cold:]) if k_cold > 0 else []
    hot_set = set(hot); cold = [d for d in cold if d not in hot_set]   # disjoint
    return hot, cold

def _compute_due(prior_rows: List[List[int]], due_n: int) -> List[int]:
    if due_n <= 0: return []
    recent = prior_rows[-due_n:] if due_n <= len(prior_rows) else prior_rows
    seen = {d for digs in recent for d in digs}
    return [d for d in range(10) if d not in seen]

def _guess_order(df: pd.DataFrame) -> str:
    # Default to Newest-first (common in DC-5 text lists)
    for col in df.columns:
        if "date" in col.lower():
            try:
                s = pd.to_datetime(df[col], errors="raise")
                if s.is_monotonic_increasing:  return "Oldest first"
                if s.is_monotonic_decreasing:  return "Newest first"
            except Exception:
                pass
    return "Newest first"

# -------------------- LOADING ----------------------
@st.cache_data(show_spinner=False)
def _load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    text = file_bytes.decode(errors="ignore")
    if filename.lower().endswith(".csv"):
        df0 = pd.read_csv(io.StringIO(text))
        # try to pick a column with many 5-digit strings
        pick = None
        for col in df0.columns:
            ok = df0[col].astype(str).str.contains(r"\d{5}", regex=True, na=False)
            if ok.mean() > 0.5: pick = col; break
        if pick is None: pick = df0.columns[0]
        df = pd.DataFrame({"raw": df0[pick].astype(str)})
    else:
        rows = [ln.strip() for ln in text.splitlines() if ln.strip()]
        df = pd.DataFrame({"raw": rows})

    df["digits"] = df["raw"].apply(_clean_combo_to_digits)
    df = df[~df["digits"].isna()].copy().reset_index(drop=True)
    return df[["raw", "digits"]]

def _annotate_tags(
    winners_oldest_to_newest: List[List[int]],
    window: int, hot_pct: int, cold_pct: int, due_n: int
):
    n = len(winners_oldest_to_newest)
    rows = []
    for i in range(n):
        prior = winners_oldest_to_newest[max(0, i-window): i] if window > 0 else winners_oldest_to_newest[:i]
        freq  = _freq_of_window(prior) if prior else np.zeros(10, dtype=int)
        hot, cold = _rank_hot_cold(freq, hot_pct, cold_pct)
        due   = _compute_due(prior, due_n)
        tagset = set(hot) | set(cold) | set(due)
        neutral = [d for d in range(10) if d not in tagset]
        rows.append({"digits": winners_oldest_to_newest[i], "hot": hot, "cold": cold, "due": due, "neutral": neutral})

    per = pd.DataFrame(rows)
    _ensure_tag_columns(per)
    # For quick list: use the most recent window (or all)
    tail = winners_oldest_to_newest[-window:] if window > 0 else winners_oldest_to_newest
    freq_quick = _freq_of_window(tail)
    hottest_order = list(np.argsort(-freq_quick))
    return per, hottest_order

@st.cache_data(show_spinner=False)
def _run_analysis(df_loaded: pd.DataFrame, order_choice: str, window: int, hot_pct: int, cold_pct: int, due_n: int):
    if order_choice == "Auto (guess)":
        order_choice = _guess_order(df_loaded)

    if order_choice == "Newest first":
        ordered = df_loaded.iloc[::-1].reset_index(drop=True)
    else:
        ordered = df_loaded.copy()

    winners = ordered["digits"].tolist()
    per, hottest = _annotate_tags(winners, window, hot_pct, cold_pct, due_n)

    # Safe composition counts (no KeyError)
    per["hot_hits"]     = composition_counts(per, "hot")
    per["cold_hits"]    = composition_counts(per, "cold")
    per["due_hits"]     = composition_counts(per, "due")
    per["neutral_hits"] = composition_counts(per, "neutral")

    summary = per[["hot_hits","cold_hits","due_hits","neutral_hits"]].describe().round(3)
    quick = ", ".join(map(str, hottest))

    # Exports
    csv_df = per.copy()
    for col in ("digits","hot","cold","due","neutral"):
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda v: ",".join(map(str, v)) if isinstance(v, (list,tuple)) else "")
    csv_bytes = csv_df.to_csv(index=False).encode()
    txt = f"Hottest → coldest (most-recent window):\n{quick}\n\nComposition summary:\n{summary.to_string()}"
    txt_bytes = txt.encode()

    return {"per": per, "summary": summary, "quick": quick,
            "csv": csv_bytes, "txt": txt_bytes, "order_used": order_choice}

# -------------------- SESSION ----------------------
if "loaded_df" not in st.session_state: st.session_state["loaded_df"] = None
if "results"   not in st.session_state: st.session_state["results"]   = None

# -------------------- CONTROLS ---------------------
with st.sidebar:
    st.header("1) Input")
    up = st.file_uploader("Upload winners (.csv or .txt)", type=["csv","txt"])
    if up is not None:
        try:
            st.session_state["loaded_df"] = _load_file(up.getvalue(), up.name)
            st.success(f"Loaded {len(st.session_state['loaded_df'])} rows from {up.name}")
        except Exception as e:
            st.error(f"Load failed: {e}")

    st.header("2) File order")
    order_choice = st.radio("Row order in file",
        options=["Auto (guess)", "Newest first", "Oldest first"], index=0)

    st.header("3) Run")
    with st.form("run_form"):
        window   = st.number_input("Window for hot/cold ranking (0 = all)", min_value=0, value=10, step=1)
        hot_pct  = st.slider("% digits labeled Hot",  0, 100, 30)
        cold_pct = st.slider("% digits labeled Cold", 0, 100, 30)
        due_n    = st.number_input("Due threshold: not seen in last N draws", min_value=0, value=2, step=1)
        prefix   = st.text_input("Export prefix", value="dc5_analysis")
        run_sidebar = st.form_submit_button("▶️ Run analysis")

# Big top-of-page run button (so you can’t miss it)
st.subheader("Run")
run_main = st.button("▶️ Run analysis (top)")

run_clicked = run_main or run_sidebar

if run_clicked:
    if st.session_state["loaded_df"] is None:
        st.warning("Please upload a winners file first.")
    else:
        params = dict(order_choice=order_choice, window=int(window), hot_pct=int(hot_pct),
                      cold_pct=int(cold_pct), due_n=int(due_n))
        _ = _hash_params(params)  # keeps cache key stable if you extend later
        st.session_state["results"] = _run_analysis(st.session_state["loaded_df"],
                                                    order_choice, int(window), int(hot_pct), int(cold_pct), int(due_n))

res = st.session_state["results"]
if not res:
    st.info("Upload a file, set your parameters, then click **Run analysis**.")
    st.stop()

left, right = st.columns([2,1])
with left:
    st.subheader("Hottest → coldest (quick view)")
    st.markdown(res["quick"] or "_(no ranking for this window)_")

    st.subheader("Pattern summaries")
    st.dataframe(res["summary"], use_container_width=True)

with right:
    st.subheader("Export")
    st.download_button("Download per-winner CSV", data=res["csv"],
        file_name=f"{prefix}_per_winner.csv", mime="text/csv", use_container_width=True, key="dl_csv")
    st.download_button("Download TXT summary", data=res["txt"],
        file_name=f"{prefix}_summary.txt", mime="text/plain", use_container_width=True, key="dl_txt")

st.caption(f"Ordering used: **{res['order_used']}** · Window={window} · Hot={hot_pct}% · Cold={cold_pct}% · Due last {due_n}")
