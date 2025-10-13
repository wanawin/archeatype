# archetype/pages/wnanalyzr.py
from __future__ import annotations

import io
import math
import re
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# ------- UI CONFIG -------------
# -------------------------------
st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")
st.title("DC-5 Hot/Cold/Due Analyzer")

# -------------------------------
# ------- UTILITIES -------------
# -------------------------------

def _hash_params(d: Dict[str, Any]) -> str:
    m = hashlib.md5()
    m.update(repr(sorted(d.items())).encode())
    return m.hexdigest()

def _clean_combo_to_digits(s: str) -> Optional[List[int]]:
    """
    Extract a 5-digit combo from a string (e.g., '2024-09-10, 04589' -> [0,4,5,8,9]).
    Returns None if not a clear 5-digit sequence.
    """
    if s is None:
        return None
    m = re.search(r"(\d{5})", str(s))
    if not m:
        return None
    return [int(ch) for ch in m.group(1)]

def _ensure_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("hot", "cold", "due", "neutral"):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    return df

def composition_counts(df: pd.DataFrame, tag_col: str, digits_col: str = "digits") -> pd.Series:
    """
    Count how many digits of each winner fall inside df[tag_col] (list-like).
    Robust to missing/empty tag columns.
    """
    if tag_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)

    def _count(row):
        tags = row.get(tag_col, [])
        if isinstance(tags, (list, tuple, set)):
            tagset = set(tags)
        else:
            tagset = set()
        digs = row.get(digits_col, [])
        return sum(int(d) in tagset for d in digs)

    return df.apply(_count, axis=1)

def _rank_hot_cold(freq: np.ndarray, hot_pct: int, cold_pct: int) -> Tuple[List[int], List[int]]:
    """
    Rank digits by frequency (desc). Take top ceil(pct_of_10) as hot and
    bottom ceil(pct_of_10) as cold. Ensure no overlap (disjoint sets).
    """
    order = np.argsort(-freq)  # descending by frequency
    k_hot = max(0, min(10, math.ceil(hot_pct * 10 / 100)))
    k_cold = max(0, min(10, math.ceil(cold_pct * 10 / 100)))

    hot = list(order[:k_hot])
    cold = list(order[-k_cold:]) if k_cold > 0 else []

    # Disjoint-ify (prefer keeping extremes; drop collisions from cold)
    hot_set = set(hot)
    cold = [d for d in cold if d not in hot_set]
    return hot, cold

def _freq_of_window(rows: List[List[int]]) -> np.ndarray:
    """Digit frequency for a list of digit lists."""
    counts = np.zeros(10, dtype=int)
    for digs in rows:
        for d in digs:
            counts[d] += 1
    return counts

def _compute_due(prior_rows: List[List[int]], due_n: int) -> List[int]:
    """
    Digits not seen in last N winners (prior_rows is ordered newest-last).
    """
    if due_n <= 0:
        return []
    recent = prior_rows[-due_n:] if due_n <= len(prior_rows) else prior_rows
    seen = set()
    for digs in recent:
        for d in digs:
            seen.add(d)
    return [d for d in range(10) if d not in seen]

# -------------------------------
# ------- FILE LOADING ----------
# -------------------------------

@st.cache_data(show_spinner=False)
def _load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Load winners from CSV or TXT. Accepts:
      - CSV with columns like ['date','combo'] or just a column containing a 5-digit string
      - TXT: one winner per line, or with other text containing a 5-digit group
    Output: DataFrame with columns ['raw','digits'] in the present order.
    """
    name_lower = filename.lower()
    text = file_bytes.decode(errors="ignore")

    if name_lower.endswith(".csv"):
        buf = io.StringIO(text)
        df = pd.read_csv(buf)
        # try to find a column with a 5-digit string
        candidate = None
        for col in df.columns:
            sample = df[col].astype(str).apply(lambda s: re.search(r"\d{5}", s) is not None)
            if sample.mean() > 0.5:
                candidate = col
                break
        if candidate is None:
            candidate = df.columns[0]
        df["raw"] = df[candidate].astype(str)
    else:
        # TXT
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        df = pd.DataFrame({"raw": rows})

    df["digits"] = df["raw"].apply(_clean_combo_to_digits)
    df = df[~df["digits"].isna()].copy()
    df.reset_index(drop=True, inplace=True)
    return df[["raw", "digits"]]

def _guess_order(df: pd.DataFrame) -> str:
    """
    Guess order:
      - If there is a 'date' column and it's monotonic → choose that.
      - Else default to 'Newest first' (many DC-5 files list most recent first).
    Returns one of: 'Newest first', 'Oldest first'
    """
    for col in df.columns:
        if col == "raw":
            continue
        if "date" in col.lower():
            try:
                s = pd.to_datetime(df[col], errors="raise")
                if s.is_monotonic_increasing:
                    return "Oldest first"
                if s.is_monotonic_decreasing:
                    return "Newest first"
            except Exception:
                pass
    return "Newest first"

# -------------------------------
# ------- ANALYSIS CORE ---------
# -------------------------------

def _annotate_tags_over_history(
    winners: List[List[int]],
    window: int,
    hot_pct: int,
    cold_pct: int,
    due_n: int,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    For each winner i, compute tag sets using the PRIOR `window` draws (true predictive mode),
    then annotate row i with list-like columns: hot, cold, due, neutral.
    """
    chronological = winners[:]  # caller ensures oldest→newest
    n = len(chronological)

    rows = []
    for i in range(n):
        prior = chronological[max(0, i - window): i] if window > 0 else chronological[:i]
        freq = _freq_of_window(prior) if len(prior) > 0 else np.zeros(10, dtype=int)
        hot, cold = _rank_hot_cold(freq, hot_pct, cold_pct)
        due = _compute_due(prior, due_n)
        tagset = set(hot) | set(cold) | set(due)
        neutral = [d for d in range(10) if d not in tagset]
        rows.append({
            "digits": chronological[i],
            "hot": hot,
            "cold": cold,
            "due": due,
            "neutral": neutral,
        })

    per = pd.DataFrame(rows)
    _ensure_tag_columns(per)

    # Quick hottest→coldest based on most recent window
    prior_for_quick = chronological[-window:] if window > 0 else chronological
    freq_quick = _freq_of_window(prior_for_quick)
    order_desc = list(np.argsort(-freq_quick))
    return per, order_desc

@st.cache_data(show_spinner=False)
def _compute_analysis_cache(param_hash: str, compute_fn, *args, **kwargs) -> Dict[str, Any]:
    return compute_fn(*args, **kwargs)

def _compute_analysis(
    df_loaded: pd.DataFrame,
    order_choice: str,
    window: int,
    hot_pct: int,
    cold_pct: int,
    due_n: int,
) -> Dict[str, Any]:
    """Full compute: order data, annotate per-winner tags, build summaries, and exports."""
    if order_choice == "Auto (guess)":
        order_choice = _guess_order(df_loaded)

    if order_choice == "Newest first":
        ordered = df_loaded.iloc[::-1].reset_index(drop=True)
    else:
        ordered = df_loaded.copy()

    winners = ordered["digits"].tolist()

    per_winner, hottest_desc_order = _annotate_tags_over_history(
        winners=winners,
        window=window,
        hot_pct=hot_pct,
        cold_pct=cold_pct,
        due_n=due_n,
    )

    per_winner["hot_hits"]     = composition_counts(per_winner, "hot")
    per_winner["cold_hits"]    = composition_counts(per_winner, "cold")
    per_winner["due_hits"]     = composition_counts(per_winner, "due")
    per_winner["neutral_hits"] = composition_counts(per_winner, "neutral")

    summary_tbl = (
        per_winner[["hot_hits", "cold_hits", "due_hits", "neutral_hits"]]
        .describe()
        .round(3)
    )

    hottest_display = ", ".join(map(str, hottest_desc_order))

    csv_df = per_winner.copy()
    for col in ("digits", "hot", "cold", "due", "neutral"):
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else "")
    csv_bytes = csv_df.to_csv(index=False).encode()

    txt_parts = []
    txt_parts.append("Hottest → coldest (most-recent window):")
    txt_parts.append(hottest_display)
    txt_parts.append("\nComposition summary (hits per tag):")
    txt_parts.append(summary_tbl.to_string())
    txt_bytes = "\n".join(txt_parts).encode()

    return {
        "per_winner": per_winner,
        "csv_bytes": csv_bytes,
        "txt_bytes": txt_bytes,
        "summary_tbl": summary_tbl,
        "hottest_display": hottest_display,
        "order_choice_effective": order_choice,
    }

# -------------------------------
# ----------- UI ----------------
# -------------------------------

with st.sidebar:
    st.header("1) Input")
    up = st.file_uploader("Upload winners file (.csv or .txt)", type=["csv", "txt"])

    st.header("2) File order")
    order_choice = st.radio(
        "Row order in file",
        options=["Auto (guess)", "Newest first", "Oldest first"],
        index=0,
        help="This affects how priors are computed (seeds). If unsure, leave on Auto."
    )

    st.header("3) Run")
    with st.form("controls", clear_on_submit=False):
        window = st.number_input("Window (trailing draws) for hot/cold ranking (0 = all)", min_value=0, value=10, step=1)
        hot_pct = st.slider("% of digits labeled Hot (by frequency)", 0, 100, 30)
        cold_pct = st.slider("% of digits labeled Cold (by frequency)", 0, 100, 30)
        due_n   = st.number_input("Due threshold: 'not seen in last N draws'", min_value=0, value=2, step=1)
        export_prefix = st.text_input("Export file prefix", value="dc5_analysis")
        run_clicked = st.form_submit_button("▶️ Run analysis")

# keep results and loaded df stable between interactions (avoid auto refresh on download)
if "results" not in st.session_state:
    st.session_state["results"] = None
if "loaded_df" not in st.session_state:
    st.session_state["loaded_df"] = None

# Load file (cached)
if up is not None:
    try:
        st.session_state["loaded_df"] = _load_file(up.getvalue(), up.name)
        st.success(f"Loaded {len(st.session_state['loaded_df'])} winners from **{up.name}**")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
else:
    st.info("Upload a winners file to begin (.csv or .txt).")

# Compute when Run is pressed
if run_clicked:
    if st.session_state.get("loaded_df") is None:
        st.warning("Please upload a file first.")
    else:
        params = dict(
            order_choice=order_choice,
            window=int(window),
            hot_pct=int(hot_pct),
            cold_pct=int(cold_pct),
            due_n=int(due_n),
        )
        param_hash = _hash_params(params)
        st.session_state["results"] = _compute_analysis_cache(
            param_hash,
            _compute_analysis,
            st.session_state["loaded_df"],
            order_choice,
            int(window),
            int(hot_pct),
            int(cold_pct),
            int(due_n),
        )

res = st.session_state["results"]

# ------- OUTPUTS -------
if res is None:
    st.stop()

left, right = st.columns([2, 1])

with left:
    st.subheader("Hottest → Coldest (quick view)")
    st.markdown(res["hottest_display"] or "_(no ranking for the chosen window)_")

    st.subheader("Pattern summaries")
    st.dataframe(res["summary_tbl"], use_container_width=True)

with right:
    st.subheader("Export")
    st.download_button(
        "Download full per-winner CSV",
        data=res["csv_bytes"],
        file_name=f"{export_prefix}_per_winner.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_csv_per_winner",
    )
    st.download_button(
        "Download TXT summary",
        data=res["txt_bytes"],
        file_name=f"{export_prefix}_summary.txt",
        mime="text/plain",
        use_container_width=True,
        key="dl_txt_summary",
    )

st.caption(f"Ordering used: **{res['order_choice_effective']}** · Window={window} · Hot={hot_pct}% · Cold={cold_pct}% · Due last {due_n}")
