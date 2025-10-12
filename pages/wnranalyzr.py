import io
import re
import json
import math
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# Helper functions
# -------------------------------

WIN_RE = re.compile(r"\b(\d{5})\b")

@st.cache_data(show_spinner=False)
def parse_input(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Return a DataFrame with at least two columns: ['draw_index', 'winner'].
    draw_index is 0 for oldest → increasing to most recent (we'll flip later if needed).
    This parser tries to be forgiving:
    - CSV: looks for a 5-digit winner column (first match wins), else extracts any 5-digit tokens from all cells.
    - TXT: treats each line, extracting first 5-digit token.
    """
    name = filename.lower()
    # naive read first
    winners: List[str] = []
    rows: List[Tuple[int, str]] = []

    if name.endswith((".csv", ".tsv")):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
        # look for exact 5-digit values in any column
        for _, row in df.iterrows():
            found = None
            for col in df.columns:
                val = str(row[col])
                m = WIN_RE.search(val)
                if m:
                    found = m.group(1)
                    break
            if found:
                winners.append(found)
    else:
        # TXT or anything else → read lines
        text = file_bytes.decode(errors='ignore')
        for line in text.splitlines():
            m = WIN_RE.search(line)
            if m:
                winners.append(m.group(1))

    # Build DataFrame
    for i, w in enumerate(winners):
        rows.append((i, w))
    out = pd.DataFrame(rows, columns=["draw_index", "winner"])  # oldest first by default
    return out


def digits_of(win: str) -> List[int]:
    return [int(c) for c in win]


def sum_and_parity(win: str) -> Tuple[int, str]:
    s = sum(digits_of(win))
    return s, ("Even" if s % 2 == 0 else "Odd")


# Default bucket edges — easily adjustable in UI
DEFAULT_BUCKETS = {
    "Very Low": (0, 15),
    "Low": (16, 24),
    "Mid": (25, 33),
    "High": (34, 45),
}


def bucket_of(total: int, buckets: Dict[str, Tuple[int, int]]) -> str:
    for name, (lo, hi) in buckets.items():
        if lo <= total <= hi:
            return name
    return "OutOfRange"


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame, buckets: Dict[str, Tuple[int, int]], most_recent_on_top: bool) -> pd.DataFrame:
    """Compute per-draw features: sum, parity, bucket, digits, etc.
    Returns a copy with columns:
      - draw_num (0 is oldest → increasing)
      - winner, sum, parity, bucket
    """
    if most_recent_on_top:
        # If file provided newest first, flip to make 0 oldest
        df = df.iloc[::-1].reset_index(drop=True)
        df["draw_index"] = range(len(df))

    df = df.copy()
    sums, parities, buckets_col, digs = [], [], [], []
    for w in df["winner"].astype(str):
        total, par = sum_and_parity(w)
        sums.append(total)
        parities.append(par)
        buckets_col.append(bucket_of(total, buckets))
        digs.append(digits_of(w))

    df["sum"] = sums
    df["sum_parity"] = parities
    df["sum_bucket"] = buckets_col
    df["digits"] = digs
    df.rename(columns={"draw_index": "draw_num"}, inplace=True)
    return df


def auto_detect_order(df: pd.DataFrame) -> bool:
    """Heuristic: if file looks like newest first.
    We check if the left-most date-looking column (if any) descends, or if there are sequence hints.
    Fallback False (assume oldest→newest). Returns True if looks like most-recent on top.
    """
    # Very light heuristic: if the first 10 winners have decreasing sums a lot, that's not reliable.
    # We'll instead expose a toggle and keep this simple: default False.
    return False


def rank_hot_cold(labels: List[int], window: int) -> pd.DataFrame:
    """Return counts and ranks for digits 0..9 within the trailing window.
    labels: list of digits flattened from winners, oldest→newest.
    """
    series = pd.Series(labels[-window:]) if window > 0 else pd.Series(labels)
    counts = series.value_counts().reindex(range(10), fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100.0).round(2) if total else counts * 0
    out = pd.DataFrame({"digit": range(10), "count": counts.values, "pct": pct.values})
    out.sort_values(["count", "digit"], ascending=[False, True], inplace=True)
    out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def classify_hot_cold_due(table: pd.DataFrame, hot_pct: float, cold_pct: float, due_gap: int, hist_digits: List[int]) -> pd.DataFrame:
    """Given ranked table, label each digit as Hot/Neutral/Cold; add Due if it hasn't appeared in last `due_gap` draws.
    - hot_pct = top X% of digits by count (e.g., 30 → top 3 digits)
    - cold_pct = bottom Y% of digits by count (e.g., 30 → bottom 3 digits)
    """
    k_hot = max(1, round(10 * (hot_pct / 100.0)))
    k_cold = max(1, round(10 * (cold_pct / 100.0)))

    tbl = table.copy()
    tbl["hc_label"] = "Neutral"
    # Mark hot
    hot_idx = tbl.index[:k_hot]
    tbl.loc[hot_idx, "hc_label"] = "Hot"
    # Mark cold
    cold_idx = tbl.index[-k_cold:]
    tbl.loc[cold_idx, "hc_label"] = np.where(tbl.loc[cold_idx, "hc_label"] == "Hot", "Hot", "Cold")

    # Due: last occurrence index per digit
    last_occ = {d: -math.inf for d in range(10)}
    for i, d in enumerate(hist_digits):
        last_occ[d] = i
    last_index = len(hist_digits) - 1

    due_flags = []
    for d in tbl["digit"]:
        gap = (last_index - last_occ[d]) if np.isfinite(last_occ[d]) else math.inf
        due_flags.append(gap >= due_gap)
    tbl["due"] = due_flags
    tbl["tags"] = [
        ",".join(filter(None, [hc, "Due" if due else ""]))
        for hc, due in zip(tbl["hc_label"], tbl["due"])
    ]
    return tbl


def summarize_winners(df: pd.DataFrame, tag_table: pd.DataFrame) -> pd.DataFrame:
    """For each winner, compute tag composition: counts of Hot/Cold/Neutral, Due, overlaps.
    Returns a per-winner summary.
    """
    tag_map = tag_table.set_index("digit")["tags"].to_dict()

    rows = []
    for _, r in df.iterrows():
        digs = r["digits"]
        tags = [tag_map[d].split(",") if tag_map.get(d) else [""] for d in digs]
        flat = [t for sub in tags for t in sub if t]
        hot = sum("Hot" in tag_map[d] for d in digs)
        cold = sum("Cold" in tag_map[d] for d in digs)
        neutral = 5 - hot - cold
        due = sum("Due" in tag_map[d] for d in digs)
        overlaps = sum("Hot" in tag_map[d] and "Due" in tag_map[d] for d in digs) + \
                   sum("Cold" in tag_map[d] and "Due" in tag_map[d] for d in digs)
        doubled = any(digs.count(d) >= 2 for d in set(digs))
        rows.append({
            "draw_num": r["draw_num"],
            "winner": r["winner"],
            "sum": r["sum"],
            "sum_parity": r["sum_parity"],
            "sum_bucket": r["sum_bucket"],
            "hot": hot,
            "cold": cold,
            "neutral": neutral,
            "due": due,
            "overlap_hot_due_or_cold_due": overlaps,
            "has_double": doubled,
        })
    return pd.DataFrame(rows)


def make_txt_summary(tag_table: pd.DataFrame, per_win: pd.DataFrame) -> str:
    lines = []
    lines.append("Digit ranking (hottest → coldest):")
    for _, r in tag_table.sort_values(["count", "digit"], ascending=[False, True]).iterrows():
        lines.append(f"  {int(r['digit'])}: count={int(r['count'])}, pct={r['pct']}%, tags={r['tags']}")

    lines.append("")
    lines.append("Per-winner composition (last 25 shown):")
    tail = per_win.tail(25)
    for _, r in tail.iterrows():
        lines.append(
            f"  #{int(r['draw_num'])} {r['winner']} | sum={int(r['sum'])} {r['sum_parity']} {r['sum_bucket']} | "
            f"hot={r['hot']} cold={r['cold']} neutral={r['neutral']} due={r['due']} overlap={r['overlap_hot_due_or_cold_due']} double={r['has_double']}"
        )
    return "\n".join(lines)


# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")
st.title("DC5 Hot/Cold/Due Analyzer")

with st.sidebar:
    st.header("1) Upload history file")
    file = st.file_uploader("CSV or TXT of winners (any column/line containing a 5-digit winner)", type=["csv", "tsv", "txt"]) 

    st.header("2) Order detection")
    auto_order = st.checkbox("Auto-detect 'most recent on top' (heuristic)", value=False)
    most_recent_on_top = False
    if file is not None and auto_order:
        tmp_df = parse_input(file.getvalue(), file.name)
        most_recent_on_top = auto_detect_order(tmp_df)
    most_recent_on_top = st.toggle("File is most-recent → oldest (top→bottom)", value=most_recent_on_top, help="Toggle if your file lists the newest draw first.")

    st.header("3) Sum buckets")
    cols = st.columns(4)
    bnames = list(DEFAULT_BUCKETS.keys())
    bucket_vals = {}
    for i, name in enumerate(bnames):
        lo, hi = DEFAULT_BUCKETS[name]
        with cols[i % 4]:
            lo_i = st.number_input(f"{name} min", value=int(lo), step=1, key=f"min_{name}")
            hi_i = st.number_input(f"{name} max", value=int(hi), step=1, key=f"max_{name}")
        bucket_vals[name] = (lo_i, hi_i)

    st.header("4) Hot/Cold/Due params")
    window = st.number_input("Window (trailing draws) for hot/cold ranking (0 = all)", value=200, min_value=0, step=10)
    hot_pct = st.slider("% of digits labeled Hot (by frequency)", 10, 50, 30, step=5)
    cold_pct = st.slider("% of digits labeled Cold (by frequency)", 10, 50, 30, step=5)
    due_gap = st.number_input("Due threshold: 'not seen in last N draws'", value=30, min_value=1, step=1)

    st.header("5) Export control")
    export_prefix = st.text_input("Export file prefix", value="dc5_analysis")

if file is None:
    st.info("Upload a CSV or TXT history file to begin.")
    st.stop()

# Parse and feature-build
raw_df = parse_input(file.getvalue(), file.name)
feat_df = build_features(raw_df, bucket_vals, most_recent_on_top)

# Flatten digits for ranking
all_digits = [d for row in feat_df["digits"] for d in row]
rank_tbl = rank_hot_cold(all_digits, window)
labels_tbl = classify_hot_cold_due(rank_tbl, hot_pct, cold_pct, due_gap, all_digits)

# Summaries
per_winner = summarize_winners(feat_df, labels_tbl)

# -------------------------------
# Display
# -------------------------------

left, right = st.columns([1, 1])
with left:
    st.subheader("Digit ranking & labels (Hot/Cold/Due)")
    st.dataframe(labels_tbl, use_container_width=True)

    st.subheader("Hottest → coldest (quick view)")
    hot_str = ", ".join(str(d) for d in labels_tbl.sort_values(["count","digit"], ascending=[False, True])["digit"].tolist())
    st.code(hot_str)

with right:
    st.subheader("Per-winner composition")
    st.dataframe(per_winner, use_container_width=True)

# Aggregate patterns helpful for prediction
st.subheader("Pattern summaries")

# 1) How many winners contain ≥K hot digits, cold digits, due digits
def composition_counts(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return pd.Series({
        ">=1": int((s >= 1).sum()),
        ">=2": int((s >= 2).sum()),
        ">=3": int((s >= 3).sum()),
        ">=4": int((s >= 4).sum()),
        "=5": int((s >= 5).sum()),
    })

comp = pd.DataFrame({
    "hot": composition_counts(per_winner, "hot"),
    "cold": composition_counts(per_winner, "cold"),
    "due": composition_counts(per_winner, "due"),
}).T
st.write("Counts of winners having at least K tagged digits:")
st.dataframe(comp, use_container_width=True)

# 2) Doubles when Hot/Due present
with st.expander("Double-rate conditional on tags"):
    total = len(per_winner)
    def rate(mask):
        den = mask.sum()
        if den == 0:
            return 0.0
        return float(per_winner.loc[mask, "has_double"].mean())
    r_any_hot = rate(per_winner["hot"] >= 1)
    r_2p_hot  = rate(per_winner["hot"] >= 2)
    r_any_due = rate(per_winner["due"] >= 1)
    st.write({
        "double_rate_any_hot": round(r_any_hot, 3),
        "double_rate_2plus_hot": round(r_2p_hot, 3),
        "double_rate_any_due": round(r_any_due, 3),
    })

# -------------------------------
# Exports
# -------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()

def to_txt_bytes(text: str) -> bytes:
    return text.encode()

summary_txt = make_txt_summary(labels_tbl, per_winner)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download digit labels (CSV)",
        data=to_csv_bytes(labels_tbl),
        file_name=f"{export_prefix}_digit_labels.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Download per-winner composition (CSV)",
        data=to_csv_bytes(per_winner),
        file_name=f"{export_prefix}_per_winner.csv",
        mime="text/csv",
    )
with c3:
    st.download_button(
        "Download summary (TXT)",
        data=to_txt_bytes(summary_txt),
        file_name=f"{export_prefix}_summary.txt",
        mime="text/plain",
    )

st.caption("Tip: tweak window / hot% / cold% / due gap and re-export to compare scenarios.")
