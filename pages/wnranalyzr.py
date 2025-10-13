# archetype/pages/wnranalyzr.py
from __future__ import annotations

import io
import re
import csv
import math
import json
import itertools as it
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide", initial_sidebar_state="expanded")

# ────────────────────────────── Helpers ──────────────────────────────

VERSION = "v0.5 (robust TXT/CSV loader + run button + hot/cold guards)"

DIGITS = [str(d) for d in range(10)]

def _ensure_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure list-like tag columns exist so we never KeyError."""
    for col in ("hot", "cold", "due", "neutral"):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    return df

def _guess_is_newest_first(series: pd.Series) -> bool:
    """Crude guess: if file has a header col name resembling date/time or index ascending, assume newest first."""
    # Fallback: assume newest first, user can override.
    return True

def parse_winners_from_text(raw: str) -> List[str]:
    """
    Find any 5-digit blocks anywhere in the text.
    Keep as strings to preserve leading zeros.
    Example matches: '00458', '27551', embedded in text, comma, space, etc.
    """
    # Capture 5 consecutive digits bounded by non-digits or boundaries
    # Using lookarounds preserves leading zeros reliably.
    pattern = re.compile(r"(?<!\d)(\d{5})(?!\d)")
    return pattern.findall(raw)

def parse_winners_from_csv_bytes(b: bytes) -> List[str]:
    """
    Try to read a CSV. We accept either:
      - a column that already contains 5-digit strings,
      - or many columns where digits appear; we regex the whole text.
    """
    # Try pandas first (it understands many dialects). If it fails, fall back to text regex.
    try:
        df = pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        # Flatten all cells to one blob of text:
        blob = " ".join(df.astype(str).values.ravel().tolist())
        return parse_winners_from_text(blob)
    except Exception:
        # Fallback: treat as text
        return parse_winners_from_text(b.decode("utf-8", errors="ignore"))

def load_uploaded_file(upload) -> Tuple[List[str], str]:
    """
    Return (winners, note). winners is a list of 5-char strings.
    """
    if upload is None:
        return [], "No file."

    name = upload.name.lower()
    data = upload.getvalue()

    if name.endswith(".txt"):
        winners = parse_winners_from_text(data.decode("utf-8", errors="ignore"))
        return winners, f"Parsed TXT: found {len(winners)} winners."
    elif name.endswith(".csv"):
        winners = parse_winners_from_csv_bytes(data)
        return winners, f"Parsed CSV: found {len(winners)} winners."
    else:
        # Try both ways
        winners = parse_winners_from_text(data.decode("utf-8", errors="ignore"))
        if not winners:
            winners = parse_winners_from_csv_bytes(data)
        return winners, f"Parsed (auto): found {len(winners)} winners."

def order_winners(winners: List[str], order_choice: str) -> List[str]:
    if order_choice == "Newest first":
        return winners
    if order_choice == "Oldest first":
        return winners[::-1]
    # Auto guess
    newest_first = _guess_is_newest_first(pd.Series(winners))
    return winners if newest_first else winners[::-1]

# ───────────────────────── Hot/Cold/Due labeling ──────────────────────

def rank_hottest_coldest(all_winners: List[str], window_draws: int) -> List[str]:
    """
    Return digits 0..9 ordered hottest -> coldest based on frequency within trailing window_draws.
    If window_draws == 0, use entire dataset.
    """
    if not all_winners:
        return DIGITS

    scope = all_winners if window_draws <= 0 else all_winners[:window_draws]
    counts = {d: 0 for d in DIGITS}
    for w in scope:
        for ch in w:
            counts[ch] += 1
    # sort by descending count, then digit
    ranked = sorted(DIGITS, key=lambda d: (-counts[d], d))
    return ranked

def label_hot_cold_neutral(digits_ranked: List[str], hot_pct: int, cold_pct: int) -> Dict[str, str]:
    """
    Map each digit to label: 'hot', 'cold', or 'neutral', based on top/bottom percentages.
    """
    n = len(digits_ranked)
    k_hot = max(0, round(n * (hot_pct / 100.0)))
    k_cold = max(0, round(n * (cold_pct / 100.0)))

    hot_set = set(digits_ranked[:k_hot])
    cold_set = set(digits_ranked[-k_cold:]) if k_cold else set()
    mapping = {}
    for d in DIGITS:
        if d in hot_set:
            mapping[d] = "hot"
        elif d in cold_set:
            mapping[d] = "cold"
        else:
            mapping[d] = "neutral"
    return mapping

def compute_due_set(all_winners: List[str], due_threshold: int) -> set:
    """
    Digits not seen in the last 'due_threshold' draws are 'due'.
    0 means 'no due' rule (empty set).
    """
    if due_threshold <= 0 or not all_winners:
        return set()

    window = all_winners[:due_threshold]
    seen = set(ch for w in window for ch in w)
    return set(DIGITS) - seen

def tag_each_winner(all_winners: List[str],
                    window_draws: int,
                    hot_pct: int,
                    cold_pct: int,
                    due_threshold: int) -> pd.DataFrame:
    """
    Build a per-winner dataframe with columns:
      winner, hot(list), cold(list), due(list), neutral(list),
      hot_hits, cold_hits, due_hits, neutral_hits
    """
    if not all_winners:
        return pd.DataFrame(columns=["winner", "hot", "cold", "due", "neutral",
                                     "hot_hits", "cold_hits", "due_hits", "neutral_hits"])

    # Rank digits using requested trailing window (relative to "now")
    ranked = rank_hottest_coldest(all_winners, window_draws)
    map_hotcold = label_hot_cold_neutral(ranked, hot_pct, cold_pct)
    due_set = compute_due_set(all_winners, due_threshold)

    rows = []
    for w in all_winners:
        tags = [map_hotcold[ch] for ch in w]
        groups = {
            "hot": [ch for ch in w if map_hotcold[ch] == "hot"],
            "cold": [ch for ch in w if map_hotcold[ch] == "cold"],
            "neutral": [ch for ch in w if map_hotcold[ch] == "neutral"],
            "due": [ch for ch in w if ch in due_set],
        }
        rows.append({
            "winner": w,
            "hot": groups["hot"],
            "cold": groups["cold"],
            "due": groups["due"],
            "neutral": groups["neutral"],
            "hot_hits": len(groups["hot"]),
            "cold_hits": len(groups["cold"]),
            "due_hits": len(groups["due"]),
            "neutral_hits": len(groups["neutral"]),
        })

    return pd.DataFrame(rows)

def composition_counts(df: pd.DataFrame, col: str) -> pd.Series:
    """
    For a list-like column (e.g., 'hot'), compute how many of those tags each winner has.
    Return a value-counts series indexed by 0..5.
    """
    df = _ensure_tag_columns(df)
    if col not in df.columns:
        return pd.Series(dtype=int)

    counts = df[col].apply(lambda xs: len(xs) if isinstance(xs, (list, tuple)) else 0)
    vc = counts.value_counts().sort_index()
    # ensure 0..5 index exists
    idx = pd.Index(range(0, 6), name=col)
    return vc.reindex(idx, fill_value=0)

# ─────────────────────────────── UI ───────────────────────────────────

with st.sidebar:
    st.title("DC-5 Filter Tracker — Hot/Cold/Due")
    st.caption("Upload winners, choose order, set parameters, then **Run analysis**.")

    uploaded = st.file_uploader("Upload winners (.csv or .txt)", type=["csv", "txt"])

    st.markdown("### 2) File order")
    order_choice = st.radio(
        "Row order in file",
        ["Auto (guess)", "Newest first", "Oldest first"],
        index=1,  # default to Newest first (most users keep newest first)
        help="Set the chronological order of rows in your file."
    )

    st.markdown("### 4) Hot/Cold/Due params")
    window_draws = st.number_input("Window (trailing draws) for hot/cold ranking (0 = all)", min_value=0, max_value=5000, value=10)
    hot_pct = st.slider("% of digits labeled Hot (by frequency)", 0, 100, 30)
    cold_pct = st.slider("% of digits labeled Cold (by frequency)", 0, 100, 30)
    due_threshold = st.number_input("Due threshold: 'not seen in last N draws'", min_value=0, max_value=5000, value=2)

    st.markdown("### 5) Export control")
    export_prefix = st.text_input("Export file prefix", value="dc5_analysis")

st.title(f"DC5 Hot/Cold/Due Analyzer — {VERSION}")

st.subheader("Run")
run_clicked_top = st.button("▶️ Run analysis (top)")

status_box = st.empty()

# quick view placeholders
hotcold_col, export_col = st.columns([2, 1])

with hotcold_col:
    st.markdown("### Hottest → coldest (quick view)")
    hotcold_placeholder = st.empty()

with export_col:
    st.markdown("### Export")
    csv_btn_slot = st.empty()
    txt_btn_slot = st.empty()

st.markdown("### Pattern summaries")
summary_placeholder = st.empty()

# ───────────────────────────── Execution ──────────────────────────────

def do_run():
    # Load winners
    winners, note = load_uploaded_file(uploaded)
    if not winners:
        status_box.warning(f"Loaded 0 rows. {note}  \nMake sure the file contains 5-digit winners anywhere in the text (we preserve leading zeros).")
        # Show some empty placeholders so the layout remains stable
        hotcold_placeholder.write("0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
        summary_placeholder.write(pd.DataFrame(columns=["hot_hits","cold_hits","due_hits","neutral_hits"]))
        csv_btn_slot.button("Download per-winner CSV", disabled=True)
        txt_btn_slot.button("Download TXT summary", disabled=True)
        return

    winners = order_winners(winners, order_choice)
    status_box.success(f"Loaded {len(winners)} rows from **{uploaded.name}** (order: {order_choice}).")

    # Tag per-winner
    per_winner = tag_each_winner(
        winners,
        window_draws=window_draws,
        hot_pct=hot_pct,
        cold_pct=cold_pct,
        due_threshold=due_threshold
    )
    per_winner = _ensure_tag_columns(per_winner)

    # Quick view: hottest → coldest now
    ranked = rank_hottest_coldest(winners, window_draws)
    hotcold_placeholder.write(", ".join(ranked))

    # Pattern summaries table (hits distribution)
    table = pd.DataFrame({
        "hot_hits": composition_counts(per_winner, "hot"),
        "cold_hits": composition_counts(per_winner, "cold"),
        "due_hits": composition_counts(per_winner, "due"),
        "neutral_hits": composition_counts(per_winner, "neutral"),
    })
    summary_placeholder.dataframe(table)

    # Downloads
    # 1) per-winner CSV
    csv_bytes = per_winner.to_csv(index=False).encode("utf-8")
    csv_btn_slot.download_button(
        "Download per-winner CSV",
        data=csv_bytes,
        file_name=f"{export_prefix}_per_winner.csv",
        mime="text/csv",
        use_container_width=True,
    )
    # 2) text summary
    txt_summary = io.StringIO()
    txt_summary.write(f"DC5 Hot/Cold/Due Analyzer — {VERSION}\n")
    txt_summary.write(f"Rows: {len(winners)} | Order: {order_choice}\n")
    txt_summary.write(f"Window={window_draws}, hot%={hot_pct}, cold%={cold_pct}, due_threshold={due_threshold}\n\n")
    txt_summary.write("Hottest → coldest:\n")
    txt_summary.write(", ".join(ranked) + "\n\n")
    txt_summary.write("Hits distribution (rows are counts of winners having k tagged digits):\n")
    txt_summary.write(table.to_string() + "\n")
    txt_btn_slot.download_button(
        "Download TXT summary",
        data=txt_summary.getvalue().encode("utf-8"),
        file_name=f"{export_prefix}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

if run_clicked_top:
    do_run()
