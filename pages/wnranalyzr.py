# wnranalyzr.py
# DC5 Hot/Cold/Due Analyzer — with Run button, robust TXT parsing, and rolling per-winner stats
# Drop-in Streamlit page/app.

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
WIN_CONTIGUOUS = re.compile(r"(?<!\d)(\d{5})(?!\d)")
WIN_HYPHENATED = re.compile(r"(?<!\d)(\d)[\-\–](\d)[\-\–](\d)[\-\–](\d)[\-\–](\d)(?!\d)")

def _normalize_winner_token(tok: str) -> str:
    """Ensure 5-char, keep leading zeros."""
    tok = tok.strip()
    return tok if len(tok) == 5 else tok.zfill(5)

def parse_winners_from_text(text: str) -> List[str]:
    """
    Parse winners from arbitrary text.
    Supports 5-digit contiguous ('00458') and hyphenated ('0-0-4-5-8' or '0–0–4–5–8').
    Preserves order of appearance.
    """
    winners: List[Tuple[int, str]] = []  # (start_pos, winner)

    # Find hyphenated matches first (to avoid them being re-matched by \d{5} fragments)
    for m in WIN_HYPHENATED.finditer(text):
        winner = "".join(m.groups())
        winners.append((m.start(), _normalize_winner_token(winner)))

    # Also accept contiguous 5-digit blocks
    for m in WIN_CONTIGUOUS.finditer(text):
        winners.append((m.start(), _normalize_winner_token(m.group(1))))

    # Stable sort by file position
    winners.sort(key=lambda x: x[0])
    return [w for _, w in winners]

def parse_uploaded_file(file) -> List[str]:
    """
    Accepts CSV or TXT.
    - TXT: parse anywhere in the text (hyphenated & contiguous).
    - CSV: look for a column that looks like winners (5-digit or hyphenated) and normalize.
    """
    name = (file.name or "").lower()
    data = file.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="ignore")

    if name.endswith(".txt") or not name.endswith(".csv"):
        winners = parse_winners_from_text(data)
        return winners

    # CSV
    df = pd.read_csv(io.StringIO(data))
    # Try to find a plausible column
    maybe_cols = list(df.columns)
    # Prefer commonly named columns
    preferred = ["winner", "winning", "result", "results", "combo", "number", "numbers"]
    for p in preferred:
        for c in maybe_cols:
            if p in str(c).lower():
                maybe_cols.insert(0, maybe_cols.pop(maybe_cols.index(c)))
                break

    winners: List[str] = []
    matched = False
    for c in maybe_cols:
        vals = df[c].astype(str).fillna("").tolist()
        tmp: List[str] = []
        ok = 0
        for v in vals:
            m1 = WIN_HYPHENATED.fullmatch(v.strip())
            m2 = WIN_CONTIGUOUS.fullmatch(v.strip())
            if m1:
                tmp.append("".join(m1.groups()))
                ok += 1
            elif m2:
                tmp.append(_normalize_winner_token(m2.group(1)))
                ok += 1
            else:
                # Try to find inside the cell
                got = parse_winners_from_text(v)
                if len(got) == 1:
                    tmp.append(got[0]); ok += 1
                elif len(got) > 1:
                    # ambiguous cell, skip column
                    ok = -999
                    break
        if ok > 0:
            winners = [_normalize_winner_token(t) for t in tmp if t]
            matched = True
            break
    if not matched:
        winners = parse_winners_from_text(data)
    return winners

def detect_order(winners: List[str]) -> str:
    """
    Heuristic: guess file is 'Newest first'.
    We expose the radio so the user can override easily.
    """
    # You can extend with smarter heuristics later. For now keep simple & transparent.
    return "Newest first"


# -----------------------------
# Rolling hot/cold/due labeling
# -----------------------------
@dataclass
class Params:
    window_for_hotcold: int   # 0 => all prior draws
    pct_hot: int              # % of digits (0..100) labeled as hot (by frequency)
    pct_cold: int             # % of digits labeled as cold (by frequency)
    due_threshold: int        # "not seen in last N draws" => due

def label_hot_cold(counts: Dict[str, int], pct_hot: int, pct_cold: int) -> Tuple[set, set]:
    """
    From a dict of digit->count, mark top pct_hot% as hot and bottom pct_cold% as cold.
    Ties are handled by rank; if overlap occurs because of rounding, hot wins precedence.
    """
    # Ensure all digits present
    full = {str(d): counts.get(str(d), 0) for d in range(10)}
    ser = pd.Series(full).sort_values(ascending=False)  # high->low
    n = len(ser)  # 10
    n_hot = int(np.floor(n * (pct_hot / 100.0)))
    n_cold = int(np.floor(n * (pct_cold / 100.0)))

    hot = set(ser.index[:n_hot]) if n_hot > 0 else set()
    cold = set(ser.sort_values(ascending=True).index[:n_cold]) if n_cold > 0 else set()
    # Avoid overlap: if a digit ends up in both because of rounding, prefer hot
    cold -= hot
    return hot, cold

def freq_counts(history: List[str]) -> Dict[str, int]:
    counts = {str(d): 0 for d in range(10)}
    for w in history:
        for ch in w:
            counts[ch] += 1
    return counts

def due_set(history: List[str], due_threshold: int) -> set:
    """
    Digits not seen in last 'due_threshold' winners.
    If due_threshold == 0 => empty set (nothing is due).
    """
    if due_threshold <= 0:
        return set()
    recent = history[-due_threshold:] if due_threshold <= len(history) else history[:]
    seen = set(ch for w in recent for ch in w)
    return set(str(d) for d in range(10)) - seen


# -----------------------------
# Analysis core
# -----------------------------
def analyze(
    winners_ordered_oldest_to_newest: List[str],
    params: Params
) -> Tuple[pd.DataFrame, Dict[str, any], List[str], List[str]]:
    """
    For each winner i, compute hot/cold/due using **only the history before i**.
    Returns:
    - per_winner DataFrame
    - overall summary dict
    - hottest->coldest (list)
    - messages (warnings/info)
    """
    msgs: List[str] = []
    W = winners_ordered_oldest_to_newest
    if len(W) == 0:
        return pd.DataFrame(), {}, [str(d) for d in range(10)], ["No winners parsed."]

    rows = []
    # Also produce a "global" hottest->coldest quick view from the last window (or all)
    refer_history = W if params.window_for_hotcold == 0 else W[-params.window_for_hotcold:]
    global_counts = freq_counts(refer_history)
    ordered_digits = list(pd.Series(global_counts).sort_values(ascending=False).index)

    # rolling analysis
    for i, winner in enumerate(W):
        prior = W[:i]  # history before this draw
        if len(prior) == 0:
            # nothing to label from… record empty stats
            rows.append({
                "index": i,
                "winner": winner,
                "sum": sum(int(ch) for ch in winner),
                "hot_hits": 0,
                "cold_hits": 0,
                "due_hits": 0,
                "neutral_hits": 0,
                "hot_and_due_hits": 0,
                "hot_digits": "",
                "cold_digits": "",
                "due_digits": "",
            })
            continue

        # Limit the window for hot/cold ranking
        if params.window_for_hotcold > 0:
            window_hist = prior[-params.window_for_hotcold:]
        else:
            window_hist = prior

        counts = freq_counts(window_hist)
        hot, cold = label_hot_cold(counts, params.pct_hot, params.pct_cold)
        due = due_set(prior, params.due_threshold)

        # Count hits & overlaps for this winner
        digits = list(winner)
        hot_hits = sum(1 for d in digits if d in hot)
        cold_hits = sum(1 for d in digits if d in cold)
        due_hits = sum(1 for d in digits if d in due)

        # "Neutral" means not labeled hot/cold/due (even if hot & due overlaps exist, they're not neutral)
        labeled = hot | cold | due
        neutral_hits = sum(1 for d in digits if d not in labeled)

        # Overlap (digit that is both hot and due). Count per occurrences in the 5 digits.
        hot_and_due_hits = sum(1 for d in digits if (d in hot and d in due))

        rows.append({
            "index": i,
            "winner": winner,
            "sum": sum(int(ch) for ch in winner),
            "hot_hits": hot_hits,
            "cold_hits": cold_hits,
            "due_hits": due_hits,
            "neutral_hits": neutral_hits,
            "hot_and_due_hits": hot_and_due_hits,
            "hot_digits": ",".join(sorted(hot)),
            "cold_digits": ",".join(sorted(cold)),
            "due_digits": ",".join(sorted(due)),
        })

    df = pd.DataFrame(rows)

    # Overall summary (simple frequency of bucket hits)
    overall = {
        "n_winners": len(W),
        "avg_hot_hits": float(df["hot_hits"].mean()),
        "avg_cold_hits": float(df["cold_hits"].mean()),
        "avg_due_hits": float(df["due_hits"].mean()),
        "avg_neutral_hits": float(df["neutral_hits"].mean()),
        "avg_hot_and_due_hits": float(df["hot_and_due_hits"].mean()),
    }

    return df, overall, ordered_digits, msgs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")

st.title("DC5 Hot/Cold/Due Analyzer — v0.5 (Run button + robust TXT parsing)")

with st.sidebar:
    st.header("1) Input")
    uploaded = st.file_uploader("Upload winners (.csv or .txt)", type=["csv", "txt"])

    st.header("2) File order")
    order_choice = st.radio(
        "Row order in file:",
        ["Auto (guess)", "Newest first", "Oldest first"],
        index=1,  # default to Newest first
        help="This matters because labels for each winner are computed from prior draws."
    )

    st.header("3) Run controls")
    run_top = st.button("▶️ Run analysis (top)")

    st.header("4) Hot/Cold/Due params")
    window = st.number_input(
        "Window (trailing draws) for hot/cold ranking (0 = all)",
        min_value=0, max_value=2000, step=1, value=10
    )
    pct_hot = st.slider("% of digits labeled Hot (by frequency)", 0, 50, 30)
    pct_cold = st.slider("% of digits labeled Cold (by frequency)", 0, 50, 30)
    due_threshold = st.number_input(
        "Due threshold: 'not seen in last N draws'",
        min_value=0, max_value=200, step=1, value=2
    )

    st.header("5) Export control")
    export_prefix = st.text_input("Export file prefix", value="dc5_analysis")

st.markdown("### Run")
col_run, _ = st.columns([1, 4])
with col_run:
    run_mid = st.button("▶️ Run analysis (duplicate)")

params = Params(
    window_for_hotcold=int(window),
    pct_hot=int(pct_hot),
    pct_cold=int(pct_cold),
    due_threshold=int(due_threshold)
)

do_run = run_top or run_mid

# Placeholders
hotcold_quick = st.empty()
export_col1, export_col2 = st.columns([1, 1])
table_placeholder = st.empty()
summary_placeholder = st.empty()
message_box = st.empty()

parsed_winners: List[str] = []
parsed_info = st.empty()

if uploaded is not None:
    try:
        uploaded.seek(0)
        winners_raw = parse_uploaded_file(uploaded)
        # Short note about parse result
        parsed_info.info(
            f"Parsed: **{len(winners_raw)}** winners found in the file. "
            f"(We accept both 5-digit forms like `00458` and hyphenated `0-0-4-5-8`.)"
        )

        # Order handling
        guessed = detect_order(winners_raw)
        chosen = order_choice if order_choice != "Auto (guess)" else guessed
        if chosen == "Newest first":
            winners = winners_raw[:]
        elif chosen == "Oldest first":
            winners = winners_raw[::-1]
        else:
            winners = winners_raw[:]  # fallback to guess
            st.caption(f"Auto guess order: **{guessed}**")
        # For rolling calc we need oldest->newest
        winners_old_to_new = winners[::-1] if chosen == "Newest first" else winners[:]

        if do_run:
            df, overall, ordered_digits, msgs = analyze(winners_old_to_new, params)

            hotcold_quick.markdown(
                f"### Hottest → coldest (quick view)\n\n"
                f"`{', '.join(ordered_digits)}`"
            )

            if df.empty:
                table_placeholder.warning("No winners parsed. Please check the input file.")
            else:
                table_placeholder.dataframe(df, use_container_width=True)

                # Summary block
                with summary_placeholder.container():
                    st.markdown("### Summary")
                    colA, colB = st.columns(2)
                    with colA:
                        st.json(overall, expanded=False)
                    with colB:
                        st.markdown(
                            f"**Parameters**  \n"
                            f"- Window for hot/cold: **{params.window_for_hotcold}**  \n"
                            f"- % Hot: **{params.pct_hot}%** | % Cold: **{params.pct_cold}%**  \n"
                            f"- Due threshold: **not seen in last {params.due_threshold} draws**"
                        )

                # Exports
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                txt_summary = io.StringIO()
                txt_summary.write("DC5 Hot/Cold/Due Analyzer Results\n")
                txt_summary.write(f"Total winners: {overall.get('n_winners', 0)}\n\n")
                txt_summary.write("Hottest→Coldest (quick view): " + ", ".join(ordered_digits) + "\n\n")
                for k, v in overall.items():
                    txt_summary.write(f"{k}: {v}\n")

                with export_col1:
                    st.download_button(
                        "Download per-winner CSV",
                        data=csv_bytes,
                        file_name=f"{export_prefix}_per_winner.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with export_col2:
                    st.download_button(
                        "Download TXT summary",
                        data=txt_summary.getvalue().encode("utf-8"),
                        file_name=f"{export_prefix}_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            # Messages
            if msgs:
                message_box.info("\n".join(msgs))

        else:
            hotcold_quick.markdown("### Hottest → coldest (quick view)\n\n*Run the analysis to populate.*")

    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info("Upload a `.txt` (hyphenated or 5-digit) or `.csv` file with winners, then click **Run analysis**.")
