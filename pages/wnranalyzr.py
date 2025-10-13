import io
import re
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict

# ------------------------------
# Helpers
# ------------------------------

DIGITS = list("0123456789")

def parse_uploaded(file) -> pd.DataFrame:
    """
    Accepts .csv or .txt.
    - CSV: looks for any 5-digit column; otherwise scans whole text.
    - TXT: scans entire text for 5-digit winners, preserving leading zeros.
    Also accepts hyphenated like 0-0-4-5-8.
    Returns DataFrame with columns: ['raw', 'winner', 'digits'] (digits = list[str] len=5)
    """
    name = file.name.lower()

    def scan_text(text: str) -> list[str]:
        winners = []
        # 5 contiguous digits
        winners += re.findall(r'(?<!\d)(\d{5})(?!\d)', text)
        # hyphenated (e.g. 0-0-4-5-8)
        winners += ["".join(g) for g in re.findall(r'(?<!\d)(\d)-(\d)-(\d)-(\d)-(\d)(?!\d)', text)]
        return winners

    if name.endswith(".csv"):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        # try to find a 5-digit column
        five_digit_cols = [c for c in df.columns
                           if df[c].astype(str).str.fullmatch(r"\d{5}").fillna(False).any()]
        winners = []
        if five_digit_cols:
            for c in five_digit_cols:
                winners += df[c].astype(str).str.extract(r"(\d{5})", expand=False).dropna().tolist()
        else:
            # fallback: whole-file scan
            text = df.to_csv(index=False)
            winners = scan_text(text)
    else:
        text = io.TextIOWrapper(file, encoding="utf-8", errors="ignore").read()
        winners = scan_text(text)

    winners = [w.strip() for w in winners if re.fullmatch(r"\d{5}", w)]
    data = [{'raw': w, 'winner': w, 'digits': list(w)} for w in winners]
    return pd.DataFrame(data)


def order_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode in {'auto','newest','oldest'}
    'auto' guesses from simple sequence in input (assumes newest first if ambiguous).
    """
    if mode == "newest":
        return df.reset_index(drop=True)
    if mode == "oldest":
        return df.iloc[::-1].reset_index(drop=True)
    # auto guess: if the first ~20 are unique (they are), treat as newest-first
    return df.reset_index(drop=True)


def rank_hot_cold(df: pd.DataFrame, window: int, pct_hot: int, pct_cold: int):
    """
    Returns:
      - hottest_to_coldest: list of digits from hottest -> coldest (by freq in trailing window)
      - maps: heat_rank[digit]=1..10 (1 hottest), cold_rank[digit]=1..10 (1 coldest)
      - sets: HOT, COLD, NEUTRAL (by percentile thresholds)
    """
    # trailing window winners (0 => use all)
    subset = df if window == 0 else df.iloc[:window]

    # frequency over window
    counts = Counter(d for row in subset["digits"] for d in row)
    # ensure all digits present
    for d in DIGITS:
        counts.setdefault(d, 0)

    # hotter -> colder
    hottest = sorted(DIGITS, key=lambda d: (-counts[d], d))
    coldest = list(reversed(hottest))

    heat_rank = {d: i+1 for i, d in enumerate(hottest)}   # 1..10 hottest→coldest
    cold_rank = {d: i+1 for i, d in enumerate(coldest)}   # 1..10 coldest→hottest

    # threshold sets (percent of 10 digits)
    n_hot  = max(1, round(10 * pct_hot/100))
    n_cold = max(1, round(10 * pct_cold/100))
    HOT  = set(hottest[:n_hot])
    COLD = set(coldest[:n_cold])
    NEUTRAL = set(DIGITS) - HOT - COLD

    return hottest, heat_rank, cold_rank, HOT, COLD, NEUTRAL


def due_mask(df: pd.DataFrame, due_lookback: int) -> list[set]:
    """
    Produces a per-row set of digits that are 'due' at that row:
      'due' means 'not seen in the previous N winners' (look-back).
    We mark due *at the time of the win*, using only prior rows.
    """
    prev_seen = Counter()
    history = []

    for i, row in df.iterrows():
        # build set of digits seen in prior N rows
        start = max(0, i - due_lookback)
        recent = df.iloc[start:i]
        seen_recent = set(d for r in recent['digits'] for d in r)
        # due digits are those NOT in seen_recent
        due_now = set(DIGITS) - seen_recent
        history.append(due_now)

    return history


def bucket_5_from_heat_rank(r: int) -> int:
    """1..10 hottest->coldest → bucket 1..5 by ceil(rank/2)."""
    return math.ceil(r / 2)

def bucket_5_from_cold_rank(r: int) -> int:
    """1..10 coldest->hottest → bucket 1..5 by ceil(rank/2)."""
    return math.ceil(r / 2)


def label_digit(d: str, heat_rank: dict, cold_rank: dict, HOT: set, COLD: set, due_set: set):
    """
    Returns a token:
      - Hot: 'Hk/5'  (k in 1..5, from heat_rank 1..10)
      - Cold: 'Ck/5' (k in 1..5, from cold_rank 1..10)
      - Due:  'D1/2' if due, else 'D2/2'
      - Neutral: 'N'
    Note: a digit can be both Hot and Due (or Cold and Due); we keep *both* tokens.
    """
    tokens = []
    if d in HOT:
        tokens.append(f"H{bucket_5_from_heat_rank(heat_rank[d])}/5")
    elif d in COLD:
        tokens.append(f"C{bucket_5_from_cold_rank(cold_rank[d])}/5")
    else:
        tokens.append("N")

    # Due status
    tokens.append("D1/2" if d in due_set else "D2/2")
    return tokens


def canonical_box_schematic(tokens_per_digit: list[list[str]]) -> str:
    """
    Order-invariant BOX schematic:
      - flatten tokens
      - count identical tokens
      - produce 'TokenxK + TokenxJ + ...' with alphabetical token order
    Example: ['H1/5','D1/2'], ['H3/5','D2/2'], ['C2/5','D1/2'], ['N','D2/2'], ['H1/5','D1/2']
     -> 'C2/5x1 + D1/2x3 + D2/2x2 + H1/5x2 + H3/5x1 + N x1'
    """
    flat = [t for pair in tokens_per_digit for t in pair]
    cnt = Counter(flat)
    parts = []
    for token in sorted(cnt.keys()):
        k = cnt[token]
        if token == "N":
            parts.append("N x{}".format(k))
        else:
            parts.append(f"{token}x{k}")
    return " + ".join(parts)


def per_winner_rows(df: pd.DataFrame,
                    window: int,
                    pct_hot: int,
                    pct_cold: int,
                    due_lookback: int) -> tuple[pd.DataFrame, dict]:
    """
    Builds a per-winner table with counts and schematic.
    Returns (table, meta)
    """
    # compute heat/cold ranks on the chosen trailing window relative to each row
    hottest, heat_rank, cold_rank, HOT, COLD, NEUTRAL = rank_hot_cold(df, window, pct_hot, pct_cold)
    due_history = due_mask(df, due_lookback)

    rows = []
    for i, row in df.iterrows():
        w = row["winner"]
        digits = row["digits"]
        due_set = due_history[i]

        token_pairs = [label_digit(d, heat_rank, cold_rank, HOT, COLD, due_set) for d in digits]

        # counts
        hot_hits = sum(1 for pair in token_pairs if any(t.startswith("H") for t in pair))
        cold_hits = sum(1 for pair in token_pairs if any(t.startswith("C") for t in pair))
        neutral_hits = sum(1 for pair in token_pairs if "N" in pair)
        due_hits = sum(1 for pair in token_pairs if "D1/2" in pair)

        schematic = canonical_box_schematic(token_pairs)

        rows.append({
            "idx": i+1,                  # 1-based index in chosen order
            "winner": w,
            "digits": "".join(digits),
            "hot_hits": hot_hits,
            "cold_hits": cold_hits,
            "due_hits": due_hits,
            "neutral_hits": neutral_hits,
            "schematic_box": schematic
        })

    table = pd.DataFrame(rows)

    # summary: frequency of schematics (order-invariant)
    freq = (table
            .groupby("schematic_box", as_index=False)
            .size()
            .rename(columns={"size": "count"}))
    total = len(table)
    freq["pct"] = (freq["count"] / total * 100).round(2)
    freq = freq.sort_values(["count", "schematic_box"], ascending=[False, True]).reset_index(drop=True)

    meta = {
        "hottest_to_coldest": hottest,
        "params": {
            "window": window,
            "pct_hot": pct_hot,
            "pct_cold": pct_cold,
            "due_lookback": due_lookback
        },
        "totals": {
            "winners": total
        }
    }
    return table, {"schematic_freq": freq, **meta}


def to_txt_summary(meta_and_freq: dict, table: pd.DataFrame) -> str:
    hottest = meta_and_freq["hottest_to_coldest"]
    params = meta_and_freq["params"]
    freq = meta_and_freq["schematic_freq"]
    totals = meta_and_freq["totals"]

    lines = []
    lines.append("DC5 Hot/Cold/Due Analyzer — BOX schematics")
    lines.append(f"Total winners: {totals['winners']}")
    lines.append(f"Window for hot/cold: {params['window']} | %Hot: {params['pct_hot']} | %Cold: {params['pct_cold']} | Due threshold: last {params['due_lookback']} draws")
    lines.append("Hottest→Coldest: " + ", ".join(hottest))
    lines.append("")
    lines.append("Top schematics (order-invariant):")
    for _, r in freq.iterrows():
        lines.append(f"  {r['schematic_box']}  —  {r['count']} ({r['pct']}%)")
    lines.append("")
    lines.append("Per-winner counts (first 25 rows):")
    head = table[["idx","winner","hot_hits","cold_hits","due_hits","neutral_hits"]].head(25)
    lines.append(head.to_string(index=False))
    return "\n".join(lines)


# ------------------------------
# UI
# ------------------------------

st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")

st.title("DC5 Hot/Cold/Due Analyzer — v0.6 (Run button + ranked H/C, Due buckets, BOX schematics)")

# 1) Input
st.sidebar.header("1) Input")
up = st.sidebar.file_uploader("Upload winners (.csv or .txt)", type=["csv","txt"])

# 2) File order
st.sidebar.header("2) File order")
order_mode = st.sidebar.radio("Row order in file:",
                              options=["Auto (guess)","Newest first","Oldest first"],
                              index=0,
                              help="This is the chronological order you want to analyze in.")
order_map = {"Auto (guess)":"auto","Newest first":"newest","Oldest first":"oldest"}
order_mode = order_map[order_mode]

# 3) Run controls
st.sidebar.header("3) Run controls")
run_clicked = st.sidebar.button("▶️ Run analysis (top)", type="primary")

# 4) Hot/Cold/Due params
st.sidebar.header("4) Hot/Cold/Due params")
window = int(st.sidebar.number_input("Window (trailing draws) for hot/cold ranking (0 = all)", min_value=0, max_value=2000, value=10, step=1))
pct_hot = int(st.sidebar.slider("% of digits labeled Hot (by frequency)", min_value=10, max_value=50, value=30, step=5))
pct_cold = int(st.sidebar.slider("% of digits labeled Cold (by frequency)", min_value=10, max_value=50, value=30, step=5))
due_lookback = int(st.sidebar.number_input("Due threshold: 'not seen in last N draws'", min_value=1, max_value=20, value=2, step=1))

# 5) Export control
st.sidebar.header("5) Export control")
export_prefix = st.sidebar.text_input("Export file prefix", value="dc5_analysis")

# --- main controls ---
st.subheader("Run")
run_top = st.button("▶️ Run analysis (top)", type="primary")

run_now = run_clicked or run_top

if not up:
    st.info("Upload a .csv or .txt file of winners to begin.")
    st.stop()

with st.spinner("Reading file…"):
    df = parse_uploaded(up)

st.caption(f"Parsed: **{len(df)} winners** found in the file. (We accept both 5-digit forms like `00458` and hyphenated `0-0-4-5-8`.)")

# file order
df = order_rows(df, order_mode)

# quick view: hottest → coldest for current parameters
hottest, heat_rank, cold_rank, HOT, COLD, NEUTRAL = rank_hot_cold(df, window, pct_hot, pct_cold)
st.markdown("### Hottest → coldest (quick view)")
st.code(", ".join(hottest))

# Guard against accidental re-runs:
if not run_now:
    st.warning("Press **Run analysis** to compute composition tables and schematics. (This page won’t auto-refresh.)")
    st.stop()

# ------------------------------
# Compute per-winner table + schematics
# ------------------------------
with st.spinner("Computing per-winner composition & schematics…"):
    per_tbl, meta = per_winner_rows(df, window, pct_hot, pct_cold, due_lookback)

st.markdown("### Pattern summaries (BOX, order-invariant)")
# Show counts table first rows
st.dataframe(per_tbl[["idx","winner","hot_hits","cold_hits","due_hits","neutral_hits","schematic_box"]].head(50),
             use_container_width=True, hide_index=True)

# Schematic frequencies
freq = meta["schematic_freq"]
st.markdown("#### Top schematics")
st.dataframe(freq.head(50), use_container_width=True, hide_index=True)

# ------------------------------
# Exports
# ------------------------------
st.markdown("### Export")

# CSV: per-winner table
csv_buf = io.StringIO()
per_tbl.to_csv(csv_buf, index=False)
st.download_button(
    label="Download per-winner CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=f"{export_prefix}_per_winner.csv",
    mime="text/csv",
)

# TXT: human summary
txt_data = to_txt_summary(meta, per_tbl)
st.download_button(
    label="Download TXT summary",
    data=txt_data.encode("utf-8"),
    file_name=f"{export_prefix}_summary.txt",
    mime="text/plain",
)

# Collapsible JSON summary (for debugging)
with st.expander("Summary (JSON)"):
    st.json({
        "params": meta["params"],
        "hottest_to_coldest": meta["hottest_to_coldest"],
        "totals": meta["totals"]
    })
