# DC5 Hot/Cold × Due (Combined) — Updated from your current analyzer
# Changes:
# - Strict Due logic (no "not-due" labels). A digit is Due iff it did NOT appear in last N draws.
# - Due is RANKED by age at each row: D1 (most overdue), D2, D3, …
# - Per-digit labels are now **combined**: Hk&Dm, Ck&Dm, N&Dm, or just Hk / Ck / N if not due.
# - Schematic is built from these combined labels (orderless, box-style).
# - Counts updated to respect combined labels (due_hits counts only digits with &D*).
# - Added Auto‑Insights (≥ threshold) surfacing only strong (≥70% by default) patterns.
# - Preserves your uploader, ordering, params, exports, and quick-view sections.

import io
import re
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

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
    # auto guess: treat input as newest-first, keep order
    return df.reset_index(drop=True)


def window_frequencies(history_digits: List[List[str]]) -> Dict[str, int]:
    c = Counter()
    for draw in history_digits:
        c.update(draw)
    for d in DIGITS:
        c.setdefault(d, 0)
    return dict(c)


def rank_hot_cold(df: pd.DataFrame, window: int, pct_hot: int, pct_cold: int):
    """
    Returns:
      - hottest_to_coldest: list of digits from hottest -> coldest (by freq in trailing window)
      - maps: heat_rank[digit]=1..10 (1 hottest), cold_rank[digit]=1..10 (1 coldest)
      - sets: HOT, COLD, NEUTRAL (by percentile thresholds)
    """
    subset = df if window == 0 else df.iloc[:window]
    counts = Counter(d for row in subset["digits"] for d in row)
    for d in DIGITS:
        counts.setdefault(d, 0)

    hottest = sorted(DIGITS, key=lambda d: (-counts[d], d))  # hot→cold
    coldest = list(reversed(hottest))                        # cold→hot

    heat_rank = {d: i+1 for i, d in enumerate(hottest)}     # 1..10
    cold_rank = {d: i+1 for i, d in enumerate(coldest)}     # 1..10

    n_hot  = max(1, round(10 * pct_hot/100))
    n_cold = max(1, round(10 * pct_cold/100))
    HOT  = set(hottest[:n_hot])
    COLD = set(coldest[:n_cold])
    NEUTRAL = set(DIGITS) - HOT - COLD

    return hottest, heat_rank, cold_rank, HOT, COLD, NEUTRAL


# ------------------------------
# Strict Due logic with ranking
# ------------------------------

def due_ranks_at_each_row(df: pd.DataFrame, due_window: int) -> List[Dict[str, Optional[int]]]:
    """
    For each row i, compute Due ranks from *all* prior rows [0..i-1]:
      - age[d] = draws since last seen (∞ if never seen)
      - eligible if age >= due_window
      - order by age desc (older = more overdue), tie-break by digit asc
      - assign ranks 1..k => D1 most overdue
    Returns a list (len = len(df)) of dicts: digit -> rank or None.
    """
    out: List[Dict[str, Optional[int]]] = []
    for i in range(len(df)):
        past = df.loc[:i-1, "digits"].tolist()
        # compute ages
        age = {d: math.inf for d in DIGITS}
        # walk back from most recent
        back = 0
        for draw in reversed(past):
            # mark first time we see each digit
            for d in set(draw):
                if math.isinf(age[d]):
                    age[d] = back
            back += 1
        # eligible
        eligible = [d for d in DIGITS if age[d] >= due_window]
        ordered = sorted(eligible, key=lambda d: (-age[d], d))
        ranks = {d: None for d in DIGITS}
        for r, d in enumerate(ordered, start=1):
            ranks[d] = r
        out.append(ranks)
    return out


def bucket_5_from_heat_rank(r: int) -> int:
    return math.ceil(r / 2)


def bucket_5_from_cold_rank(r: int) -> int:
    return math.ceil(r / 2)


def combined_label_for_digit(d: str,
                             heat_label_sets: Tuple[Dict[str, int], Dict[str, int], set, set],
                             due_rank_map: Dict[str, Optional[int]]) -> str:
    """Return combined label per digit: Hk&Dm, Ck&Dm, N&Dm, or just Hk/Ck/N if not due."""
    heat_rank, cold_rank, HOT, COLD = heat_label_sets

    # Base H/C/N tier + rank
    if d in HOT:
        base = f"H{bucket_5_from_heat_rank(heat_rank[d])}"
    elif d in COLD:
        base = f"C{bucket_5_from_cold_rank(cold_rank[d])}"
    else:
        base = "N"

    # Due tag if applicable
    r = due_rank_map.get(d)
    if r is None:
        return base
    return f"{base}&D{r}"


def canonical_box_schematic_from_combined(labels: List[str]) -> str:
    """Order-invariant BOX schematic from 5 combined labels.
    Example 5 labels → 'H1&D1x2 + C2x1 + Nx2'
    """
    cnt = Counter(labels)
    parts = []
    for token in sorted(cnt.keys()):
        parts.append(f"{token}x{cnt[token]}")
    return " + ".join(parts)


def per_winner_rows(df: pd.DataFrame,
                    window: int,
                    pct_hot: int,
                    pct_cold: int,
                    due_lookback: int) -> tuple[pd.DataFrame, dict]:
    """
    Builds a per-winner table with counts and combined schematic.
    Returns (table, meta)
    """
    # global H/C/N based on the chosen trailing window
    hottest, heat_rank, cold_rank, HOT, COLD, NEUTRAL = rank_hot_cold(df, window, pct_hot, pct_cold)

    # strict Due ranks for each row
    due_ranks_all_rows = due_ranks_at_each_row(df, due_lookback)

    rows = []
    for i, row in df.iterrows():
        winner_digits = row["digits"]
        due_map = due_ranks_all_rows[i]

        # build 5 combined labels
        labels = [
            combined_label_for_digit(d, (heat_rank, cold_rank, HOT, COLD), due_map)
            for d in winner_digits
        ]

        # counts (orderless)
        hot_hits = sum(1 for lb in labels if lb.startswith("H"))
        cold_hits = sum(1 for lb in labels if lb.startswith("C"))
        neutral_hits = sum(1 for lb in labels if lb == "N" or lb.startswith("N&"))
        due_hits = sum(1 for lb in labels if "&D" in lb)

        schematic = canonical_box_schematic_from_combined(labels)

        rows.append({
            "idx": i+1,
            "winner": row["winner"],
            "digits": "".join(winner_digits),
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
        },
        "schematic_freq": freq
    }
    return table, meta


def to_txt_summary(meta_and_freq: dict, table: pd.DataFrame) -> str:
    hottest = meta_and_freq["hottest_to_coldest"]
    params = meta_and_freq["params"]
    freq = meta_and_freq["schematic_freq"]
    totals = meta_and_freq["totals"]

    lines = []
    lines.append("DC5 Hot/Cold × Due Analyzer — BOX schematics (combined labels)")
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

st.set_page_config(page_title="DC5 Hot/Cold × Due Analyzer (Combined)", layout="wide")

st.title("DC5 Hot/Cold × Due Analyzer — v0.7 (Combined labels + Auto‑Insights)")

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

# 5) Insights
st.sidebar.header("5) Insights")
min_support = int(st.sidebar.slider("Insights threshold (≥ this %)", min_value=50, max_value=100, value=70, step=1))

# 5.5) Display
st.sidebar.header("5.5) Display")
rows_to_show = int(st.sidebar.number_input("Max rows to display (0 = all)", min_value=0, max_value=5000, value=200, step=50))

# 6) Export control
st.sidebar.header("6) Export control")
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
    st.warning("Press **Run analysis** to compute composition tables, schematics, and insights. (No auto-refresh.)")
    st.stop()

# ------------------------------
# Compute per-winner table + schematics
# ------------------------------
with st.spinner("Computing per-winner composition & schematics…"):
    per_tbl, meta = per_winner_rows(df, window, pct_hot, pct_cold, due_lookback)

st.markdown("### Pattern summaries (BOX, order-invariant)")
_cols = ["idx","winner","hot_hits","cold_hits","due_hits","neutral_hits","schematic_box"]
display_df = per_tbl[_cols].copy()
if rows_to_show > 0:
    display_df = display_df.head(rows_to_show)
st.dataframe(display_df, use_container_width=True, hide_index=True)

freq = meta["schematic_freq"]
st.markdown("#### Top schematics")
freq_display = freq.copy()
if rows_to_show > 0:
    freq_display = freq_display.head(rows_to_show)
st.dataframe(freq_display, use_container_width=True, hide_index=True)

# ------------------------------
# Auto‑Insights (≥ threshold)
# ------------------------------

st.markdown("### Auto‑Insights (≥ threshold)")

# helper: composition counts H/C/N ignoring &D

def composition_counts_from_labels(labels: List[str]) -> Tuple[int,int,int]:
    H = sum(1 for lb in labels if lb.startswith("H"))
    C = sum(1 for lb in labels if lb.startswith("C"))
    N = sum(1 for lb in labels if lb == "N" or lb.startswith("N&"))
    return H, C, N

rows_for_insights = []
for _, r in per_tbl.iterrows():
    # explode 5 combined tokens back from schematic
    # since schematic is deterministic, we can reconstruct labels by repeated tokens
    parts = []
    for piece in r["schematic_box"].split(" + "):
        token, times = piece.split("x")
        parts.extend([token] * int(times))
    rows_for_insights.append(parts)

insights = []

# 1) Composition patterns like 2H + 1C (orderless)
comp_counter = Counter()
for lbls in rows_for_insights:
    h,c,n = composition_counts_from_labels(lbls)
    comp_counter[(h,c,n)] += 1

total_rows = len(rows_for_insights)
for (h,c,n), cnt in comp_counter.most_common():
    pct = cnt / total_rows * 100 if total_rows else 0
    if pct >= min_support:
        parts = []
        if h: parts.append(f"{h} Hot")
        if c: parts.append(f"{c} Cold")
        if n: parts.append(f"{n} Neutral")
        desc = " + ".join(parts) if parts else "0 Hot/Cold/Neutral"
        insights.append(f"**{pct:.0f}%** of winners contain **{desc}**.")

# 2) Within top composition, example tier detail: H1 present + C2 present
if comp_counter:
    top_comp, _ = comp_counter.most_common(1)[0]
    def has_tiers(lbls: List[str]) -> bool:
        h,c,n = composition_counts_from_labels(lbls)
        if (h,c,n) != top_comp:
            return False
        return any(lb.startswith("H1") for lb in lbls) and any(lb.startswith("C2") for lb in lbls)
    cnt = sum(1 for lbls in rows_for_insights if has_tiers(lbls))
    pct = cnt / total_rows * 100 if total_rows else 0
    if pct >= min_support:
        insights.append(f"Within that composition, **{pct:.0f}%** include **H1** and **C2**.")

# 3) Cold & Due overlap — ≥1 digit both Cold & Due
cnt = sum(1 for lbls in rows_for_insights if any(lb.startswith("C") and "&D" in lb for lb in lbls))
pct = cnt / total_rows * 100 if total_rows else 0
if pct >= min_support:
    insights.append(f"**{pct:.0f}%** of winners have **≥1 digit that is both Cold & Due**.")

# 4) Due presence levels: ≥1 Due, ≥2 Due, contains D1
for need, name in [(1, "≥1 Due"), (2, "≥2 Due")]:
    cnt = sum(1 for lbls in rows_for_insights if sum(1 for lb in lbls if "&D" in lb) >= need)
    pct = cnt / total_rows * 100 if total_rows else 0
    if pct >= min_support:
        insights.append(f"**{pct:.0f}%** of winners contain **{name} digit(s)**.")

cnt = sum(1 for lbls in rows_for_insights if any("&D1" in lb for lb in lbls))
pct = cnt / total_rows * 100 if total_rows else 0
if pct >= min_support:
    insights.append(f"**{pct:.0f}%** include at least one **D1 (most overdue)** digit.")

# 5) Hot & Due intersection
cnt = sum(1 for lbls in rows_for_insights if any(lb.startswith("H") and "&D" in lb for lb in lbls))
pct = cnt / total_rows * 100 if total_rows else 0
if pct >= min_support:
    insights.append(f"**{pct:.0f}%** include **≥1 Hot & Due** digit.")

# 6) Absence patterns
cnt = sum(1 for lbls in rows_for_insights if sum(1 for lb in lbls if lb.startswith("C")) <= 1)
pct = cnt / total_rows * 100 if total_rows else 0
if pct >= min_support:
    insights.append(f"**{pct:.0f}%** have **no more than 1 Cold** digit.")

cnt = sum(1 for lbls in rows_for_insights if all(not (lb == "N" or lb.startswith("N&")) for lb in lbls))
pct = cnt / total_rows * 100 if total_rows else 0
if pct >= min_support:
    insights.append(f"**{pct:.0f}%** of winners have **0 Neutral** digits.")

if insights:
    for line in insights:
        st.markdown("- " + line)
else:
    st.info("No insights met the current threshold. Try lowering the slider or adjusting windows.")

# ------------------------------
# Exports
# ------------------------------

st.markdown("### Export")

csv_buf = io.StringIO()
per_tbl.to_csv(csv_buf, index=False)
st.download_button(
    label="Download per-winner CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=f"{export_prefix}_per_winner.csv",
    mime="text/csv",
)

txt_data = to_txt_summary(meta, per_tbl)
st.download_button(
    label="Download TXT summary",
    data=txt_data.encode("utf-8"),
    file_name=f"{export_prefix}_summary.txt",
    mime="text/plain",
)

with st.expander("Summary (JSON)"):
    st.json({
        "params": meta["params"],
        "hottest_to_coldest": meta["hottest_to_coldest"],
        "totals": meta["totals"]
    })
