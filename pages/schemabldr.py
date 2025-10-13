# Mini Filter Tester + Top-40 Schemas
# ------------------------------------------------------------
# Features:
# - Upload winners (CSV or TXT). You can analyze ALL uploaded rows or only the last N (default: 10).
# - Compute Hot/Cold/Neutral tiers (frequency in trailing window) and Due ranks (not seen in last W draws).
# - Show ordered lists: Hot (hottest→), Cold (coldest→), Due (D1 most overdue→), Neutral.
# - Build combined per-digit tokens (e.g., H1&D1, C2, N&D2) and per-winner orderless schematics.
# - Create **Top 40 schemas** (most→least frequent) with counts and %.
# - Enter a 5-digit combo and check if its **schema exists** (and whether it’s in Top 40).
# - Export Top 40 as CSV/TXT.
#
# Conventions (aligned with your schema analyzer):
# - H/C tiers by frequency over a chosen trailing window; H1 = hottest, C1 = coldest.
# - Due: a digit is Due iff it did NOT appear in the last W draws; ranked by age across all prior draws
#   in the current analysis slice (D1 most overdue, then D2, ...). No "not-due" labels.
# - Neutral are digits not in Hot nor Cold for the frequency window. N has no internal rank.

import io
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

DIGITS = list("0123456789")

st.set_page_config(page_title="Mini Filter Tester + Top 40 Schemas", layout="wide")
st.title("Mini Filter Tester + Top 40 Schemas")

# ---------------------------
# Helpers
# ---------------------------

def parse_uploaded(file) -> pd.DataFrame:
    """Accept .csv or .txt of winners. Returns DataFrame with columns:
    ['winner','digits'] where digits is list[str] length 5.
    CSV: scans for any 5-digit col; else scans the full CSV text.
    TXT: scans full text; also accepts hyphenated like 0-0-4-5-8.
    """
    name = file.name.lower()

    def scan_text(text: str) -> list[str]:
        wins = []
        wins += re.findall(r'(?<!\d)(\d{5})(?!\d)', text)
        wins += ["".join(g) for g in re.findall(r'(?<!\d)(\d)-(\d)-(\d)-(\d)-(\d)(?!\d)', text)]
        return wins

    if name.endswith('.csv'):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        five_cols = [c for c in df.columns if df[c].astype(str).str.fullmatch(r"\d{5}").fillna(False).any()]
        winners = []
        if five_cols:
            for c in five_cols:
                winners += df[c].astype(str).str.extract(r"(\d{5})", expand=False).dropna().tolist()
        else:
            winners = scan_text(df.to_csv(index=False))
    else:
        text = io.TextIOWrapper(file, encoding='utf-8', errors='ignore').read()
        winners = scan_text(text)

    winners = [w for w in winners if re.fullmatch(r"\d{5}", w)]
    return pd.DataFrame({
        'winner': winners,
        'digits': [list(w) for w in winners]
    })


def window_frequencies(history_digits: List[List[str]]) -> Dict[str, int]:
    c = Counter()
    for draw in history_digits:
        c.update(draw)
    for d in DIGITS:
        c.setdefault(d, 0)
    return dict(c)


def rank_hot_cold(all_rows: List[List[str]], freq_window: int, hot_k: int, cold_k: int):
    """Return (heat_rank, cold_rank, HOT_set, COLD_set, ordered_hot_to_cold).
    Ranking by frequency in the trailing freq_window draws (or all if window=0).
    H1 = highest freq; C1 = lowest freq.
    """
    subset = all_rows if freq_window == 0 else all_rows[-freq_window:]
    counts = Counter(d for row in subset for d in row)
    for d in DIGITS:
        counts.setdefault(d, 0)
    hot_to_cold = sorted(DIGITS, key=lambda d: (-counts[d], d))
    cold_to_hot = list(reversed(hot_to_cold))

    heat_rank = {d: i+1 for i, d in enumerate(hot_to_cold)}  # 1..10
    cold_rank = {d: i+1 for i, d in enumerate(cold_to_hot)}  # 1..10

    HOT = set(hot_to_cold[:max(0, hot_k)])
    COLD = set(cold_to_hot[:max(0, cold_k)])
    return heat_rank, cold_rank, HOT, COLD, hot_to_cold


def due_ranks_by_age(all_rows: List[List[str]], due_window: int) -> Dict[str, Optional[int]]:
    """Compute per-digit Due ranks using ALL prior draws in the current slice.
    A digit is Due iff it did NOT appear in the last due_window draws.
    Age = draws since last appearance (∞ if never seen). Sort by age desc; D1 most overdue.
    Returns mapping digit -> rank (1,2,3,...) or None if not due.
    """
    # age init
    age = {d: math.inf for d in DIGITS}
    # Walk back from most recent
    for back, draw in enumerate(reversed(all_rows), start=0):  # back=0 means last draw
        for d in set(draw):
            if math.isinf(age[d]):
                age[d] = back
    # due eligibility
    due_digits = [d for d in DIGITS if age[d] >= due_window]
    ordered = sorted(due_digits, key=lambda d: (-age[d], d))
    ranks = {d: None for d in DIGITS}
    for r, d in enumerate(ordered, start=1):
        ranks[d] = r
    return ranks


def bucket_from_rank(r: int) -> int:
    """Map 1..10 to 1..5 in steps of 2 (1-2→1, 3-4→2, ..., 9-10→5)."""
    return (r + 1) // 2


def combined_token(d: str, heat_rank: Dict[str,int], cold_rank: Dict[str,int], HOT: set, COLD: set, due_rank: Optional[int]) -> str:
    # base H/C/N with ranks for H/C only
    if d in HOT:
        base = f"H{bucket_from_rank(heat_rank[d])}"
    elif d in COLD:
        base = f"C{bucket_from_rank(cold_rank[d])}"
    else:
        base = "N"
    # append Due tag if present
    return f"{base}&D{due_rank}" if due_rank is not None else base


def bag_schematic(tokens: List[str]) -> str:
    """Orderless schematic like 'H1&D1x2 + C2x1 + Nx2'"""
    cnt = Counter(tokens)
    parts = []
    for k in sorted(cnt.keys()):
        parts.append(f"{k}x{cnt[k]}")
    return " + ".join(parts)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload winners (.csv or .txt)", type=["csv","txt"])
    last_n = st.number_input("Analyze last N winners (0 = all)", min_value=0, max_value=1000, value=10, step=1)
    st.caption("Set to 10 to match your 'last 10 winners' use case. Use 0 to analyze all uploaded rows.")

    st.header("Parameters")
    freq_window = st.number_input("Frequency window for H/C ranking (draws)", min_value=1, max_value=500, value=10, step=1)
    hot_k = st.slider("How many digits are Hot?", min_value=0, max_value=10, value=4)
    cold_k = st.slider("How many digits are Cold?", min_value=0, max_value=10, value=4)
    due_window = st.number_input("Due threshold W: not seen in last W draws", min_value=1, max_value=20, value=2, step=1)

    st.header("Schema Table")
    top_k = st.slider("Top-K schemas to display", min_value=10, max_value=100, value=40, step=5)

# ---------------------------
# Main logic
# ---------------------------
if not up:
    st.info("Upload winners to begin.")
    st.stop()

with st.spinner("Reading…"):
    raw_df = parse_uploaded(up)

if len(raw_df) == 0:
    st.error("No 5-digit winners found in the upload.")
    st.stop()

# Choose the slice to analyze (last N or all)
if last_n and last_n > 0:
    df = raw_df.tail(last_n).reset_index(drop=True)
else:
    df = raw_df.copy().reset_index(drop=True)

st.caption(f"Loaded {len(df)} winners for analysis (from {len(raw_df)} uploaded).")

# Compute H/C/D/N on the chosen slice
all_rows = df['digits'].tolist()
heat_rank, cold_rank, HOT, COLD, hot_to_cold = rank_hot_cold(all_rows, freq_window=int(freq_window), hot_k=int(hot_k), cold_k=int(cold_k))
due_ranks_map = due_ranks_by_age(all_rows, due_window=int(due_window))

# Ordered lists for display
hot_list = sorted(list(HOT), key=lambda d: (heat_rank[d]))  # H1 hottest first
cold_list = sorted(list(COLD), key=lambda d: (-cold_rank[d]))  # C1 coldest first (larger cold_rank means colder)
due_list = sorted([d for d, r in due_ranks_map.items() if r is not None], key=lambda d: (due_ranks_map[d]))  # D1 first
neutral_list = [d for d in DIGITS if d not in HOT and d not in COLD]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("#### Hot (hottest →)")
    st.write(", ".join(hot_list) if hot_list else "—")
with col2:
    st.markdown("#### Cold (coldest →)")
    st.write(", ".join(cold_list) if cold_list else "—")
with col3:
    st.markdown("#### Due (D1 →)")
    if due_list:
        st.write(", ".join(f"{d}(D{due_ranks_map[d]})" for d in due_list))
    else:
        st.write("—")
with col4:
    st.markdown("#### Neutral")
    st.write(", ".join(sorted(neutral_list)) if neutral_list else "—")

# Build combined tokens & schematics for each winner in slice
rows = []
for i, row in df.iterrows():
    tokens = [
        combined_token(d,
                       heat_rank=heat_rank, cold_rank=cold_rank,
                       HOT=HOT, COLD=COLD,
                       due_rank=due_ranks_map[d])
        for d in row['digits']
    ]
    rows.append({
        'idx': i+1,
        'winner': row['winner'],
        'tokens': tokens,
        'schema': bag_schematic(tokens)
    })

schem_df = pd.DataFrame(rows)

st.markdown("### Winners → Combined Tokens → Schema")
st.dataframe(schem_df[['idx','winner','schema']], use_container_width=True, hide_index=True)

# Top-K schemas (most→least frequent) across the chosen slice
counts = Counter(schem_df['schema'])
top_schemas = counts.most_common(int(top_k))
top_df = pd.DataFrame([(s, c, round(c/len(schem_df)*100, 2)) for s, c in top_schemas],
                      columns=["Schema (orderless)", "Count", "% of Winners"])

st.markdown("### Top Schemas (Most → Least Likely)")
st.dataframe(top_df, use_container_width=True, hide_index=True)

# Exports
csv_buf = io.StringIO()
top_df.to_csv(csv_buf, index=False)
st.download_button("Download Top Schemas (CSV)", data=csv_buf.getvalue().encode('utf-8'), file_name="top_schemas.csv", mime="text/csv")

# TXT export
lines = ["Top Schemas (Most → Least Likely)\n"]
for i, r in top_df.iterrows():
    lines.append(f"{i+1:>2}. {r['Schema (orderless)']}  |  Count: {int(r['Count'])}  |  % of Winners: {r['% of Winners']:.2f}%")
st.download_button("Download Top Schemas (TXT)", data="\n".join(lines).encode('utf-8'), file_name="top_schemas.txt", mime="text/plain")

# ---------------------------
# Combo tester
# ---------------------------
st.markdown("## Combo Tester")
combo = st.text_input("Enter a 5-digit combo (e.g., 27500)", value="")

if combo:
    if not re.fullmatch(r"\d{5}", combo):
        st.error("Please enter exactly 5 digits (0–9). Leading zeros allowed, e.g., 02750.")
    else:
        # Build the 5 tokens under current mapping
        combo_digits = list(combo)
        combo_tokens = [
            combined_token(d,
                           heat_rank=heat_rank, cold_rank=cold_rank,
                           HOT=HOT, COLD=COLD,
                           due_rank=due_ranks_map[d])
            for d in combo_digits
        ]
        combo_schema = bag_schematic(combo_tokens)

        st.write("**Combo tokens:** ", ", ".join(combo_tokens))
        st.write("**Combo schema:** ", combo_schema)

        # Check against observed schemas in the current slice
        count = counts.get(combo_schema, 0)
        total = len(schem_df)
        pct = (count/total*100) if total else 0.0
        if count > 0:
            st.success(f"Schema MATCH found in data: {count}/{total} winners ({pct:.2f}%).")
        else:
            st.info("This schema did not occur in the analyzed slice.")

# ---------------------------
# Known Winner Search (Box / Straight)
# ---------------------------
st.markdown("## Known Winner Search (Box / Straight)")
known = st.text_input("Enter a known 5-digit winner to search for", value="", key="known_search")
search_box = st.checkbox("Search box (orderless match)", value=True)
search_straight = st.checkbox("Search straight (exact order)", value=False)

if known:
    if not re.fullmatch(r"[0-9]{5}", known):
        st.error("Please enter exactly 5 digits (0–9). Example: 27500 or 02750.")
    else:
        total = len(df)
        # Straight search
        if search_straight:
            straight_hits = df[df['winner'] == known].copy()
            st.write(f"**Straight matches:** {len(straight_hits)}/{total} ({(len(straight_hits)/total*100 if total else 0):.2f}%)")
            if not straight_hits.empty:
                straight_hits = straight_hits.reset_index().rename(columns={"index":"row_idx"})
                straight_hits['row_idx'] = straight_hits['row_idx'] + 1
                st.dataframe(straight_hits[['row_idx','winner']], use_container_width=True, hide_index=True)
        # Box search (orderless by digits)
        if search_box:
            target = "".join(sorted(list(known)))
            _tmp = df.copy()
            _tmp['_sorted'] = _tmp['winner'].apply(lambda s: "".join(sorted(list(s))))
            box_hits = _tmp[_tmp['_sorted'] == target].copy()
            st.write(f"**Box matches (orderless):** {len(box_hits)}/{total} ({(len(box_hits)/total*100 if total else 0):.2f}%)")
            if not box_hits.empty:
                box_hits = box_hits.reset_index().rename(columns={"index":"row_idx"})
                box_hits['row_idx'] = box_hits['row_idx'] + 1
                st.dataframe(box_hits[['row_idx','winner']], use_container_width=True, hide_index=True)

st.caption("Notes: H/C are computed from the trailing frequency window you selected; Due is 'not seen in last W draws' ranked by age (D1 most overdue). Neutral carries no rank. Schemas are orderless bags of 5 combined tokens.")
