# Filter Tester PLUS — Type Last 10 • Upload Schema List • Top 40 • Combo Search
# -----------------------------------------------------------------------------
# This is the upgraded build you asked for, meant to replace schemabldr.py.
# Key features:
# • Winners Input: choose "Type last 10" (text area) OR "Upload file" (.csv/.txt)
# • H/C/N tiers (by trailing frequency window) and Due ranks (strict: not in last W; D1 most overdue)
# • Ordered lists: Hot (hottest→), Cold (coldest→), Due (D1→), Neutral
# • Combined tokens per digit (e.g., H1&D1, C2, N&D2) → orderless schema per winner
# • Top‑K schemas (default 40) with CSV/TXT exports
# • Combo Tester: enter a 5‑digit combo → shows tokens & schema, tells if it appears in the slice and/or uploaded schema list
# • Known Winner Search: search Box (orderless digits) and/or Straight (exact order)

import io
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

DIGITS = list("0123456789")

st.set_page_config(page_title="Filter Tester PLUS", layout="wide")
st.title("Filter Tester PLUS — Type Last 10 • Upload Schema List • Top 40 • Combo Search")

# ---------------------------
# Helpers
# ---------------------------

def parse_uploaded(file) -> pd.DataFrame:
    """Accept .csv or .txt of winners. Return DataFrame with ['winner','digits'].
    CSV: scan for any 5‑digit column; else scan whole text. TXT: scan full text; also accepts 0-0-4-5-8.
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
    return pd.DataFrame({'winner': winners, 'digits': [list(w) for w in winners]})


def rank_hot_cold(all_rows: List[List[str]], freq_window: int, hot_k: int, cold_k: int):
    """Return (heat_rank, cold_rank, HOT_set, COLD_set, hot_to_cold_order).
    Ranking by frequency in trailing freq_window draws (or all if window=0).
    H1 = hottest; C1 = coldest.
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
    """Due iff NOT seen in the last due_window draws; rank by age across all prior draws in slice.
    Age = draws since last appearance (∞ if never seen). Sort by age desc; D1 most overdue.
    """
    age = {d: math.inf for d in DIGITS}
    for back, draw in enumerate(reversed(all_rows), start=0):
        for d in set(draw):
            if math.isinf(age[d]):
                age[d] = back
    due_digits = [d for d in DIGITS if age[d] >= due_window]
    ordered = sorted(due_digits, key=lambda d: (-age[d], d))
    ranks = {d: None for d in DIGITS}
    for r, d in enumerate(ordered, start=1):
        ranks[d] = r
    return ranks


def bucket_from_rank(r: int) -> int:
    """Map heat/cold rank 1..10 → bucket 1..5 (1-2→1, 3-4→2, ..., 9-10→5)."""
    return (r + 1) // 2


def combined_token(d: str, heat_rank: Dict[str,int], cold_rank: Dict[str,int], HOT: set, COLD: set, due_rank: Optional[int]) -> str:
    # Base H/C/N with ranks for H/C only
    if d in HOT:
        base = f"H{bucket_from_rank(heat_rank[d])}"
    elif d in COLD:
        base = f"C{bucket_from_rank(cold_rank[d])}"
    else:
        base = "N"
    return f"{base}&D{due_rank}" if due_rank is not None else base


def bag_schematic(tokens: List[str]) -> str:
    cnt = Counter(tokens)
    parts = [f"{k}x{cnt[k]}" for k in sorted(cnt.keys())]
    return " + ".join(parts)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Winners Input")
    input_mode = st.radio("Provide winners via…", options=["Type last 10", "Upload file"], index=0)

    typed_block = None
    up = None

    if input_mode == "Type last 10":
        typed_block = st.text_area("Enter your last 10 winners (one per line)", height=200, help="Enter up to 10 lines of 5 digits each, e.g., 27500. Extra lines are ignored.")
        st.caption("We’ll analyze exactly the last 10 you type (or fewer if <10). Newest last is fine.")
    else:
        up = st.file_uploader("Upload winners (.csv or .txt)", type=["csv","txt"]) 
        last_n = st.number_input("Analyze last N winners (0 = all)", min_value=0, max_value=1000, value=10, step=1)

    st.header("Parameters")
    freq_window = st.number_input("Frequency window for H/C ranking (draws)", min_value=1, max_value=500, value=10, step=1)
    hot_k = st.slider("How many digits are Hot?", min_value=0, max_value=10, value=4)
    cold_k = st.slider("How many digits are Cold?", min_value=0, max_value=10, value=4)
    due_window = st.number_input("Due threshold W: not seen in last W draws", min_value=1, max_value=20, value=2, step=1)

    st.header("Schema List (optional)")
    external_schema = st.file_uploader("Upload an existing schema list (CSV/TXT)", type=["csv","txt"], key="ext_schema")

    st.header("Top‑K Schemas")
    top_k = st.slider("Top‑K to display", min_value=10, max_value=100, value=40, step=5)

# ---------------------------
# Load winners
# ---------------------------
if input_mode == "Type last 10":
    lines = [l.strip() for l in (typed_block or "").splitlines() if l.strip()]
    winners = [l for l in lines if re.fullmatch(r"\d{5}", l)]
    winners = winners[-10:]  # take last 10 typed
    if not winners:
        st.info("Type at least one 5‑digit winner to begin.")
        st.stop()
    raw_df = pd.DataFrame({'winner': winners, 'digits': [list(w) for w in winners]})
else:
    if not up:
        st.info("Upload winners to begin.")
        st.stop()
    raw_df = parse_uploaded(up)
    if len(raw_df) == 0:
        st.error("No 5‑digit winners found in the upload.")
        st.stop()
    if 'last_n' in locals() and last_n and last_n > 0:
        raw_df = raw_df.tail(int(last_n)).reset_index(drop=True)

st.caption(f"Loaded {len(raw_df)} winners for analysis.")

# ---------------------------
# Compute H/C/D/N and schematics
# ---------------------------
all_rows = raw_df['digits'].tolist()
heat_rank, cold_rank, HOT, COLD, hot_to_cold = rank_hot_cold(all_rows, freq_window=int(freq_window), hot_k=int(hot_k), cold_k=int(cold_k))
due_ranks_map = due_ranks_by_age(all_rows, due_window=int(due_window))

# Ordered lists for display
hot_list = sorted(list(HOT), key=lambda d: (heat_rank[d]))  # H1 hottest first
cold_list = sorted(list(COLD), key=lambda d: (-cold_rank[d]))  # C1 coldest first
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

# Build combined tokens & schematics per winner
rows = []
for i, row in raw_df.iterrows():
    tokens = [
        combined_token(d, heat_rank=heat_rank, cold_rank=cold_rank, HOT=HOT, COLD=COLD, due_rank=due_ranks_map[d])
        for d in row['digits']
    ]
    rows.append({'idx': i+1, 'winner': row['winner'], 'schema': bag_schematic(tokens)})

schem_df = pd.DataFrame(rows)

st.markdown("### Winners → Schema")
st.dataframe(schem_df, use_container_width=True, hide_index=True)

# Top‑K schemas (most→least likely)
counts = Counter(schem_df['schema'])
top_schemas = counts.most_common(int(top_k))
top_df = pd.DataFrame([(s, c, round(c/len(schem_df)*100, 2)) for s, c in top_schemas],
                      columns=["Schema (orderless)", "Count", "% of Winners"])

st.markdown("### Top Schemas")
st.dataframe(top_df, use_container_width=True, hide_index=True)
# ---------------------------
# ≥70% Pass-Rate Filters (from your 200-winner study)
# ---------------------------
st.markdown("## ≥70% Pass-Rate Filters")

def _base_class(t: str) -> str:
    b = t.split("&D")[0]
    if b.startswith("H"): return "H"
    if b.startswith("C"): return "C"
    return "N"

def _due_rank(t: str):
    m = re.search(r"&D(\d+)", t)
    return int(m.group(1)) if m else None

def _is_due(t: str) -> bool:
    return "&D" in t

# Filters (defaults ON, in the order we validated ≥70% on your dataset)
colf1, colf2 = st.columns(2)
with colf1:
    f_cold_any = st.checkbox("Contains ≥1 Cold (any C*)", value=True)
    f_neutral_any = st.checkbox("Contains ≥1 Neutral (N or N&Dx)", value=True)
    f_due_any = st.checkbox("Contains ≥1 Due (any &Dx)", value=True)
    f_c1 = st.checkbox("Contains C1 (with/without Due)", value=True)
    f_hot_any = st.checkbox("Contains ≥1 Hot (any H*)", value=True)
with colf2:
    f_cold_le2 = st.checkbox("Cold ≤ 2", value=True)
    f_neutral_plain = st.checkbox("Contains ≥1 Neutral (plain N)", value=True)
    f_h1 = st.checkbox("Contains H1 (with/without Due)", value=True)
    f_d1_or_d2 = st.checkbox("Contains D1 or D2", value=True)

selected_filters = [
    ("C_any", f_cold_any),
    ("N_any", f_neutral_any),
    ("D_any", f_due_any),
    ("C1", f_c1),
    ("H_any", f_hot_any),
    ("C_le2", f_cold_le2),
    ("N_plain", f_neutral_plain),
    ("H1", f_h1),
    ("D1_or_D2", f_d1_or_d2),
]
active = [key for key, on in selected_filters if on]

# Apply to current slice
survivor_rows = []
for _, r in schem_df.iterrows():
    toks = r['tokens']
    nC = sum(1 for t in toks if _base_class(t) == "C")
    nH = sum(1 for t in toks if _base_class(t) == "H")
    nN = sum(1 for t in toks if _base_class(t) == "N")
    conds = {
        "C_any": (nC >= 1),
        "N_any": (nN >= 1),
        "D_any": any(_is_due(t) for t in toks),
        "C1": any(t.startswith("C1") for t in toks),
        "H_any": (nH >= 1),
        "C_le2": (nC <= 2),
        "N_plain": any((_base_class(t) == "N" and not _is_due(t)) for t in toks),
        "H1": any(t.startswith("H1") for t in toks),
        "D1_or_D2": any((_is_due(t) and (_due_rank(t) in (1, 2))) for t in toks),
    }
    if all(conds[k] for k in active):
        survivor_rows.append(r)

survivor_count = len(survivor_rows)

a, b = st.columns([1, 3])
with a:
    st.metric("Survivors (pass all checked)", survivor_count)
with b:
    if survivor_count:
        st.caption("Preview of survivors")
        st.dataframe(pd.DataFrame(survivor_rows)[['idx', 'winner', 'schema']], use_container_width=True, hide_index=True)
    else:
        st.info("No winners pass the currently selected ≥70% filters.")

# External schema list (optional) — normalize into a set of strings to compare
external_schemas_set = set()
if external_schema is not None:
    try:
        if external_schema.name.lower().endswith('.csv'):
            ext_df = pd.read_csv(external_schema, dtype=str, keep_default_na=False)
            col = None
            for cand in ["Schema (orderless)", "schema", "schematic_box", ext_df.columns[0]]:
                if cand in ext_df.columns:
                    col = cand
                    break
            if col is not None:
                external_schemas_set = set(ext_df[col].astype(str).str.strip())
        else:
            lines = io.TextIOWrapper(external_schema, encoding='utf-8', errors='ignore').read().splitlines()
            external_schemas_set = set(l.strip() for l in lines if l.strip())
    except Exception as e:
        st.warning(f"Could not parse external schema file: {e}")

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
# Combo Tester (schema presence)
# ---------------------------
st.markdown("## Combo Tester")
combo = st.text_input("Enter a 5‑digit combo (e.g., 27500)", value="")

if combo:
    if not re.fullmatch(r"\d{5}", combo):
        st.error("Please enter exactly 5 digits (0–9). Leading zeros allowed, e.g., 02750.")
    else:
        digits = list(combo)
        combo_tokens = [combined_token(d, heat_rank=heat_rank, cold_rank=cold_rank, HOT=HOT, COLD=COLD, due_rank=due_ranks_map[d]) for d in digits]
        combo_schema = bag_schematic(combo_tokens)

        st.write("**Combo tokens:** ", ", ".join(combo_tokens))
        st.write("**Combo schema:** ", combo_schema)

        # Check within this session's analyzed slice
        count = counts.get(combo_schema, 0)
        total = len(schem_df)
        pct = (count/total*100) if total else 0.0
        if count > 0:
            st.success(f"Schema MATCH in current data: {count}/{total} winners ({pct:.2f}%).")
        else:
            st.info("This schema did not occur in the analyzed slice.")

        # Also check external schema list if provided
        if external_schemas_set:
            if combo_schema in external_schemas_set:
                st.success("Schema also FOUND in your uploaded schema list.")
            else:
                st.warning("Schema NOT found in your uploaded schema list.")

# ---------------------------
# Known Winner Search (Box / Straight)
# ---------------------------
st.markdown("## Known Winner Search (Box / Straight)")
known = st.text_input("Enter a known 5‑digit winner to search for", value="", key="known_search")
search_box = st.checkbox("Search box (orderless match)", value=True)
search_straight = st.checkbox("Search straight (exact order)", value=False)

# We search inside the currently analyzed winners (raw_df)
if known:
    if not re.fullmatch(r"[0-9]{5}", known):
        st.error("Please enter exactly 5 digits (0–9). Example: 27500 or 02750.")
    else:
        total = len(raw_df)
        if search_straight:
            straight_hits = raw_df[raw_df['winner'] == known].copy()
            st.write(f"**Straight matches:** {len(straight_hits)}/{total} ({(len(straight_hits)/total*100 if total else 0):.2f}%)")
            if not straight_hits.empty:
                straight_hits = straight_hits.reset_index().rename(columns={"index":"row_idx"})
                straight_hits['row_idx'] = straight_hits['row_idx'] + 1
                st.dataframe(straight_hits[['row_idx','winner']], use_container_width=True, hide_index=True)
        if search_box:
            target = "".join(sorted(list(known)))
            _tmp = raw_df.copy()
            _tmp['_sorted'] = _tmp['winner'].apply(lambda s: "".join(sorted(list(s))))
            box_hits = _tmp[_tmp['_sorted'] == target].copy()
            st.write(f"**Box matches (orderless):** {len(box_hits)}/{total} ({(len(box_hits)/total*100 if total else 0):.2f}%)")
            if not box_hits.empty:
                box_hits = box_hits.reset_index().rename(columns={"index":"row_idx"})
                box_hits['row_idx'] = box_hits['row_idx'] + 1
                st.dataframe(box_hits[['row_idx','winner']], use_container_width=True, hide_index=True)

st.caption("Notes: H/C from trailing frequency window; Due = not seen in last W draws (ranked D1 most overdue). Neutral has no rank. Schemas are orderless bags of 5 combined tokens.")
