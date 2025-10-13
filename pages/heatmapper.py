# Heatmap A–J Generator (Streamlit)
# ------------------------------------------------------------
# Purpose
# • You type/upload your last N winners (default N=10) — we build the pre‑draw heatmap.
# • Map ranks (hottest→coldest) to letters **A..J** (A=rank1 … J=rank10).
# • Generate **box‑unique** combos from letter schemas (e.g., `A B E E J`, `A A E F J`).
# • Options: drift (±1 rank), adjacency anchors, Due overlay (≥1 Due, Due on I/J, D1/D2), top‑K with +100 bump.
# • Export CSV of all generated candidates with scores.
#
# Quick start:
#   pip install streamlit pandas
#   streamlit run heatmap_aj_generator.py

import io
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJ")  # A..J

st.set_page_config(page_title="Heatmap A–J Generator", layout="wide")
st.title("Heatmap A–J Generator (Box)")

# ==========================
# Helpers
# ==========================

def parse_winners_text(txt: str) -> List[str]:
    lines = [l.strip() for l in (txt or "").splitlines() if l.strip()]
    wins = [l for l in lines if re.fullmatch(r"\d{5}", l)]
    return wins


def parse_winners_upload(file) -> List[str]:
    name = file.name.lower()
    winners: List[str] = []
    def scan_text(text: str) -> List[str]:
        out = []
        out += re.findall(r"(?<!\d)(\d{5})(?!\d)", text)
        out += ["".join(g) for g in re.findall(r"(?<!\d)(\d)-(\d)-(\d)-(\d)-(\d)(?!\d)", text)]
        return out
    if name.endswith('.csv'):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        # try to find a column with 5‑digit strings
        five_cols = [c for c in df.columns if df[c].astype(str).str.fullmatch(r"\d{5}").fillna(False).any()]
        if five_cols:
            col = five_cols[0]
            winners = df[col].astype(str).str.extract(r"(\d{5})", expand=False).dropna().tolist()
        else:
            winners = scan_text(df.to_csv(index=False))
    else:
        text = io.TextIOWrapper(file, encoding='utf-8', errors='ignore').read()
        winners = scan_text(text)
    winners = [w for w in winners if re.fullmatch(r"\d{5}", w)]
    return winners


def rank_hot_cold(last_rows: List[List[str]]) -> Tuple[List[str], Dict[str, int]]:
    """Return order (hottest→coldest) and counts for the given slice of rows."""
    counts = Counter(d for row in last_rows for d in row)
    for d in DIGITS:
        counts.setdefault(d, 0)
    order = sorted(DIGITS, key=lambda d: (-counts[d], d))  # tie‑break by digit
    return order, counts


def build_letter_map(order: List[str]) -> Dict[str, str]:
    """Map letters A..J to the digit currently at rank 1..10."""
    return {LETTERS[i]: order[i] for i in range(10)}


def neighbor_letters(letter: str, span: int = 1) -> List[str]:
    idx = LETTERS.index(letter)
    lo = max(0, idx - span)
    hi = min(9, idx + span)
    return LETTERS[lo:hi+1]


def due_flags_from_history(all_rows: List[List[str]], due_window: int) -> Dict[str, bool]:
    """Digit is Due if NOT seen in last `due_window` draws."""
    recent = all_rows[-due_window:] if due_window > 0 else []
    seen = set(d for row in recent for d in row)
    return {d: (d not in seen) for d in DIGITS}


def d1_d2_from_history(all_rows: List[List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """Return the most/second‑most overdue digits (D1, D2) by age since last seen (across all_rows)."""
    age = {d: None for d in DIGITS}
    # Walk backwards to find first occurrence age
    for back, row in enumerate(reversed(all_rows), start=0):
        s = set(row)
        for d in DIGITS:
            if age[d] is None and d in s:
                age[d] = back
    # None => never seen in provided history; treat as very large age
    items = [(d, (9999 if age[d] is None else age[d])) for d in DIGITS]
    items.sort(key=lambda x: (-x[1], x[0]))
    d1 = items[0][0] if items else None
    d2 = items[1][0] if len(items) > 1 else None
    return d1, d2


def build_anchors_from_order(order: List[str], top_pairs: int, top_trips: int) -> List[List[str]]:
    anchors: List[List[str]] = []
    for i in range(9):
        anchors.append([order[i], order[i+1]])
    for i in range(8):
        anchors.append([order[i], order[i+1], order[i+2]])
    # Keep first K pairs and first K trips, then merge preserving order
    pairs = [[order[i], order[i+1]] for i in range(9)][:max(0, top_pairs)]
    trips = [[order[i], order[i+1], order[i+2]] for i in range(8)][:max(0, top_trips)]
    return pairs + trips


def box_key(combo: str) -> str:
    return ''.join(sorted(combo))


def generate_from_schema(
    schema_letters: List[str],
    letter_to_digits: Dict[str, List[str]],
    must_hit_any_anchor: Optional[List[List[str]]],
    due_on_cold_letters: Optional[List[str]],
    d1: Optional[str], d2: Optional[str],
    is_due: Dict[str, bool],
    prefer_n_double: bool,
    cap: int = 50000,
) -> List[Tuple[str, int]]:
    """Expand a letter schema into concrete combos and score them.
    Scoring:
      +2 if any anchor subset is contained in digits
      +2 if a **Due** digit appears from any `due_on_cold_letters` pool (e.g., I/J)
      +1 if D1 present, +1 if D2 present
      +1 if schema's double is realized; +1 extra if the double is from neutral letters (E/F) or the doubled digit is Due
    Returns list of (combo, score).
    """
    from collections import Counter

    neutral_letters = {"E", "F"}
    pools: List[List[str]] = [letter_to_digits.get(L, []) for L in schema_letters]

    res: List[Tuple[str, int]] = []
    seen_box = set()

    def score_combo(ds: List[str]) -> int:
        s = 0
        ds_set = set(ds)
        # Anchors
        if must_hit_any_anchor:
            for a in must_hit_any_anchor:
                if set(a).issubset(ds_set):
                    s += 2
                    break
        # Due on cold letters (e.g., I/J) — only if the digit is Due
        if due_on_cold_letters:
            hit = False
            for L in due_on_cold_letters:
                pool = set(letter_to_digits.get(L, []))
                if any((d in pool) and is_due.get(d, False) for d in ds):
                    hit = True
                    break
            if hit:
                s += 2
        # D1/D2 presence
        if d1 and d1 in ds_set:
            s += 1
        if d2 and d2 in ds_set:
            s += 1
        # Double bonus
        cnts = Counter(ds)
        for L in set(schema_letters):
            if schema_letters.count(L) >= 2:
                pool = set(letter_to_digits.get(L, []))
                doubled = [x for x, c in cnts.items() if c >= 2]
                if any(x in pool for x in doubled):
                    s += 1
                    if (L in neutral_letters) or any(is_due.get(x, False) for x in doubled):
                        s += 1
        return s

    # Backtracking (box-unique by sorted key)
    def backtrack(i: int, cur: List[str]):
        if len(res) >= cap:
            return
        if i == len(pools):
            if len(cur) != 5:
                return
            key = box_key(''.join(cur))
            if key in seen_box:
                return
            seen_box.add(key)
            res.append((''.join(cur), score_combo(cur)))
            return
        for d in pools[i]:
            cur.append(d)
            backtrack(i + 1, cur)
            cur.pop()

    backtrack(0, [])
    return res

# ==========================
# Sidebar — inputs & params
# ==========================
with st.sidebar:
    st.header("Winners Input")
    input_mode = st.radio("Provide winners via…", ["Type last N", "Upload file"], index=0)
    N_default = 10
    if input_mode == "Type last N":
        N = st.number_input("N (draws for heatmap)", 5, 50, N_default, 1)
        winners_text = st.text_area("Enter your last N winners (one per line)", height=220,
                                    placeholder="e.g.\n12345\n98765\n…")
        winners = parse_winners_text(winners_text)
        winners = winners[-int(N):]
    else:
        up = st.file_uploader("Upload winners (.csv/.txt)", type=["csv","txt"])
        N = st.number_input("N (draws for heatmap)", 5, 50, N_default, 1)
        winners = parse_winners_upload(up) if up else []
        winners = winners[-int(N):]

    st.divider()
    st.header("Generation Params")
    drift_on = st.checkbox("Allow ±1 rank drift per letter", value=True,
                           help="A can draw from {A,B}; E from {D,E,F}; J from {I,J}.")
    anchors_on = st.checkbox("Require at least one heat‑order anchor (adjacent pair/triplet)", value=True)
    top_pairs = st.slider("Top‑K adjacent PAIRS to allow as anchors", 0, 9, 4)
    top_trips = st.slider("Top‑K adjacent TRIPLETS to allow as anchors", 0, 8, 2)

    st.divider()
    st.header("Due Overlay")
    due_window = st.number_input("Due threshold W (not seen in last W draws)", 1, 10, 2, 1)
    require_due_any = st.checkbox("Require ≥1 Due digit", value=False)
    prefer_cold_due = st.checkbox("Prefer Due on cold tail (I/J)", value=True)
    require_d1d2 = st.checkbox("Require D1 or D2 present", value=False)

    st.divider()
    st.header("Schemas & Output")
    use_schema1 = st.checkbox("Use schema A B E E J (Q2 double)", value=True)
    use_schema2 = st.checkbox("Use schema A A E F J (no double)", value=True)
    top_k = st.number_input("Show Top‑K now", 50, 1000, 100, 50)

# ==========================
# Validate input & compute map
# ==========================
if len(winners) < max(5, int(N)):
    st.info("Enter/upload at least N winners to build the heatmap.")
    st.stop()

rows = [list(w) for w in winners]
order, counts = rank_hot_cold(rows)
letter_map = build_letter_map(order)  # A..J -> digit

st.markdown("### Current A–J map (hottest → coldest)")
map_str = " | ".join([f"{LETTERS[i]}={order[i]}" for i in range(10)])
st.write(map_str)

# Drift‑aware candidate digits per letter
letter_to_digits: Dict[str, List[str]] = {}
for L in LETTERS:
    cand_letters = neighbor_letters(L, span=1) if drift_on else [L]
    # candidate digits are the digits currently at those neighbor letters
    digits = [letter_map[cL] for cL in cand_letters]
    # de‑dupe while preserving order
    seen = set(); dd = []
    for d in digits:
        if d not in seen:
            seen.add(d); dd.append(d)
    letter_to_digits[L] = dd

# Anchors (from current order)
anchors = build_anchors_from_order(order, top_pairs=int(top_pairs), top_trips=int(top_trips)) if anchors_on else []

# Due info from history
all_rows_history = rows  # using provided slice as current history
is_due = due_flags_from_history(all_rows_history, int(due_window))
d1, d2 = d1_d2_from_history(all_rows_history)

# Optionally enforce Due on cold letters by restricting those pools to Due digits first
cold_letters = ["I", "J"]  # tail; could extend to H via UI if desired
if prefer_cold_due:
    for L in cold_letters:
        pool = letter_to_digits[L]
        prioritized = [d for d in pool if is_due.get(d, False)] + [d for d in pool if not is_due.get(d, False)]
        # keep order but move Due first
        # remove duplicates
        seen = set(); dd = []
        for d in prioritized:
            if d not in seen:
                seen.add(d); dd.append(d)
        letter_to_digits[L] = dd

# Schemas to use
schemas: List[List[str]] = []
if use_schema1:
    schemas.append(["A","B","E","E","J"])  # Q2 double on E
if use_schema2:
    schemas.append(["A","A","E","F","J"])  # no double

if not schemas:
    st.warning("Select at least one schema to generate combos.")
    st.stop()

# Generate
run_it = st.button("Run generator")
if not run_it:
    st.info("Enter winners then click **Run generator** to build combos.")
    st.stop()

all_results: List[Tuple[str,int]] = []
for schema in schemas:
    res = generate_from_schema(
        schema_letters=schema,
        letter_to_digits=letter_to_digits,
        must_hit_any_anchor=anchors if anchors_on else None,
        due_on_cold_letters=cold_letters if prefer_cold_due else None,
        d1=d1, d2=d2,
        is_due=is_due,
        prefer_n_double=True,
        cap=50000,
    )
    # optional filters: require >=1 Due; require D1/D2
    if require_due_any:
        res = [(c, s) for (c, s) in res if any(is_due.get(d, False) for d in c)]
    if require_d1d2:
        res = [(c, s) for (c, s) in res if (d1 and d1 in c) or (d2 and d2 in c)]
    all_results.extend(res)

if not all_results:
    st.warning("No combos generated under current settings — relax anchors, drift, or Due requirements.")
    st.stop()

# Build a straight lookup set BEFORE dedupe (for exact-order search)
straight_set = set(c for c, _ in all_results)

# Combine and dedupe (box‑unique via key) keeping max score
best: Dict[str, Tuple[str,int]] = {}
for combo, sc in all_results:
    k = ''.join(sorted(combo))
    if k not in best or sc > best[k][1]:
        best[k] = (combo, sc)

out = pd.DataFrame([{"combo": v[0], "score": v[1]} for k, v in best.items()])
out = out.sort_values(["score","combo"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### Top‑K (Box Unique)")
show_df = out.head(int(top_k))
st.dataframe(show_df, use_container_width=True, hide_index=True)

# +100 bump
col1, col2 = st.columns([1,3])
with col1:
    if st.button("+100 more"):
        new_k = min(int(top_k)+100, len(out))
        show_df = out.head(new_k)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

# Export all
csv_buf = io.StringIO()
out.to_csv(csv_buf, index=False)
st.download_button(
    "Download ALL generated combos (CSV)",
    data=csv_buf.getvalue().encode('utf-8'),
    file_name="aj_heatmap_generated_combos.csv",
    mime="text/csv",
)

# ---------------------------
# Search Known Winner (Box & Straight)
# ---------------------------
st.markdown("## Search for a known winner in generated combos")
query = st.text_input("Enter a 5-digit combo to search (e.g., 12345)", max_chars=5)
if query:
    import re as _re
    if not _re.fullmatch(r"\d{5}", query):
        st.error("Please enter exactly 5 digits.")
    else:
        straight_hit = query in straight_set
        box_hit = box_key(query) in best
        colx, coly = st.columns(2)
        with colx:
            st.metric("Straight match present", "Yes" if straight_hit else "No")
        with coly:
            st.metric("Box match present", "Yes" if box_hit else "No")
        if box_hit:
            st.caption("Representative generated ordering for this box:")
            st.code(best[box_key(query)][0])

# Bulk search (paste many combos)
st.markdown("### Bulk search (paste many combos)")
bulk = st.text_area("Paste combos (whitespace/newline separated)", height=100)
if bulk:
    import re as _re
    tokens = _re.findall(r"\d{5}", bulk)
    if not tokens:
        st.warning("No 5-digit combos found in your paste.")
    else:
        rows = []
        for q in tokens:
            rows.append({
                "combo": q,
                "straight": q in straight_set,
                "box": box_key(q) in best,
            })
        dfhits = pd.DataFrame(rows)
        st.dataframe(dfhits, use_container_width=True, hide_index=True)

# Legend
st.caption("Legend: A..J are current ranks (A=hottest, J=coldest). Drift ±1 lets each letter borrow neighbors (e.g., E∈{D,E,F}). Anchors require the combo to include at least one adjacent pair/triplet from today's heat order. Due uses the last W draws (default W=2). Box uniqueness is applied.")
