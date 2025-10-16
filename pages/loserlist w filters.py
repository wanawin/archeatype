# =========================================================
#  Loser List (Least → Most Likely) — ±1 Neighborhood Method
#  Updated: Adds automatic filter export (LL007–LL009, etc.)
# =========================================================
import streamlit as st
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd
import io

DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJ")

st.set_page_config(page_title="Loser List (Least → Most Likely)", layout="wide")
st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

# ======================
# Core helper functions
# ======================
def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    idx = LETTERS.index(letter)
    lo, hi = max(0, idx - span), min(9, idx + span)
    return LETTERS[lo:hi + 1]

def due_set(last_two_rows: List[List[str]]) -> set:
    seen = set(d for r in last_two_rows for d in r)
    return set(DIGITS) - seen

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    raw = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
    out = []
    for tok in raw:
        if not tok.isdigit():
            raise ValueError(f"Non-digit token: {tok!r}")
        if len(tok) == 4 and pad4:
            tok = tok.zfill(5)
        if len(tok) != 5:
            raise ValueError(f"Each item must be 5 digits")
        out.append(tok)
    return out

def loser_list(last13_mr_to_oldest: List[str]) -> Tuple[List[str], Dict]:
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need 13 winners (most-recent → oldest).")
    rows = [list(s) for s in last13_mr_to_oldest]

    prev10 = rows[1:11]
    order_prev = heat_order(prev10)
    rank_prev = rank_of_digit(order_prev)

    most_recent = rows[0]
    digit_to_letter_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    core_letters = sorted(set(digit_to_letter_prev[d] for d in most_recent), key=lambda L: LETTERS.index(L))

    U = set()
    for L in core_letters:
        U.update(neighbors(L, 1))
    U_letters = sorted(U, key=lambda L: LETTERS.index(L))

    curr10 = rows[0:10]
    order_curr = heat_order(curr10)
    rank_curr = rank_of_digit(order_curr)
    digit_to_letter_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    due = due_set(rows[0:2])

    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):
        s = set(r)
        for d in DIGITS:
            if age[d] is None and d in s:
                age[d] = back
    for d in DIGITS:
        if age[d] is None:
            age[d] = 9999

    tiers = {}
    for d in DIGITS:
        Lc = digit_to_letter_curr[d]
        if Lc not in U:
            tiers[d] = 0
        elif Lc not in core_letters:
            tiers[d] = 2 if d in due else 1
        else:
            tiers[d] = 3

    def sort_key(d: str):
        return (tiers[d], -rank_curr[d], age[d])

    ranking = sorted(DIGITS, key=sort_key)

    info = {
        "previous_map_order": "".join(order_prev),
        "current_map_order": "".join(order_curr),
        "core_letters": core_letters,
        "U_letters": U_letters,
        "due_set": sorted(list(due)),
        "digit_current_letters": {d: digit_to_letter_curr[d] for d in DIGITS},
        "digit_prev_letters": {d: digit_to_letter_prev[d] for d in DIGITS},
        "rank_curr_map": rank_curr,
        "rank_prev_map": rank_prev,
    }
    return ranking, info

def format_report(winners: List[str], ranking: List[str], info: Dict) -> str:
    lines = []
    lines.append("=== INPUT ===")
    lines.append(", ".join(winners))
    lines.append("")
    lines.append(f"Prev map: {info['previous_map_order']}")
    lines.append(f"Curr map: {info['current_map_order']}")
    lines.append(f"Core letters: {', '.join(info['core_letters'])}")
    lines.append(f"U (±1): {', '.join(info['U_letters'])}")
    lines.append(f"Due: {', '.join(info['due_set'])}")
    lines.append("")
    lines.append("Loser list (Least → Most Likely):")
    lines.append(" ".join(ranking))
    return "\n".join(lines)

# =============== Streamlit UI ===============
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries (08162)", value=True)
    example_btn = st.button("Load example")

if example_btn:
    winners_text = "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
else:
    winners_text = st.text_area("Enter 13 winners (MR→Oldest):", height=140)

if st.button("Compute"):
    winners = parse_winners_text(winners_text, pad4=pad4)
    ranking, info = loser_list(winners)

    st.subheader("Loser list (Least → Most Likely)")
    st.code(" ".join(ranking))

    st.write("### Digit Classifications")
    rows = [{"Digit": d, "Prev Rank": info['rank_prev_map'][d],
             "Curr Rank": info['rank_curr_map'][d],
             "Prev→Curr": f"{info['digit_prev_letters'][d]}→{info['digit_current_letters'][d]}"} for d in DIGITS]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ==================================================
    # NEW FILTER GENERATION SECTION (auto CSV block)
    # ==================================================
    st.markdown("### 📋 Auto-Generated Filters (Copy-Paste Block)")
    filters = [
        ("LL007","If previous heatmap F moves to I, require combo to include I(8)",
         "(5 in info['rank_prev_map'].values() and 8 in info['rank_curr_map'].values()) and not (8 in combo_digits)"),
        ("LL008","If previous heatmap G moves to I, require combo to include I(8)",
         "(6 in info['rank_prev_map'].values() and 8 in info['rank_curr_map'].values()) and not (8 in combo_digits)"),
        ("LL009","Eliminate if any digit that cooled appears more than once",
         "any(combo_digits.count(d) > 1 for d in [d for d in DIGITS if info['rank_curr_map'][d] > info['rank_prev_map'][d]])"),
        ("LL004R","Eliminate combos with two or more new-core letters",
         "sum(1 for d in combo_digits if info['digit_current_letters'][d] not in info['core_letters']) >= 2"),
        ("XXXLL003B","Eliminate combos with ≤2 new-core letters (aggressive trim)",
         "sum(1 for d in combo_digits if info['digit_current_letters'][d] not in info['core_letters']) <= 2"),
        ("XXXLL002B","Eliminate combos missing ≥2 of loser list digits 7–9 (aggressive trim)",
         "sum(1 for d in combo_digits if d in ranking[7:10]) < 2"),
        ("XXXLL001B","Eliminate combos containing any of digits 0,9,1,2,4 (simple trim, no dup check)",
         "any(d in ['0','9','1','2','4'] for d in combo_digits)"),
    ]

    csv_lines = ["id,name,enabled,applicable_if,expression,Unnamed:5,Unnamed:6,Unnamed:7,Unnamed:8,Unnamed:9,Unnamed:10,Unnamed:11,Unnamed:12,Unnamed:13,Unnamed:14"]
    for fid, name, expr in filters:
        csv_lines.append(f'{fid},"{name}",True,,"{expr}",,,,,,,,,,')
    csv_block = "\n".join(csv_lines)
    st.code(csv_block, language="csv")

    st.success("✅ Copy this block and paste directly into your Filter Tester CSV under LL006.")
