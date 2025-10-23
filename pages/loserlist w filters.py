# =====================================================
# loserlist_full.py — Loser List (Least → Most Likely)
# Full version with complete numeric filter generation
# + Digits-only Export/Verification Panel (non-destructive add-on)
# =====================================================
import streamlit as st
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd

# NEW: import the non-destructive export/verification panel
from digits_export_extension import render_export_panel

DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJ")
LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

st.set_page_config(page_title="Loser List (Least → Most Likely)", layout="wide")
st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

# ---------------- Core helpers ----------------
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
    prev10 = rows[1:11]; curr10 = rows[0:10]
    order_prev, order_curr = heat_order(prev10), heat_order(curr10)
    rank_prev, rank_curr = rank_of_digit(order_prev), rank_of_digit(order_curr)

    most_recent = rows[0]
    digit_to_letter_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    digit_to_letter_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    core_letters = sorted({digit_to_letter_prev[d] for d in most_recent}, key=lambda L: LETTERS.index(L))
    U = set().union(*[neighbors(L, 1) for L in core_letters])
    due = due_set(rows[0:2])

    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):
        for d in DIGITS:
            if age[d] is None and d in r: age[d] = back
    for d in DIGITS:
        if age[d] is None: age[d] = 9999

    tiers = {d:(3 if digit_to_letter_curr[d] in core_letters else (2 if (digit_to_letter_curr[d] in U and d in due) else (1 if digit_to_letter_curr[d] in U else 0))) for d in DIGITS}
    ranking = sorted(DIGITS, key=lambda d:(tiers[d], -rank_curr[d], age[d]))
    return ranking, {
        "previous_map_order": "".join(order_prev),
        "current_map_order": "".join(order_curr),
        "core_letters": core_letters,
        "digit_current_letters": digit_to_letter_curr,
        "digit_prev_letters": digit_to_letter_prev,
        "rank_curr_map": rank_curr,
        "rank_prev_map": rank_prev,
        "ranking": ranking
    }

# ---------------- UI ----------------
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries", value=True)
    example_btn = st.button("Load example")
if example_btn:
    winners_text = "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
else:
    winners_text = st.text_area("13 winners (MR→Oldest)", height=140)

if st.button("Compute"):
    try:
        winners = parse_winners_text(winners_text, pad4=pad4)[:13]
        ranking, info = loser_list(winners)

        st.subheader("Loser list (Least → Most Likely)")
        st.code(" ".join(ranking))

        # --- numeric expansions for filters (KEEP SEMANTICS AS-IS) ---
        core_digits = [LETTER_TO_NUM[L] for L in info["core_letters"]]
        new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in info["core_letters"]]
        cooled_digits = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
        loser_7_9 = info["ranking"][7:10]

        def moved(prevL, currL):
            prev_digit = next((d for d in DIGITS if info["digit_prev_letters"][d] == prevL), None)
            return (prev_digit and info["digit_current_letters"][prev_digit] == currL)
        f_to_i, g_to_i = moved('F','I'), moved('G','I')

        def literal(seq): return "[" + ",".join(f"'{x}'" for x in seq) + "]"
        expr_ll007 = "not ('8' in combo_digits)" if f_to_i else "False"
        expr_ll008 = "not ('8' in combo_digits)" if g_to_i else "False"

        # --- full filter list (UNCHANGED) ---
        filters = [
            ("LL001","Eliminate combos with >=3 digits in [0,9,1,2,4]",
             "sum(1 for d in combo_digits if d in ['0','9','1','2','4']) >= 3"),
            ("LL001A","Eliminate combos with no core digits",
             f"sum(1 for d in combo_digits if d in {literal(core_digits)}) == 0"),
            ("LL001B","Eliminate combos with <=2 core digits",
             f"sum(1 for d in combo_digits if d in {literal(core_digits)}) <= 2"),
            ("LL002","Eliminate combos with <2 of loser list 7–9",
             f"sum(1 for d in combo_digits if d in {literal(loser_7_9)}) < 2"),
            ("LL003","Eliminate combos missing >=3 new-core digits",
             f"sum(1 for d in combo_digits if d in {literal(new_core_digits)}) >= 3"),
            ("LL004","Eliminate combos including J(9) unless prev had J",
             "('9' in combo_digits) and not ('9' in seed_digits)"),
            ("LL004R","Eliminate combos with >=2 new-core digits",
             f"sum(1 for d in combo_digits if d in {literal(new_core_digits)}) >= 2"),
            ("LL005","Eliminate combos missing B or E core digits",
             f"not any(d in combo_digits for d in {literal([LETTER_TO_NUM['B'],LETTER_TO_NUM['E']])})"),
            ("LL005A","Eliminate combos with >=3 consecutive core letters",
             "any(ord(combo_digits[i+1])-ord(combo_digits[i])==1 for i in range(len(combo_digits)-1))"),
            ("LL005B","Eliminate combos missing loser list 7–9 entirely",
             f"sum(1 for d in combo_digits if d in {literal(loser_7_9)}) == 0"),
            ("LL007","If prev heatmap F→I, require I(8)",expr_ll007),
            ("LL008","If prev heatmap G→I, require I(8)",expr_ll008),
            ("LL009","Eliminate if cooled digit repeats",
             f"any(combo_digits.count(d)>1 for d in {literal(cooled_digits)})"),
            ("XXXLL003B","Eliminate combos with <=2 new-core digits (aggressive)",
             f"sum(1 for d in combo_digits if d in {literal(new_core_digits)}) <= 2"),
            ("XXXLL002B","Eliminate combos missing >=2 of loser list 7–9 (aggressive)",
             f"sum(1 for d in combo_digits if d in {literal(loser_7_9)}) < 2"),
            ("XXXLL001B","Eliminate combos containing any of 0,9,1,2,4 (simple trim)",
             "any(d in ['0','9','1','2','4'] for d in combo_digits)")
        ]

        csv_lines = ["id,name,enabled,applicable_if,expression,Unnamed:5,Unnamed:6,Unnamed:7,Unnamed:8,Unnamed:9,Unnamed:10,Unnamed:11,Unnamed:12,Unnamed:13,Unnamed:14"]
        for fid,name,expr in filters:
            csv_lines.append(f'{fid},"{name}",True,,"{expr}",,,,,,,,,,')

        st.markdown("### 📋 Auto-Generated Filters (copy/paste to tester)")
        st.code("\n".join(csv_lines), language="csv")

        # =====================
        # NEW: Export/Verification Panel (digits-only + state display)
        # =====================
        render_export_panel(
            filters_csv_text="\n".join(csv_lines),
            digit_current_letters=info["digit_current_letters"],
            digit_prev_letters=info["digit_prev_letters"],
            prev_core_letters=set(info["core_letters"]),
            cooled_digits=set(cooled_digits),
            new_core_digits=set(new_core_digits),
            loser_7_9=list(loser_7_9),
        )

        with st.expander("Details used for numeric mapping"):
            st.write("Core digits:", core_digits)
            st.write("New-core digits:", new_core_digits)
            st.write("Cooled digits:", cooled_digits)
            st.write("Loser list 7–9:", loser_7_9)
            st.write("F→I:", f_to_i, " | G→I:", g_to_i)

    except Exception as e:
        st.error(str(e))
