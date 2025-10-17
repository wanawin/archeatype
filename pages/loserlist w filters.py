# loserlist_with_filters.py  — Loser List (Least → Most Likely)
# NOTE: UI + core logic preserved; only the CSV filter generator was added/changed.
import streamlit as st
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd

DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJ")
LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
NUM_TO_LETTER = {v:k for k,v in LETTER_TO_NUM.items()}

st.set_page_config(page_title="Loser List (Least → Most Likely)", layout="wide")
st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

# ---------------- Core helpers (unchanged) ----------------
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
            raise ValueError(f"Each item must be 5 digits (or 4 with 'Pad 4-digits')")
        out.append(tok)
    return out

def loser_list(last13_mr_to_oldest: List[str]) -> Tuple[List[str], Dict]:
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need 13 winners (most-recent → oldest).")
    rows = [list(s) for s in last13_mr_to_oldest]

    # previous map = winners #2..#11
    prev10 = rows[1:11]
    order_prev = heat_order(prev10)                 # hot→cold (digits as strings)
    rank_prev = rank_of_digit(order_prev)           # digit -> 1..10

    # current map = winners #1..#10
    curr10 = rows[0:10]
    order_curr = heat_order(curr10)
    rank_curr = rank_of_digit(order_curr)

    # letters for MR winner on previous map (core letters)
    most_recent = rows[0]
    digit_to_letter_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    core_letters = sorted({digit_to_letter_prev[d] for d in most_recent}, key=lambda L: LETTERS.index(L))

    # ±1 union around core letters (letters)
    U = set()
    for L in core_letters:
        U.update(neighbors(L, 1))
    U_letters = sorted(U, key=lambda L: LETTERS.index(L))

    # digit -> current/prev letters
    digit_to_letter_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    # due set from last 2 winners (MR & 2-back)
    due = due_set(rows[0:2])

    # ages in current window
    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):
        s = set(r)
        for d in DIGITS:
            if age[d] is None and d in s:
                age[d] = back
    for d in DIGITS:
        if age[d] is None:
            age[d] = 9999

    # tiers per your original logic
    tiers = {}
    for d in DIGITS:
        Lc = digit_to_letter_curr[d]
        if Lc not in U_letters:
            tiers[d] = 0
        elif Lc not in core_letters:
            tiers[d] = 2 if d in due else 1
        else:
            tiers[d] = 3

    def sort_key(d: str):
        return (tiers[d], -rank_curr[d], age[d])

    ranking = sorted(DIGITS, key=sort_key)  # least→most likely per your method

    info = {
        "previous_map_order": "".join(order_prev),
        "current_map_order": "".join(order_curr),
        "core_letters": core_letters,
        "U_letters": U_letters,
        "due_set": sorted(list(due)),
        "digit_current_letters": digit_to_letter_curr,
        "digit_prev_letters":    digit_to_letter_prev,
        "rank_curr_map": rank_curr,
        "rank_prev_map": rank_prev,
        "ranking": ranking
    }
    return ranking, info

# ---------------- UI (unchanged) ----------------
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries with a leading 0 (e.g., 8162 → 08162)", value=True)
    example_btn = st.button("Load example (demo)")

st.caption("Enter **13 winners** in **MOST-RECENT → OLDEST** order, separated by commas or newlines.")
default_text = "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921" if example_btn else ""
winners_text = st.text_area("Winners (MR → Oldest)", value=default_text, height=140)

if st.button("Compute"):
    try:
        winners = parse_winners_text(winners_text, pad4=pad4)
        winners = winners[:13]
        ranking, info = loser_list(winners)

        st.subheader("Loser list (Least → Most Likely)")
        st.code(" ".join(ranking))

        st.markdown("### Digit classification (today)")
        df = pd.DataFrame([{
            "digit": d,
            "prev_letter": info["digit_prev_letters"][d],
            "curr_letter": info["digit_current_letters"][d],
            "prev_rank": info["rank_prev_map"][d],
            "curr_rank": info["rank_curr_map"][d],
        } for d in DIGITS]).sort_values(["curr_rank"])
        st.dataframe(df, hide_index=True, use_container_width=True)

        # ---------- NUMERIC EXPANSIONS FOR FILTERS ----------
        # previous-core digits (numbers 0..9) from previous-map letters used by MR winner
        core_digits = [LETTER_TO_NUM[L] for L in info["core_letters"]]         # list[int]
        core_digits_s = ",".join(str(x) for x in core_digits)                  # "2,5,7" etc
        core_digits_list_literal = "[" + ",".join(f"'{x}'" for x in core_digits) + "]"

        # new-core digits today (digits whose CURRENT letter not in previous core letters)
        new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in info["core_letters"]]
        new_core_list_literal = "[" + ",".join(f"'{d}'" for d in new_core_digits) + "]"

        # digits that cooled (rank number increased, i.e., got colder)
        cooled_digits = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
        cooled_list_literal = "[" + ",".join(f"'{d}'" for d in cooled_digits) + "]"

        # loser list 7–9 (positions 7,8,9 from least→most likely)
        loser_7_9 = info["ranking"][7:10] if len(info["ranking"]) >= 10 else []
        loser_7_9_literal = "[" + ",".join(f"'{d}'" for d in loser_7_9) + "]"

        # transition checks (previous letter -> current letter for a specific digit)
        # F→I means: the digit that was 'F' on prev map now maps to 'I' on curr map.
        def moved(prev_letter: str, curr_letter: str) -> bool:
            prev_digit = next((d for d in DIGITS if info["digit_prev_letters"][d] == prev_letter), None)
            return (prev_digit is not None) and (info["digit_current_letters"][prev_digit] == curr_letter)

        f_to_i = moved('F', 'I')
        g_to_i = moved('G', 'I')

        # Expressions: if transition is absent, make them inert with "False"
        expr_ll007 = "not ('8' in combo_digits)" if f_to_i else "False"
        expr_ll008 = "not ('8' in combo_digits)" if g_to_i else "False"

        # ---------- CSV FILTER BLOCK (ALL NUMERIC, NO info/ranking REFERENCES) ----------
        rows = []
        rows.append("id,name,enabled,applicable_if,expression,Unnamed:5,Unnamed:6,Unnamed:7,Unnamed:8,Unnamed:9,Unnamed:10,Unnamed:11,Unnamed:12,Unnamed:13,Unnamed:14")

        # LL007 / LL008 (transitions)
        rows.append(f'LL007,"If previous heatmap F moves to I, require combo to include I(8)",True,,"{expr_ll007}",,,,,,,,,,')
        rows.append(f'LL008,"If previous heatmap G moves to I, require combo to include I(8)",True,,"{expr_ll008}",,,,,,,,,,')

        # LL009 (cooled digit doubles)
        rows.append(f'LL009,"Eliminate if any digit that cooled appears more than once in combo",True,,"any(combo_digits.count(d) > 1 for d in {cooled_list_literal})",,,,,,,,,,')

        # LL004R (reverse) — eliminate if ≥2 new-core digits
        rows.append(f'LL004R,"Eliminate combos with two or more new-core digits",True,,"sum(1 for d in combo_digits if d in {new_core_list_literal}) >= 2",,,,,,,,,,')

        # XXXLL003B (aggressive last resort) — eliminate if ≤2 new-core digits
        rows.append(f'XXXLL003B,"Eliminate combos with ≤2 new-core digits (aggressive trim)",True,,"sum(1 for d in combo_digits if d in {new_core_list_literal}) <= 2",,,,,,,,,,')

        # XXXLL002B (aggressive) — missing ≥2 of loser list 7–9
        rows.append(f'XXXLL002B,"Eliminate combos missing ≥2 of loser list digits 7–9 (aggressive)",True,,"sum(1 for d in combo_digits if d in {loser_7_9_literal}) < 2",,,,,,,,,,')

        # XXXLL001B (aggressive) — simple trim on {0,9,1,2,4}
        rows.append("XXXLL001B,\"Eliminate combos containing any of digits 0,9,1,2,4 (simple trim)\",True,,\"any(d in ['0','9','1','2','4'] for d in combo_digits)\",,,,,,,,,,")

        csv_block = "\n".join(rows)

        st.markdown("### 📋 Auto-Generated Filters (copy/paste to your tester)")
        st.code(csv_block, language="csv")

        # (optional) show the concretized sets for sanity
        with st.expander("Details used for filters (resolved numbers)"):
            st.write("Previous core digits:", core_digits)
            st.write("New-core digits today:", new_core_digits)
            st.write("Cooled digits:", cooled_digits)
            st.write("Loser list digits 7–9:", loser_7_9)
            st.write("F→I transition:", f_to_i, " | G→I transition:", g_to_i)

    except Exception as e:
        st.error(str(e))
