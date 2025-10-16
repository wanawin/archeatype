# loserlist.py — corrected version (safe patch, UI untouched)
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
            raise ValueError(f"Each item must be 5 digits (or 4 with 'Pad 4-digits'): got {tok!r}")
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
    core_letters = sorted(
        set(digit_to_letter_prev[d] for d in most_recent),
        key=lambda L: LETTERS.index(L)
    )

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
        tier = tiers[d]
        heat_rank = rank_curr[d]
        return (tier, -heat_rank, age[d])

    ranking = sorted(DIGITS, key=sort_key)

    info = {
        "previous_map_order": "".join(order_prev),
        "current_map_order": "".join(order_curr),
        "core_letters": core_letters,
        "U_letters": U_letters,
        "due_set": sorted(list(due)),
        "digit_current_letters": {d: digit_to_letter_curr[d] for d in DIGITS},
        "digit_tiers": tiers,
        "digit_ages": age,
        "rank_curr_map": rank_curr
    }

    # --- new patch: numeric equivalents for filters ---
    LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
    NUM_TO_LETTER = {v:k for k,v in LETTER_TO_NUM.items()}
    info["core_digits"] = [LETTER_TO_NUM[L] for L in info["core_letters"]]
    info["U_digits"] = [LETTER_TO_NUM[L] for L in info["U_letters"]]
    info["letter_to_num"] = LETTER_TO_NUM
    info["num_to_letter"] = NUM_TO_LETTER
    # --- end patch ---

    return ranking, info

# =============== UI ===============
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries with a leading 0 (e.g., 8162 → 08162)", value=True)
    example_btn = st.button("Load example (demo)")

st.caption("Enter **13 winners** in **MOST-RECENT → OLDEST** order, separated by commas or newlines.")

default_text = ""
if example_btn:
    default_text = "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"

winners_text = st.text_area("Winners (MR → Oldest)", value=default_text, height=140)

if st.button("Compute"):
    try:
        winners = parse_winners_text(winners_text, pad4=pad4)
        winners = winners[:13]
        ranking, info = loser_list(winners)

        st.subheader("Loser list (Least → Most Likely)")
        st.code(" ".join(ranking))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Previous map (for MR winner)** — hot → cold")
            st.code(info["previous_map_order"])
            st.markdown("**Core letters from MR winner (prev map)**")
            st.code(", ".join(info["core_letters"]))
            st.markdown("**±1 union U (letters)**")
            st.code(", ".join(info["U_letters"]))
        with col2:
            st.markdown("**Current map (for next draw)** — hot → cold")
            st.code(info["current_map_order"])
            st.markdown("**Due (W=2)**")
            st.code(", ".join(info["due_set"]))

        st.markdown("### Digit classification (today)")
        rows = []
        for d in DIGITS:
            rows.append({
                "digit": d,
                "letter_today": info["digit_current_letters"][d],
                "tier": info["digit_tiers"][d],
                "age": info["digit_ages"][d],
                "heat_rank_today (1=hottest)": info["rank_curr_map"][d],
            })
        df = pd.DataFrame(rows).sort_values(
            ["tier", "heat_rank_today (1=hottest)", "age"],
            ascending=[True, True, True]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

        # ---- FILTER CSV OUTPUT BLOCK (new) ----
        st.subheader("📋 Copy-Paste Filter CSV Block (LL001–LL005a)")
        filters_csv = """id,name,enabled,applicable_if,expression,Unnamed:5,Unnamed:6,Unnamed:7,Unnamed:8,Unnamed:9,Unnamed:10,Unnamed:11,Unnamed:12,Unnamed:13,Unnamed:14
LL001,"Eliminate if combo contains ≥3 of digits {0,9,1,2,4}",True,,"sum(int(d) in [0,9,1,2,4] for d in combo_digits) >= 3",,,,,,,,,,
LL001a,"Eliminate if ≥3 core digits and lacks B(1)/E(4) and lacks top3 hot digits (7,3,6)",True,,"sum(int(d) in info['core_digits'] for d in combo_digits) >=3 and not any(int(d) in [1,4] for d in combo_digits) and not any(int(d) in [7,3,6] for d in combo_digits)",,,,,,,,,,
LL002,"Eliminate if combo lacks both B(1) and E(4)",True,,"not any(int(d) in [1,4] for d in combo_digits)",,,,,,,,,,
LL002a,"Eliminate if missing both B/E and has fewer than 2 of {1,9,0}",True,,"(not any(int(d) in [1,4] for d in combo_digits)) and sum(int(d) in [1,9,0] for d in combo_digits) < 2",,,,,,,,,,
LL003,"Eliminate if combo includes J(9) unless seed has J",True,,"(9 in combo_digits) and not (9 in seed_digits)",,,,,,,,,,
LL005,"Eliminate if combo includes ≥3 of core_digits (from current map)",True,,"sum(int(d) in info['core_digits'] for d in combo_digits) >= 3",,,,,,,,,,
LL005a,"Soft penalty if ≥3 core digits (reduce score only, do not eliminate)",True,,"sum(int(d) in info['core_digits'] for d in combo_digits) >= 3",,,,,,,,,,
"""
        st.code(filters_csv, language="csv")

    except Exception as e:
        st.error(str(e))
