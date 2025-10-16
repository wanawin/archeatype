# loserlist.py  — Streamlit page
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
# ... (all existing helpers unchanged)

def loser_list(last13_mr_to_oldest: List[str]) -> Tuple[List[str], Dict]:
    """Compute least→most likely digits for NEXT draw using the ±1 method."""
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need 13 winners (most-recent → oldest).")
    rows = [list(s) for s in last13_mr_to_oldest]

    # Previous map: winners #2..#11
    prev10 = rows[1:11]
    order_prev = heat_order(prev10)
    rank_prev = rank_of_digit(order_prev)

    # Core letters from most-recent winner on previous map
    # ... (unchanged code)

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
    return ranking, info

def build_report(winners: List[str], ranking: List[str], info: Dict) -> str:
    lines = []
    lines.append("=== INPUT (most-recent -> oldest, 13 winners) ===")
    lines.append(", ".join(winners))
    lines.append("")
    lines.append("=== Maps ===")
    lines.append(f"Previous map hot->cold (for most-recent winner): {info['previous_map_order']}")
    lines.append(f"Current  map hot->cold (for next draw):         {info['current_map_order']}")
    lines.append("")
    lines.append(f"Core letters (from winner #1 on previous map): {', '.join(info['core_letters'])}")
    lines.append(f"U (±1 union): {', '.join(info['U_letters'])}")
    lines.append(f"Due (W=2): {', '.join(info['due_set'])}")
    lines.append("")
    lines.append("=== Digit classifications (today) ===")
    lines.append("digit : letter  tier  age  heat_rank_today")
    for d in DIGITS:
        lines.append(
            f"  {d}    :   {info['digit_current_letters'][d]}     {info['digit_tiers'][d]}    "
            f"{info['digit_ages'][d]}    {info['rank_curr_map'][d]}"
        )
    lines.append("")
    lines.append("=== Loser list (least -> most likely) ===")
    # ►►► Single minimal change: add position numbers 0..9
    lines.append(" ".join(f"{i}:{d}" for i, d in enumerate(ranking)))
    return "\n".join(lines)

# ===============
# Streamlit UI
# ===============
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries with a leading 0 (e.g., 8162 → 08162)", value=True)
    example_btn = st.button("Load example (demo)")

st.caption("Enter **13 winners** in **MOST-RECENT → OLDEST** order, separated by commas or newlines.")

default_text = ""
if example_btn:
    default_text = "74650, 78845, 88231, 19424, 37852, 91664, 33627, 95465, 53502, 41621, 05847, 35515, 81921"

winners_text = st.text_area("Winners (MR → Oldest)", value=default_text, height=140,
                            placeholder="e.g.\n74650,78845,88231,... (13 total)")

if st.button("Compute"):
    try:
        winners = parse_winners_text(winners_text, pad4=pad4)
        ranking, info = loser_list(winners)

        st.subheader("Loser list (Least → Most Likely)")
        # ►►► Single minimal change: add position numbers 0..9
        st.code(" ".join(f"{i}:{d}" for i, d in enumerate(ranking)))

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
            st.markdown("**Due (W=2)** — digits not present in winners #1 & #2")
            st.code(", ".join(info["due_set"]))

        # Downloads (unchanged)
        report = build_report(winners, ranking, info)
        st.download_button("Download text report", data=report, file_name="loser_list_report.txt")
        df = pd.DataFrame({
            "digit": DIGITS,
            "letter_today": [info["digit_current_letters"][d] for d in DIGITS],
            "tier": [info["digit_tiers"][d] for d in DIGITS],
            "age": [info["digit_ages"][d] for d in DIGITS],
            "heat_rank_today": [info["rank_curr_map"][d] for d in DIGITS],
        })
        st.download_button("Download classifications (CSV)",
                           data=df.to_csv(index=False), file_name="loser_list_classification.csv")

        with st.expander("Method details"):
            st.markdown(r"""
**Windows:** previous map = winners #2..#11; current map = winners #1..#10.  
**Core letters:** letters used by winner #1 on the previous map.  
**U:** union of ±1 neighborhoods around core letters.  
**Due (W=2):** digits not seen in winners #1 & #2.

**Tiering**
- Tier 0: outside **U** (least)
- Tier 1: in **U**, not core, **not Due**
- Tier 2: in **U**, not core, **Due**
- Tier 3: **core letters** (most)

**Ordering inside tiers**
- Cooler earlier (higher heat rank number), hotter later.  
- For Due digits, older later (more likely).
""")
    except Exception as e:
        st.error(str(e))
