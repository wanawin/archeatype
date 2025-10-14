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
def heat_order(rows10: List[List[str]]) -> List[str]:
    """Return digits hot→cold from 10 rows (each row is list of 5 digits)."""
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    # tie-break by digit for determinism
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    """Digit -> heat rank (1=hottest … 10=coldest)."""
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    """±span letters around a given A..J letter (clamped)."""
    idx = LETTERS.index(letter)
    lo, hi = max(0, idx - span), min(9, idx + span)
    return LETTERS[lo:hi + 1]

def due_set(last_two_rows: List[List[str]]) -> set:
    """Digits not seen in the last two draws (W=2)."""
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
    """Compute least→most likely digits for NEXT draw using the ±1 method."""
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need 13 winners (most-recent → oldest).")
    rows = [list(s) for s in last13_mr_to_oldest]

    # Previous map: winners #2..#11
    prev10 = rows[1:11]
    order_prev = heat_order(prev10)
    rank_prev = rank_of_digit(order_prev)

    # Core letters from most-recent winner on previous map
    most_recent = rows[0]
    digit_to_letter_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    core_letters = sorted(
        set(digit_to_letter_prev[d] for d in most_recent),
        key=lambda L: LETTERS.index(L)
    )

    # U = union of ±1 neighborhoods of core letters
    U = set()
    for L in core_letters:
        U.update(neighbors(L, 1))
    U_letters = sorted(U, key=lambda L: LETTERS.index(L))

    # Current map (for NEXT draw): winners #1..#10
    curr10 = rows[0:10]
    order_curr = heat_order(curr10)
    rank_curr = rank_of_digit(order_curr)
    digit_to_letter_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    # Due set (W=2) from winners #1 & #2
    due = due_set(rows[0:2])

    # Age since seen in curr10 (0 = seen in most recent)
    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):  # rows[0] is most-recent
        s = set(r)
        for d in DIGITS:
            if age[d] is None and d in s:
                age[d] = back
    for d in DIGITS:
        if age[d] is None:
            age[d] = 9999

    # Tiering
    tiers = {}
    for d in DIGITS:
        Lc = digit_to_letter_curr[d]
        if Lc not in U:
            tiers[d] = 0                                  # outside U (least)
        elif Lc not in core_letters:
            tiers[d] = 2 if d in due else 1              # neighbors: Due vs not-Due
        else:
            tiers[d] = 3                                  # core (most)

    # Sort inside tier:
    #  - cooler earlier (higher rank number)  ⇒ sort by -heat_rank
    #  - among Due, older later (more likely) ⇒ sort by age ASC
    def sort_key(d: str):
        tier = tiers[d]
        heat_rank = rank_curr[d]      # 1=hottest .. 10=coldest
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
    return ranking, info

def format_report(winners: List[str], ranking: List[str], info: Dict) -> str:
    """Pretty text report for download."""
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
    lines.append(" ".join(ranking))
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

winners_text = st.text_area("Winners (MR → Oldest)", value=default_text, height=140, placeholder="e.g.\n74650,78845,88231,... (13 total)")

if st.button("Compute"):
    try:
        winners = parse_winners_text(winners_text, pad4=pad4)
        if len(winners) < 13:
            st.error(f"Need 13 winners; got {len(winners)}")
        else:
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

            # Downloads
            report_txt = format_report(winners, ranking, info)
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download full report (TXT)",
                data=report_txt.encode("utf-8"),
                file_name="loser_list_report.txt",
                mime="text/plain"
            )
            st.download_button(
                "Download classification (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="loser_list_classification.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(str(e))

with st.expander("How this works"):
    st.markdown("""
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
