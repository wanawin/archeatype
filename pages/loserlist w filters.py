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
# digits-only export panel (inline helper) — no external import needed
import io, re
from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")


def _digits_by_letter(letter_map: Dict[str, str]) -> Dict[str, List[str]]:
    out = {L: [] for L in LETTERS}
    for d, L in letter_map.items():
        out[L].append(d)
    for L in LETTERS:
        out[L] = sorted(out[L])
    return out


def _ring_digits(prev_core_letters: Set[str], digit_current_letters: Dict[str, str]) -> List[str]:
    if not prev_core_letters:
        return []
    neigh = set()
    for L in prev_core_letters:
        if L in LETTERS:
            i = LETTERS.index(L)
            if i - 1 >= 0:
                neigh.add(LETTERS[i - 1])
            if i + 1 < len(LETTERS):
                neigh.add(LETTERS[i + 1])
    return sorted([d for d, L in digit_current_letters.items() if L in neigh])


def _csv_to_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text), header=0)


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _replace_sets(
    expr: str,
    digits_by_letter: Dict[str, List[str]],
    prev_core_letters: Set[str],
    cooled_digits: Set[str],
    new_core_digits: Set[str],
    loser_7_9: List[str],
    ring_digits: List[str],
) -> str:
    out = expr

    def _fmt(lst: List[str] | Set[str]) -> str:
        return "[" + ",".join(f"'{d}'" for d in sorted(list(lst))) + "]"

    out = re.sub(r"\bin\s+cooled_digits\b", f"in {_fmt(cooled_digits)}", out)
    out = re.sub(r"\bin\s+new_core_digits\b", f"in {_fmt(new_core_digits)}", out)
    out = re.sub(r"\bin\s+loser_7_9\b", f"in {_fmt(loser_7_9)}", out)
    out = re.sub(r"\bin\s+ring_digits\b", f"in {_fmt(ring_digits)}", out)

    pat_letter_membership = re.compile(
        r"digit_current_letters\s*\[\s*([a-zA-Z_][\w]*)\s*\]\s*in\s*\[(.*?)\]"
    )

    def _sub_letter_mem(m: re.Match) -> str:
        var_d = m.group(1)
        letters_raw = m.group(2)
        letters = [tok.strip().strip("'\"") for tok in letters_raw.split(',') if tok.strip()]
        digits = sorted({d for L in letters for d in digits_by_letter.get(L, [])})
        lst = "[" + ",".join(f"'{x}'" for x in digits) + "]"
        return f"{var_d} in {lst}"

    out = pat_letter_membership.sub(_sub_letter_mem, out)

    def _letter_in_set(expr_in: str, varname: str, ref_set: Set[str]) -> str:
        pat = re.compile(rf"'([A-J])'\s+in\s+{varname}")
        def repl(mm: re.Match) -> str:
            L = mm.group(1)
            return "True" if L in ref_set else "False"
        return pat.sub(repl, expr_in)

    out = _letter_in_set(out, "prev_core_letters", prev_core_letters)
    out = _letter_in_set(out, "core_letters", prev_core_letters)
    return out


def digits_only_transform(
    csv_text: str,
    digit_current_letters: Dict[str, str],
    digit_prev_letters: Dict[str, str],
    prev_core_letters: Set[str],
    cooled_digits: Set[str],
    new_core_digits: Set[str],
    loser_7_9: List[str],
):
    digits_by_letter_curr = _digits_by_letter(digit_current_letters)
    ring = _ring_digits(prev_core_letters, digit_current_letters)
    df = _csv_to_df(csv_text)
    rows = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        desc = str(row.get("description", "")).strip()
        expr = str(row.get("expression", ""))
        new_expr = _replace_sets(
            expr,
            digits_by_letter_curr,
            prev_core_letters,
            cooled_digits,
            new_core_digits,
            loser_7_9,
            ring,
        )
        rows.append({"name": name, "description": desc, "expression": new_expr})
    return pd.DataFrame(rows, columns=["name", "description", "expression"])


def render_export_panel(
    filters_csv_text: str,
    digit_current_letters: Dict[str, str],
    digit_prev_letters: Dict[str, str],
    prev_core_letters: Set[str],
    cooled_digits: Set[str],
    new_core_digits: Set[str],
    loser_7_9: List[str],
) -> None:
    st.subheader("Digits-only Filter Export • Verification Panel")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Current heatmap (digit → letter)**")
        df_curr = pd.DataFrame({"digit": DIGITS, "letter": [digit_current_letters[d] for d in DIGITS]})
        st.dataframe(df_curr, use_container_width=True, hide_index=True)
    with colB:
        st.markdown("**Previous heatmap (digit → letter)**")
        df_prev = pd.DataFrame({"digit": DIGITS, "letter": [digit_prev_letters[d] for d in DIGITS]})
        st.dataframe(df_prev, use_container_width=True, hide_index=True)

    ring = _ring_digits(prev_core_letters, digit_current_letters)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**prev_core_letters**")
        st.code(", ".join(sorted(prev_core_letters)) or "∅")
        st.markdown("**loser_7_9 (digits)**")
        st.code(", ".join(loser_7_9) or "∅")
    with col2:
        st.markdown("**cooled_digits (digits)**")
        st.code(", ".join(sorted(cooled_digits)) or "∅")
        st.markdown("**new_core_digits (digits)**")
        st.code(", ".join(sorted(new_core_digits)) or "∅")
    with col3:
        st.markdown("**ring_digits (computed, digits)**")
        st.code(", ".join(ring) or "∅")

    st.divider()
    st.markdown("**Original CSV (input)** — edit below if you want, then export:")
    csv_in = st.text_area("filters.csv (name,description,expression)", value=filters_csv_text, height=240)

    out_df = digits_only_transform(
        csv_in,
        digit_current_letters,
        digit_prev_letters,
        prev_core_letters,
        cooled_digits,
        new_core_digits,
        loser_7_9,
    )

    st.markdown("**Digits-only CSV (output)** — copy/paste:")
    st.code(out_df.to_csv(index=False), language="csv")

    st.download_button(
        "Download digits-only CSV",
        data=_df_to_csv_bytes(out_df),
        file_name="filters_export_digits_only.csv",
        mime="text/csv",
    )
# --- END inline helper ---

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
st.code("
".join(csv_lines), language="csv")

# NEW: Let user supply the MEGA CSV (or use generated) for export/verification
st.markdown("### CSV Source for Digits-Only Export")
use_mega = st.toggle("Use my mega CSV below (recommended)", value=True)
mega_csv = st.text_area("Paste your MEGA CSV (name,description,expression)", height=180, value="")
csv_source_text = mega_csv if (use_mega and mega_csv.strip()) else "
".join(csv_lines)

# Render verification + digits-only exporter
render_export_panel(
    filters_csv_text=csv_source_text,
    digit_current_letters=info["digit_current_letters"],
    digit_prev_letters=info["digit_prev_letters"],
    prev_core_letters=set(info["core_letters"]),
    cooled_digits=set(cooled_digits),
    new_core_digits=set(new_core_digits),
    loser_7_9=list(loser_7_9),
)

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
