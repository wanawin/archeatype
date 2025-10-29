# streamlit_app_loserlist_tester_export.py
import io
import re
from typing import Dict, List, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")

# =========================
# Helpers
# =========================
def normalize_quotes(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\r\n", "\n").replace("\r", "\n")
    )

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    txt = normalize_quotes(txt)
    tokens = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
    out = []
    for t in tokens:
        if not t.isdigit():
            raise ValueError(f"Non-digit token: {t!r}")
        if len(t) == 4 and pad4:
            t = t.zfill(5)
        if len(t) != 5:
            raise ValueError("Each entry must be exactly 5 digits.")
        out.append(t)
    return out

def heat_order(rows10: List[List[str]]) -> List[str]:
    """Return digits (as str) ordered by frequency desc, then digit asc."""
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo:hi + 1]

def loser_list(last13: List[str]) -> Tuple[List[str], Dict]:
    """Build maps and ranking from 13 winners: [MR, 2nd, ..., oldest]."""
    if len(last13) < 13:
        raise ValueError("Need 13 winners (Most Recent → Oldest).")
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]

    order_prev, order_curr = heat_order(prev10), heat_order(curr10)
    rank_prev,  rank_curr  = rank_of_digit(order_prev), rank_of_digit(order_curr)

    most_recent = rows[0]
    # Map digit->letter using ranks (A hottest … J coldest)
    digit_prev_letters  = {d: LETTERS[rank_prev[d]  - 1] for d in DIGITS}
    digit_curr_letters  = {d: LETTERS[rank_curr[d]  - 1] for d in DIGITS}

    # Core letters are the letters (from previous map) of the most recent digits
    core_letters_prevmap = sorted({digit_prev_letters[d] for d in most_recent},
                                  key=lambda L: LETTERS.index(L))

    # Ring letters are ±1 neighbors around the core letters (in current map space)
    ring_letters = set()
    for L in core_letters_prevmap:
        ring_letters.update(neighbors(L, 1))

    # Build a “loser list” ranking just as an example (you may keep your own)
    # For this app, we mainly need loser_7_9 (positions 7,8,9 among 0..9)
    # We'll derive a simple ranking by (distance from core) and some tiebreakers.
    # You can swap this with your exact logic if different.
    def tier(d: str) -> int:
        L = digit_curr_letters[d]
        # closer to core => higher tier; but we want “losers” later
        if L in core_letters_prevmap: return 3
        if L in ring_letters:         return 2
        return 1
    ranking = sorted(DIGITS, key=lambda d: (tier(d), -rank_curr[d], d))

    return ranking, {
        "current_map_order": "".join(order_curr),
        "previous_map_order": "".join(order_prev),
        "digit_current_letters": digit_curr_letters,
        "digit_prev_letters":    digit_prev_letters,
        "core_letters_prevmap":  core_letters_prevmap,
        "ring_letters":          sorted(ring_letters, key=lambda L: LETTERS.index(L)),
        "rank_curr_map":         rank_curr,
        "rank_prev_map":         rank_prev,
    }

def digits_by_letter(letter_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Return {Letter: [digits]} using a digit->letter map."""
    out = {L: [] for L in LETTERS}
    for d, L in letter_map.items():
        out[L].append(d)
    return out

def ring_digits_from_letters(ring_letters: Set[str],
                             digit_curr_letters: Dict[str, str]) -> List[str]:
    """Digits whose current letter is in ring_letters; keep 0..9 order."""
    return [d for d in DIGITS if digit_curr_letters.get(d) in ring_letters]

def _read_csv_loose(text: str) -> pd.DataFrame:
    text = normalize_quotes(text)
    last_err = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python",
                quotechar='"',
                escapechar="\\",
                dtype=str,
                on_bad_lines="skip",
            )
            df.columns = [str(c).strip() for c in df.columns]
            return df.fillna("")
        except Exception as e:
            last_err = e
    raise last_err

def to_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(c.lower() for c in df.columns)
    lower_map = {c.lower(): c for c in df.columns}

    # 3-col path: name,description,expression
    if {"name","description","expression"}.issubset(cols) and "id" not in cols:
        out = df[[lower_map["name"], lower_map["description"], lower_map["expression"]]].copy()
        out.columns = ["name","description","expression"]
        out = out[out["expression"].astype(str).str.strip() != ""]
        # Drop accidental header rows inside the body
        out = out[out["name"].str.lower() != "name"]
        return out.reset_index(drop=True)

    # 5-col tester path: id,name,enabled,applicable_if,expression
    needed = {"id","name","enabled","applicable_if","expression"}
    if needed.issubset(cols):
        out = df[[lower_map["id"], lower_map["name"], lower_map["enabled"],
                  lower_map["applicable_if"], lower_map["expression"]]].copy()
        out.columns = ["id","name","enabled","applicable_if","expression"]
        # Convert to 3-col while preserving info
        out3 = pd.DataFrame({
            "name":        out["id"].astype(str),
            "description": out["name"].astype(str),
            "expression":  out["expression"].astype(str),
        })
        out3 = out3[out3["expression"].astype(str).str.strip() != ""]
        out3 = out3[out3["name"].str.lower() != "name"]
        return out3.reset_index(drop=True)

    raise ValueError("CSV must be either 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression).")

def fmt_digits_list(xs: List[str]) -> str:
    # show as bare ints, no spaces, preserve provided order
    return "[" + ",".join(str(int(d)) for d in xs) + "]"

# =========================
# Expression Resolver
# =========================
def resolve_expression(expr: str,
                       digit_curr_letters: Dict[str, str],
                       core_letters_prevmap: List[str],
                       prev_core_letters: List[str],
                       cooled_digits: List[str],
                       new_core_digits: List[str],
                       loser_7_9: List[str],
                       ring_digits: List[str],
                       hot7_last10: List[str],
                       hot7_last20: List[str],
                       seed_digits: List[str],
                       prev_digits: List[str]) -> str:
    """
    Replace variables with numeric lists and translate letter-based membership tests.
    """
    out = expr or ""

    # 0) normalize quoted digits into bare ints: '9' -> 9 ; ['4','5'] -> [4,5]
    out = re.sub(r"'([0-9])'", r"\1", out)

    # 1) Replace simple "in <set_var>" with numeric lists
    repl_map = {
        "cooled_digits":   fmt_digits_list(cooled_digits),
        "new_core_digits": fmt_digits_list(new_core_digits),
        "loser_7_9":       fmt_digits_list(loser_7_9),
        "ring_digits":     fmt_digits_list(ring_digits),
        "hot7_last10":     fmt_digits_list(hot7_last10),
        "hot7_last20":     fmt_digits_list(hot7_last20),
        "seed_digits":     fmt_digits_list(seed_digits),
        "prev_digits":     fmt_digits_list(prev_digits),
    }
    for name, lst in repl_map.items():
        # whole-word replacement for "in <name>" and "in(<name>)" patterns
        out = re.sub(rf"\bin\s+{name}\b", " in " + lst, out)

    # 2) Translate: digit_current_letters[d] in ['A','B',...]
    #    → d in [digits whose current letter ∈ {A,B,...}]
    # generic var for digit (d, x, varname...)
    pat = re.compile(
        r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*\[(.*?)\]"
    )
    digits_by_letter_map = digits_by_letter(digit_curr_letters)

    def sub_letter_membership(m: re.Match) -> str:
        var_d = m.group(1)
        raw   = m.group(2)
        letters_raw = [tok.strip() for tok in raw.split(",") if tok.strip()]
        letters = [s.strip("'\"") for s in letters_raw]
        # Build allowed digits set based on current letters
        allowed: List[str] = []
        for d in DIGITS:
            if digit_curr_letters.get(d) in letters:
                allowed.append(d)
        return f"{var_d} in {fmt_digits_list(allowed)}"

    out = pat.sub(sub_letter_membership, out)

    # 3) Replace "'X' in core_letters" / "prev_core_letters" with True/False
    def letter_contains(txt: str, varname: str, letters: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + varname)
        return p.sub(lambda mm: "True" if mm.group(1) in letters else "False", txt)

    out = letter_contains(out, "core_letters", set(core_letters_prevmap))
    out = letter_contains(out, "prev_core_letters", set(prev_core_letters))

    return out

def build_tester_csv_from_paste(pasted_text: str,
                                digit_curr_letters: Dict[str, str],
                                core_letters_prevmap: List[str],
                                prev_core_letters: List[str],
                                cooled_digits: List[str],
                                new_core_digits: List[str],
                                loser_7_9: List[str],
                                ring_digits: List[str],
                                hot7_last10: List[str],
                                hot7_last20: List[str],
                                seed_digits: List[str],
                                prev_digits: List[str]) -> pd.DataFrame:
    df3 = to_three_cols(_read_csv_loose(pasted_text))
    # Resolve expressions row by row
    resolved_expr = [
        resolve_expression(
            expr=r["expression"],
            digit_curr_letters=digit_curr_letters,
            core_letters_prevmap=core_letters_prevmap,
            prev_core_letters=prev_core_letters,
            cooled_digits=cooled_digits,
            new_core_digits=new_core_digits,
            loser_7_9=loser_7_9,
            ring_digits=ring_digits,
            hot7_last10=hot7_last10,
            hot7_last20=hot7_last20,
            seed_digits=seed_digits,
            prev_digits=prev_digits,
        )
        for _, r in df3.iterrows()
    ]
    df3 = df3.copy()
    df3["expression"] = resolved_expr

    # Convert to tester 15-column schema
    out = pd.DataFrame({
        "id":            df3["name"].astype(str),
        "name":          df3["description"].astype(str),
        "enabled":       ["TRUE"] * len(df3),
        "applicable_if": [""     ] * len(df3),
        "expression":    df3["expression"].astype(str),
    })
    for i in range(5, 15):
        out[f"Unnamed: {i}"] = ""
    return out

# =========================
# UI
# =========================
st.title("Loser List (Least → Most Likely) — Tester-ready Export")

with st.sidebar:
    st.header("Inputs")
    pad4 = st.checkbox("Pad 4-digit entries to 5", value=True)
    if st.button("Load example 13"):
        st.session_state["winners_text"] = (
            "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
        )
    if st.button("Load example 20"):
        st.session_state["winners20_text"] = (
            "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,"
            "05847,35515,81921,31406,69018,42735,50319,86420,22345,90112"
        )

with st.form("winners_form"):
    winners_text  = st.text_area("13 winners (Most Recent → Oldest)", key="winners_text", height=120)
    winners20_txt = st.text_area("Optional: Last 20 winners (Most Recent → Oldest)", key="winners20_text", height=100,
                                 help="Needed only for *Last-20 Hot* filters.")
    compute = st.form_submit_button("Compute")

def render_context_panels(info: Dict, last13: List[str], last20_opt: List[str]):
    # Most recent & previous draws
    seed_digits = list(last13[0]) if last13 else []
    prev_digits = list(last13[1]) if len(last13) > 1 else []

    # Core / ring (based on previous map letters)
    core_letters_prevmap = info["core_letters_prevmap"]
    ring_letters = set()
    for L in core_letters_prevmap:
        ring_letters.update(neighbors(L, 1))
    ring_digits = ring_digits_from_letters(ring_letters, info["digit_current_letters"])

    # cooled & new_core derived from rank change
    cooled_digits = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
    new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in core_letters_prevmap]

    # loser_7_9 from ranking (positions 7..9)
    # Build a loser list as sorted by (farther from core first)
    loser_ranking = sorted(
        DIGITS,
        key=lambda d: (
            0 if info["digit_current_letters"][d] in core_letters_prevmap else
            1 if info["digit_current_letters"][d] in ring_letters else
            2,
            info["rank_curr_map"][d]  # colder first
        )
    )
    loser_7_9 = loser_ranking[7:10]

    # hot7_last10 from current heatmap order
    order_curr_str = info.get("current_map_order", "0123456789")
    hot7_last10    = list(order_curr_str[:7])

    # hot7_last20 (optional)
    hot7_last20 = []
    if last20_opt and len(last20_opt) >= 20:
        c = Counter(d for s in last20_opt[:20] for d in s)
        for d in DIGITS:
            c.setdefault(d, 0)
        hot7_last20 = [d for d, _ in c.most_common(7)]

    # Store in session for consistency if needed
    st.session_state["seed_digits"]      = seed_digits
    st.session_state["prev_digits"]      = prev_digits
    st.session_state["loser_7_9"]        = loser_7_9
    st.session_state["ring_digits"]      = ring_digits
    st.session_state["new_core_digits"]  = new_core_digits
    st.session_state["cooled_digits"]    = cooled_digits
    st.session_state["hot7_last10"]      = hot7_last10
    st.session_state["hot7_last20"]      = hot7_last20

    st.subheader("Resolved variables (this run)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**seed_digits**")
        st.code(", ".join(seed_digits) or "∅")
        st.markdown("**prev_digits**")
        st.code(", ".join(prev_digits) or "∅")
        st.markdown("**loser_7_9**")
        st.code(", ".join(loser_7_9) or "∅")
    with c2:
        st.markdown("**ring_digits**")
        st.code(", ".join(ring_digits) or "∅")
        st.markdown("**new_core_digits**")
        st.code(", ".join(new_core_digits) or "∅")
        st.markdown("**cooled_digits**")
        st.code(", ".join(cooled_digits) or "∅")
    with c3:
        st.markdown("**hot7_last10**")
        st.code(", ".join(hot7_last10) or "∅")
        st.markdown("**hot7_last20**")
        st.code(", ".join(hot7_last20) or "∅")
    with c4:
        st.markdown("**digit_current_letters (0–9)**")
        st.dataframe(
            pd.DataFrame({
                "digit": DIGITS,
                "letter": [info["digit_current_letters"][d] for d in DIGITS]
            }),
            hide_index=True, use_container_width=True
        )
        st.markdown("**core_letters (prev map)**")
        st.code(", ".join(core_letters_prevmap) or "∅")

if compute:
    try:
        last13 = parse_winners_text(st.session_state.get("winners_text",""), pad4=st.session_state.get("pad4", True))
        if len(last13) < 13:
            st.error("Please provide at least 13 winners (MR→Oldest).")
        else:
            ranking, info = loser_list(last13)
            st.session_state["info"] = info
            st.session_state["last13"] = last13
            # optional last 20
            last20_opt = parse_winners_text(st.session_state.get("winners20_text",""), pad4=st.session_state.get("pad4", True)) if st.session_state.get("winners20_text","").strip() else []
            st.session_state["last20"] = last20_opt
            render_context_panels(info, last13, last20_opt)
    except Exception as e:
        st.error(str(e))

# Re-render panels if already computed
if "info" in st.session_state and "last13" in st.session_state:
    render_context_panels(st.session_state["info"], st.session_state["last13"], st.session_state.get("last20", []))

    st.markdown("---")
    st.markdown("### Paste Filters (3-col or 5-col)")
    with st.form("csv_form", clear_on_submit=False):
        mega_csv = st.text_area(
            "CSV content",
            key="mega_csv",
            height=220,
            help="Accepts either 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression)."
        )
        build = st.form_submit_button("Build Tester CSV")

    if build:
        try:
            info = st.session_state["info"]
            last13 = st.session_state["last13"]
            last20 = st.session_state.get("last20", [])

            seed_digits = list(last13[0]) if last13 else []
            prev_digits = list(last13[1]) if len(last13) > 1 else []

            core_letters_prevmap = info["core_letters_prevmap"]
            ring_letters = set()
            for L in core_letters_prevmap:
                ring_letters.update(neighbors(L, 1))
            ring_digits = ring_digits_from_letters(ring_letters, info["digit_current_letters"])

            cooled_digits   = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
            new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in core_letters_prevmap]

            order_curr_str = info.get("current_map_order", "0123456789")
            hot7_last10    = list(order_curr_str[:7])

            hot7_last20 = []
            if last20 and len(last20) >= 20:
                c = Counter(d for s in last20[:20] for d in s)
                for d in DIGITS:
                    c.setdefault(d, 0)
                hot7_last20 = [d for d, _ in c.most_common(7)]

            # Simple loser 7-9 from current context (consistent with panel)
            loser_ranking = sorted(
                DIGITS,
                key=lambda d: (
                    0 if info["digit_current_letters"][d] in core_letters_prevmap else
                    1 if info["digit_current_letters"][d] in ring_letters else
                    2,
                    info["rank_curr_map"][d]
                )
            )
            loser_7_9 = loser_ranking[7:10]

            tester_df = build_tester_csv_from_paste(
                pasted_text=mega_csv,
                digit_curr_letters=info["digit_current_letters"],
                core_letters_prevmap=core_letters_prevmap,
                prev_core_letters=core_letters_prevmap,  # prev_core_letters: same idea (prev map of prev draw)
                cooled_digits=cooled_digits,
                new_core_digits=new_core_digits,
                loser_7_9=loser_7_9,
                ring_digits=ring_digits,
                hot7_last10=hot7_last10,
                hot7_last20=hot7_last20,
                seed_digits=seed_digits,
                prev_digits=prev_digits,
            )

            st.markdown("### Tester-ready CSV (copy/paste)")
            csv_text = tester_df.to_csv(index=False)
            st.code(csv_text, language="csv")
            st.download_button(
                "Download tester CSV",
                data=csv_text.encode("utf-8"),
                file_name="filters_for_tester.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(str(e))
else:
    st.info("Enter winners and click **Compute** first.")
