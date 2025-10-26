import io
import re
from typing import Dict, List, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")

# --------------------- helpers ---------------------
def _digits_by_letter(letter_map: Dict[str, str]) -> Dict[str, List[str]]:
    out = {L: [] for L in LETTERS}
    for d, L in letter_map.items():
        out[L].append(d)
    for L in LETTERS:
        out[L] = sorted(out[L])
    return out

def _ring_digits(prev_core_letters: Set[str],
                 digit_current_letters: Dict[str, str]) -> List[str]:
    if not prev_core_letters:
        return []
    neigh = set()
    for L in prev_core_letters:
        if L in LETTERS:
            i = LETTERS.index(L)
            if i - 1 >= 0:
                neigh.add(LETTERS[i - 1])
            if i + 1 < 10:
                neigh.add(LETTERS[i + 1])
    return sorted([d for d, L in digit_current_letters.items() if L in neigh])

def _read_csv_loose(text: str) -> pd.DataFrame:
    t = (text or "")
    # normalize quotes/newlines
    t = (t.replace("\u201c", '"').replace("\u201d", '"')
           .replace("\u2018", "'").replace("\u2019", "'"))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    last = None
    df = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(t),
                sep=sep,
                engine="python",
                quotechar='"',
                escapechar="\\",
                dtype=str,
                on_bad_lines="skip",
            )
            break
        except Exception as e:
            last = e
            df = None
    if df is None:
        raise last
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.fillna("")

def _to_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    # already 3-col
    if {"name", "description", "expression"}.issubset(cols) and "id" not in cols:
        out = df[["name", "description", "expression"]].copy()
        out = out[out["expression"].astype(str).str.strip() != ""]
        return out.reset_index(drop=True)

    # tester 5-col → 3-col
    id_col   = "id" if "id" in cols else None
    name_col = "name" if "name" in cols else None
    desc_col = "description" if "description" in cols else None
    expr_col = "expression" if "expression" in cols else None
    if not expr_col:
        raise ValueError("No 'expression' column found.")

    name_vals = (df[id_col] if id_col else
                 (df[name_col] if name_col else pd.Series([""] * len(df))))
    desc_vals = (df[name_col] if (id_col and name_col) else
                 (df[desc_col] if desc_col else pd.Series([""] * len(df))))

    out = pd.DataFrame({
        "name":        name_vals.astype(str).str.strip(),
        "description": desc_vals.astype(str).str.strip(),
        "expression":  df[expr_col].astype(str).str.strip(),
    })
    out = out[out["expression"] != ""]
    # drop accidental mid-file header row
    out = out[out["name"].str.lower() != "name"]
    return out.reset_index(drop=True)

def _replace_sets(expr: str,
                  digits_by_letter: Dict[str, List[str]],
                  prev_core_letters: Set[str],
                  cooled_digits: Set[str],
                  new_core_digits: Set[str],
                  loser_7_9: List[str],
                  ring_digits: List[str],
                  hot7_last10: List[str]) -> str:
    def fmt(xs):
        return "[" + ",".join("'" + d + "'" for d in sorted(xs)) + "]"

    out = expr

    # Replace named digit-sets with literal digit lists
    out = re.sub(r"\bin\s+cooled_digits\b",   " in " + fmt(cooled_digits), out)
    out = re.sub(r"\bin\s+new_core_digits\b", " in " + fmt(new_core_digits), out)
    out = re.sub(r"\bin\s+loser_7_9\b",       " in " + fmt(loser_7_9), out)
    out = re.sub(r"\bin\s+ring_digits\b",     " in " + fmt(ring_digits), out)
    # NEW: hot7_last10 → literal list of digits
    out = re.sub(r"\bin\s+hot7_last10\b",     " in " + fmt(hot7_last10), out)

    # Translate: digit_current_letters[d] in ['A','B',...]  ->  d in ['x','y',...]
    pat = re.compile(
        r"digit_current_letters\s*\[\s*([a-zA-Z_][\w]*)\s*\]\s*in\s*\[(.*?)\]"
    )
    def _sub(m: re.Match) -> str:
        var_d = m.group(1)
        letters = [tok.strip().strip("'\"") for tok in m.group(2).split(",") if tok.strip()]
        digs = sorted({d for L in letters for d in digits_by_letter.get(L, [])})
        inner = ",".join("'" + x + "'" for x in digs)
        return f"{var_d} in [{inner}]"
    out = pat.sub(_sub, out)

    # Replace "'X' in prev_core_letters/core_letters" → True/False
    def letter_bool(txt: str, varname: str, s: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + varname)
        return p.sub(lambda mm: "True" if mm.group(1) in s else "False", txt)

    out = letter_bool(out, "prev_core_letters", prev_core_letters)
    out = letter_bool(out, "core_letters",       prev_core_letters)
    return out

def digits_only_df(in_csv_text: str,
                   digit_current_letters: Dict[str, str],
                   digit_prev_letters: Dict[str, str],
                   prev_core_letters: Set[str],
                   cooled_digits: Set[str],
                   new_core_digits: Set[str],
                   loser_7_9: List[str],
                   hot7_last10: List[str]) -> pd.DataFrame:
    df_in  = _to_three_cols(_read_csv_loose(in_csv_text))
    byL    = _digits_by_letter(digit_current_letters)
    ring   = _ring_digits(prev_core_letters, digit_current_letters)
    rows = []
    for _, r in df_in.iterrows():
        name = str(r.get("name", "")).strip()
        desc = str(r.get("description", "")).strip()
        expr = str(r.get("expression", ""))

        expr = _replace_sets(
            expr, byL, prev_core_letters,
            cooled_digits, new_core_digits, loser_7_9, ring,
            hot7_last10
        )
        rows.append({"name": name, "description": desc, "expression": expr})

    return pd.DataFrame(rows, columns=["name", "description", "expression"])

def to_tester_schema(df3: pd.DataFrame) -> pd.DataFrame:
    # Map 3-col → 5-col (+ Unnamed: 5..14 to match tester CSVs)
    out = pd.DataFrame({
        "id":            df3["name"],
        "name":          df3["description"],
        "enabled":       ["True"] * len(df3),
        "applicable_if": ["" ]    * len(df3),
        "expression":    df3["expression"],
    })
    for i in range(5, 15):
        out[f"Unnamed: {i}"] = ""
    return out

# ----------------- loser list logic -----------------
def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(L: str, span: int = 1) -> List[str]:
    i = LETTERS.index(L)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo:hi + 1]

def due_set(last_two: List[List[str]]) -> set:
    seen = set(d for r in last_two for d in r)
    return set(DIGITS) - seen

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    toks = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
    out = []
    for t in toks:
        if not t.isdigit():
            raise ValueError(f"Non-digit token: {t!r}")
        if len(t) == 4 and pad4:
            t = t.zfill(5)
        if len(t) != 5:
            raise ValueError("Each item must be 5 digits")
        out.append(t)
    return out

def loser_list(last13: List[str]) -> Tuple[List[str], Dict]:
    if len(last13) < 13:
        raise ValueError("Need 13 winners (MR→Oldest).")
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]

    order_prev, order_curr = heat_order(prev10), heat_order(curr10)
    rank_prev,  rank_curr  = rank_of_digit(order_prev), rank_of_digit(order_curr)

    most_recent = rows[0]
    d2L_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    d2L_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    core_letters = sorted(
        {d2L_prev[d] for d in most_recent},
        key=lambda L: LETTERS.index(L)
    )

    U = set().union(*[neighbors(L, 1) for L in core_letters])
    due = due_set(rows[0:2])

    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):
        for d in DIGITS:
            if age[d] is None and d in r:
                age[d] = back
    for d in DIGITS:
        if age[d] is None:
            age[d] = 9999

    tiers = {
        d: (3 if d2L_curr[d] in core_letters else
            2 if (d2L_curr[d] in U and d in due) else
            1 if d2L_curr[d] in U else 0)
        for d in DIGITS
    }
    ranking = sorted(DIGITS, key=lambda d: (tiers[d], -rank_curr[d], age[d]))
    return ranking, {
        "current_map_order": "".join(order_curr),
        "previous_map_order": "".join(order_prev),
        "core_letters": core_letters,
        "digit_current_letters": d2L_curr,
        "digit_prev_letters":    d2L_prev,
        "rank_curr_map": rank_curr,
        "rank_prev_map": rank_prev,
        "ranking": ranking
    }

# --------------------- UI & panels ---------------------
st.title("Loser List (Least → Most Likely) — Tester-ready Export")

with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries", value=True)
    if st.button("Load example"):
        st.session_state["winners_text"] = (
            "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
        )

with st.form("winners_form", clear_on_submit=False):
    winners_text = st.text_area("13 winners (MR→Oldest)", key="winners_text", height=140)
    compute = st.form_submit_button("Compute")

def render_verification_panels(info: Dict):
    LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

    core_digits     = [str(LETTER_TO_NUM[L]) for L in info["core_letters"]]
    new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in info["core_letters"]]
    cooled_digits   = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
    loser_7_9       = info["ranking"][7:10]

    # NEW: derive Hot-7 from the current 10 draws (most→least hot order)
    order_curr_str = info.get("current_map_order", "0123456789")
    hot7_last10    = list(order_curr_str[:7])

    st.session_state["core_digits"]     = core_digits
    st.session_state["new_core_digits"] = new_core_digits
    st.session_state["cooled_digits"]   = cooled_digits
    st.session_state["loser_7_9"]       = loser_7_9
    st.session_state["hot7_last10"]     = hot7_last10

    st.subheader("Loser list (Least → Most Likely)")
    st.code(" ".join(info["ranking"]))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Current heatmap (digit → letter)**")
        st.dataframe(
            pd.DataFrame({
                "digit": DIGITS,
                "letter": [info["digit_current_letters"][d] for d in DIGITS]
            }),
            use_container_width=True, hide_index=True
        )
    with colB:
        st.markdown("**Previous heatmap (digit → letter)**")
        st.dataframe(
            pd.DataFrame({
                "digit": DIGITS,
                "letter": [info["digit_prev_letters"][d] for d in DIGITS]
            }),
            use_container_width=True, hide_index=True
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**prev_core_letters**")
        st.code(", ".join(sorted(info["core_letters"])) or "∅")
        st.markdown("**loser_7_9 (digits)**")
        st.code(", ".join(loser_7_9) or "∅")
    with c2:
        st.markdown("**cooled_digits (digits)**")
        st.code(", ".join(sorted(cooled_digits)) or "∅")
        st.markdown("**new_core_digits (digits)**")
        st.code(", ".join(sorted(new_core_digits)) or "∅")
    with c3:
        ring = _ring_digits(set(info["core_letters"]), info["digit_current_letters"])
        st.markdown("**ring_digits (digits)**")
        st.code(", ".join(ring) or "∅")
        st.markdown("**hot7_last10 (digits)**")
        st.code(", ".join(hot7_last10) or "∅")

if compute:
    try:
        winners = parse_winners_text(st.session_state["winners_text"], pad4=pad4)[:13]
        ranking, info = loser_list(winners)
        info["ranking"] = ranking
        st.session_state["info"] = info
        render_verification_panels(info)
    except Exception as e:
        st.error(str(e))

# Persist panels & exporter after any successful Compute
if "info" in st.session_state:
    render_verification_panels(st.session_state["info"])

    st.markdown("### CSV Source")
    with st.form("csv_form", clear_on_submit=False):
        st.text_area(
            "Paste MEGA CSV (3-col or 5-col)",
            key="mega_csv",
            height=180,
            value=st.session_state.get("mega_csv", "")
        )
        build = st.form_submit_button("Build Tester CSV")

    if build:
        info = st.session_state["info"]
        tester_ready = digits_only_df(
            in_csv_text=st.session_state.get("mega_csv", ""),
            digit_current_letters=info["digit_current_letters"],
            digit_prev_letters=info["digit_prev_letters"],
            prev_core_letters=set(info["core_letters"]),
            cooled_digits=set(st.session_state["cooled_digits"]),
            new_core_digits=set(st.session_state["new_core_digits"]),
            loser_7_9=list(st.session_state["loser_7_9"]),
            hot7_last10=list(st.session_state.get("hot7_last10", [])),
        )
        tester_df = to_tester_schema(tester_ready)

        st.markdown("### Tester-ready CSV (copy/paste)")
        csv_bytes = tester_df.to_csv(index=False).encode("utf-8")
        st.code(csv_bytes.decode("utf-8"), language="csv")
        st.download_button(
            "Download tester CSV",
            data=csv_bytes,
            file_name="filters_for_tester.csv",
            mime="text/csv"
        )
else:
    st.info("Enter winners and click **Compute** first.")
