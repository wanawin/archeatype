# Loser List → Tester Export (fixed)
# Complete Streamlit page. Paste winners + paste filters CSV (3-col or 5-col) →
# emits tester‑ready 15‑column CSV with all dynamic variables inlined.

from __future__ import annotations
import io
import re
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List — Tester Export (Fixed)", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}

# ───────────────────────────── small utils ─────────────────────────────

def normalize_quotes(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\r\n", "\n").replace("\r", "\n"))


def parse_winners_text(txt: str) -> List[str]:
    """Accept comma/space/newline separated 5‑digit results; MR first."""
    txt = normalize_quotes(txt)
    raw = re.split(r"[\s,]+", txt.strip()) if txt.strip() else []
    out: List[str] = []
    for t in raw:
        if not t:
            continue
        if not t.isdigit():
            raise ValueError(f"Non‑numeric token: {t!r}")
        if len(t) != 5:
            raise ValueError(f"Each entry must be 5 digits (got {t!r}).")
        out.append(t)
    return out


def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    # hottest first (desc freq), tie‑break by digit
    return sorted(DIGITS, key=lambda d: (-c[d], d))


def rank_map(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}


def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo : hi + 1]


def loser_pipeline(last13: List[str]) -> Tuple[Dict, Dict]:
    """Compute all dynamic variable sets from the last 13 winners (MR→Oldest)."""
    if len(last13) < 13:
        raise ValueError("Need 13 winners (most recent first).")
    rows = [list(s) for s in last13]
    most_recent = rows[0]

    prev10 = rows[1:11]
    curr10 = rows[0:10]

    order_prev = heat_order(prev10)
    order_curr = heat_order(curr10)
    rprev      = rank_map(order_prev)
    rcurr      = rank_map(order_curr)

    digit_prev_letters = {d: LETTERS[rprev[d] - 1] for d in DIGITS}
    digit_curr_letters = {d: LETTERS[rcurr[d] - 1] for d in DIGITS}

    # Core letters (previous map) for the most‑recent seed digits
    prev_core_letters = sorted({digit_prev_letters[d] for d in most_recent},
                               key=lambda L: LETTERS.index(L))
    # Ring letters: ±1 around each prev core letter
    ring_letters = sorted({L for C in prev_core_letters for L in neighbors(C, 1)},
                          key=lambda L: LETTERS.index(L))

    # Numeric sets derived from letters/maps
    def digits_with_letters(letters: List[str]) -> List[str]:
        return [d for d in DIGITS if digit_curr_letters[d] in letters]

    ring_digits = [d for d in DIGITS if digit_curr_letters[d] in set(ring_letters)]

    # "loser_7_9": the three *coldest* digits in the current 10 (positions 8–10)
    loser_7_9 = order_curr[-3:]

    # Top‑7 hottest in the last 10
    hot7_last10 = order_curr[:7]

    # naive cooled digits example (bottom‑3 in previous map)
    cooled_digits = order_prev[-3:]

    # convenience sizes used by some filters
    core_size_prev    = len(prev_core_letters)
    core_size_current = len({digit_curr_letters[d] for d in most_recent})

    dyn = dict(
        seed_digits=list(most_recent),
        prev_digits=list(rows[1]),
        loser_7_9=loser_7_9,
        hot7_last10=hot7_last10,
        ring_digits=ring_digits,
        cooled_digits=cooled_digits,
        prev_core_letters=prev_core_letters,
        core_letters=sorted({digit_curr_letters[d] for d in most_recent}, key=lambda L: LETTERS.index(L)),
        core_size_prev=core_size_prev,
        core_size_current=core_size_current,
        digit_prev_letters=digit_prev_letters,
        digit_current_letters=digit_curr_letters,
        order_prev=order_prev,
        order_curr=order_curr,
    )

    meta = dict(
        most_recent="".join(most_recent),
        order_prev="".join(order_prev),
        order_curr="".join(order_curr),
    )
    return dyn, meta


def read_csv_loose(text: str) -> pd.DataFrame:
    text = normalize_quotes(text)
    last_err = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                             quotechar='"', escapechar='\\', dtype=str,
                             on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df.fillna("")
        except Exception as e:
            last_err = e
    raise last_err


def to_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    low  = set(cols)

    if {"name","description","expression"}.issubset(low) and "id" not in low:
        out = df[[cols["name"], cols["description"], cols["expression"]]].copy()
        out.columns = ["name","description","expression"]
        return out[out["expression"].astype(str).str.strip() != ""].reset_index(drop=True)

    need = {"id","name","enabled","applicable_if","expression"}
    if need.issubset(low):
        part = df[[cols["id"], cols["name"], cols["enabled"], cols["applicable_if"], cols["expression"]]].copy()
        part.columns = ["id","name","enabled","applicable_if","expression"]
        out3 = pd.DataFrame({
            "name":        part["id"].astype(str),
            "description": part["name"].astype(str),
            "expression":  part["expression"].astype(str),
        })
        out3 = out3[out3["expression"].astype(str).str.strip() != ""].reset_index(drop=True)
        return out3

    raise ValueError("CSV must be 3‑col (name,description,expression) or 5‑col (id,name,enabled,applicable_if,expression).")


def fmt_list(xs: List[str]) -> str:
    # show as [9,8,2]
    return "[" + ",".join(str(int(d)) for d in xs) + "]"


# ───────────────────────── expression resolver ─────────────────────────

def eval_letter_eq(txt: str, varname: str, letters_map: Dict[str,str]) -> str:
    # digit_prev_letters['8'] == 'F'  or  digit_current_letters["3"] != 'B'
    pat = re.compile(rf"{varname}\\s*\\[\\s*'?(?P<d>[0-9])'?\\s*\\]\\s*(?P<op>==|!=)\\s*'(?P<L>[A-J])'")
    def _sub(m: re.Match) -> str:
        d  = m.group("d")
        op = m.group("op")
        L  = m.group("L")
        val = (letters_map.get(d) == L)
        return str(val if op == "==" else (not val))
    return pat.sub(_sub, txt)


def resolve_expression(expr: str, dyn: Dict) -> str:
    x = normalize_quotes(expr)

    # Replace list‑valued variables with numeric lists
    repl_lists = {
        "loser_7_9":        fmt_list(dyn["loser_7_9"]),
        "hot7_last10":      fmt_list(dyn["hot7_last10"]),
        "ring_digits":      fmt_list(dyn["ring_digits"]),
        "cooled_digits":    fmt_list(dyn["cooled_digits"]),
        "seed_digits":      fmt_list(dyn["seed_digits"]),
        "prev_digits":      fmt_list(dyn["prev_digits"]),
    }
    for k, v in repl_lists.items():
        x = re.sub(rf"\b{k}\b", v, x)

    # len(core_letters) and len(prev_core_letters)
    x = re.sub(r"\blen\(\s*core_letters\s*\)",        str(dyn.get("core_size_current", 0)), x)
    x = re.sub(r"\blen\(\s*prev_core_letters\s*\)",   str(dyn.get("core_size_prev", 0)),    x)

    # equality/inequality tests on letter maps
    x = eval_letter_eq(x, "digit_prev_letters",    dyn["digit_prev_letters"])
    x = eval_letter_eq(x, "digit_current_letters", dyn["digit_current_letters"])

    return x


def build_tester_csv(df_three: pd.DataFrame, dyn: Dict) -> pd.DataFrame:
    # 15‑column tester schema
    cols15 = [
        "id","name","enabled","applicable_if","expression",
        "Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9",
        "Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14",
    ]
    rows = []
    for _, r in df_three.iterrows():
        name = str(r["name"]).strip()
        desc = str(r["description"]).strip()
        exp  = str(r["expression"]).strip()
        if not name or not exp:
            continue
        exp_resolved = resolve_expression(exp, dyn)
        rows.append([
            name, desc, "True", "True", exp_resolved,
            "", "", "", "", "", "", "", "", "", ""
        ])
    out = pd.DataFrame(rows, columns=cols15)
    return out


# ─────────────────────────────── UI ───────────────────────────────

st.title("Loser List (Least → Most Likely) — Tester‑ready Export")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    winners_txt = st.text_area(
        "Paste last 13 winners (most recent first)",
        height=160,
        placeholder="27500\n28825\n43769\n… (13 total)")
with c2:
    st.markdown("**Paste Filters (3‑col or 5‑col)**")
    csv_in = st.text_area("CSV content", height=240,
                          placeholder="name,description,expression\nLL002,Eliminate if loser 7–9 count == 5, sum(1 for d in combo_digits if d in loser_7_9) == 5")
with c3:
    st.info("This page replaces dynamic variables (loser_7_9, ring_digits, hot7_last10, seed/prev digits, len(core_letters), etc.) with **actual numbers** for the tester app.")

if winners_txt.strip() and csv_in.strip():
    try:
        last13 = parse_winners_text(winners_txt)
        dyn, meta = loser_pipeline(last13)

        st.subheader("Derived variables (this run)")
        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("**loser_7_9 (digits)**")
            st.code(", ".join(dyn["loser_7_9"]))
            st.markdown("**hot7_last10 (digits)**")
            st.code(", ".join(dyn["hot7_last10"]))
        with cB:
            st.markdown("**ring_digits (digits)**")
            st.code(", ".join(dyn["ring_digits"]))
            st.markdown("**cooled_digits (digits)**")
            st.code(", ".join(dyn["cooled_digits"]))
        with cC:
            st.markdown("**digit_prev_letters → example**")
            st.code({k: dyn["digit_prev_letters"][k] for k in sorted(dyn["digit_prev_letters"])})

        df_in  = read_csv_loose(csv_in)
        three  = to_three_cols(df_in)
        out15  = build_tester_csv(three, dyn)

        st.subheader("Tester‑ready CSV (copy/paste)")
        st.dataframe(out15.head(50), use_container_width=True)
        csv_bytes = out15.to_csv(index=False).encode("utf-8")
        st.download_button("Download tester CSV", data=csv_bytes, file_name="filters_tester_ready.csv", mime="text/csv")

    except Exception as e:
        st.error(str(e))
else:
    st.caption("Paste 13 winners and your filters to build the export.")
