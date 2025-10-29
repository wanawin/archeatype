# loserlist_w_filters.py  — Loser List → Tester Export (fixed letter tests, core sizes, mirrors)

import io
import re
from typing import Dict, List, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

# ───────────────────────────────── helpers ─────────────────────────────────

def normalize_quotes(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\r\n", "\n").replace("\r", "\n"))

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    txt = normalize_quotes(txt)
    toks = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
    out  = []
    for t in toks:
        if not t.isdigit():
            raise ValueError(f"Non-digit token: {t!r}")
        if len(t) == 4 and pad4:
            t = t.zfill(5)
        if len(t) != 5:
            raise ValueError("Each entry must be exactly 5 digits.")
        out.append(t)
    return out

def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS: c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_map(order: List[str]) -> Dict[str,int]:
    return {d:i+1 for i,d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i-span), min(9, i+span)
    return LETTERS[lo:hi+1]

def loser_list(last13: List[str]) -> Tuple[List[str], Dict]:
    if len(last13) < 13:
        raise ValueError("Need 13 winners (Most Recent → Oldest).")
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]

    order_prev = heat_order(prev10)
    order_curr = heat_order(curr10)
    rprev      = rank_map(order_prev)
    rcurr      = rank_map(order_curr)

    digit_prev_letters = {d: LETTERS[rprev[d]-1] for d in DIGITS}
    digit_curr_letters = {d: LETTERS[rcurr[d]-1] for d in DIGITS}

    most_recent = rows[0]
    core_letters_prevmap = sorted({digit_prev_letters[d] for d in most_recent},
                                  key=lambda L: LETTERS.index(L))

    ring_letters = set()
    for L in core_letters_prevmap: ring_letters.update(neighbors(L, 1))

    # simple tiering
    def tier(d: str) -> int:
        L = digit_curr_letters[d]
        if L in core_letters_prevmap: return 3
        if L in ring_letters:         return 2
        return 1

    ranking = sorted(DIGITS, key=lambda d: (tier(d), -rcurr[d], d))
    return ranking, {
        "current_map_order":  "".join(order_curr),
        "previous_map_order": "".join(order_prev),
        "digit_current_letters": digit_curr_letters,
        "digit_prev_letters":    digit_prev_letters,
        "core_letters_prevmap":  core_letters_prevmap,
        "ring_letters":          sorted(ring_letters, key=lambda L: LETTERS.index(L)),
        "rank_curr_map": rcurr,
        "rank_prev_map": rprev,
    }

def ring_digits_from_letters(ring_letters: Set[str],
                             digit_curr_letters: Dict[str,str]) -> List[str]:
    return [d for d in DIGITS if digit_curr_letters.get(d) in ring_letters]

def _read_csv_loose(text: str) -> pd.DataFrame:
    text = normalize_quotes(text)
    last_err = None
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                             quotechar='"', escapechar="\\", dtype=str,
                             on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df.fillna("")
        except Exception as e:
            last_err = e
    raise last_err

def to_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower():c for c in df.columns}
    low  = set(cols)

    # 3-col: name,description,expression
    if {"name","description","expression"}.issubset(low) and "id" not in low:
        out = df[[cols["name"], cols["description"], cols["expression"]]].copy()
        out.columns = ["name","description","expression"]
        out = out[out["expression"].astype(str).str.strip()!=""]
        out = out[out["name"].str.lower()!="name"]
        return out.reset_index(drop=True)

    # 5-col: id,name,enabled,applicable_if,expression
    need = {"id","name","enabled","applicable_if","expression"}
    if need.issubset(low):
        out = df[[cols["id"], cols["name"], cols["enabled"], cols["applicable_if"], cols["expression"]]].copy()
        out.columns = ["id","name","enabled","applicable_if","expression"]
        out3 = pd.DataFrame({
            "name":        out["id"].astype(str),
            "description": out["name"].astype(str),
            "expression":  out["expression"].astype(str),
        })
        out3 = out3[out3["expression"].astype(str).str.strip()!=""]
        out3 = out3[out3["name"].str.lower()!="name"]
        return out3.reset_index(drop=True)

    raise ValueError("CSV must be 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression).")

def fmt_list(xs: List[str]) -> str:
    return "[" + ","join(str(int(d)) for d in xs) + "]"
