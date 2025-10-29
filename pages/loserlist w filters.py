# Loser List → Tester Export (Builder)
# COMPLETE FILE — UI unchanged from prior builder: left winners box + filters CSV box + table + download.
# Adds: prev_mirror_digits, union_last2, due_last2, hot7_last20, seed_sum/prev_sum, and
# resolves ALL variables to numeric/boolean literals before export (tester has zero symbols left).

from __future__ import annotations
import io
import re
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

# ───────────────────────────── App config ─────────────────────────────
st.set_page_config(page_title="Loser List — Tester Export (Builder)", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}

# ───────────────────────────── Utilities ─────────────────────────────

def normalize_quotes(text: str) -> str:
    if not text:
        return ""
    return (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\r\n", "\n").replace("\r", "\n"))

def parse_winners_text(txt: str, need: int = 13) -> List[str]:
    """Parse comma/space/newline separated 5‑digit results; MR first."""
    txt = normalize_quotes(txt)
    raw = re.split(r"[\s,]+", txt.strip()) if txt.strip() else []
    out: List[str] = []
    for t in raw:
        if not t:
            continue
        if not t.isdigit() or len(t) != 5:
            raise ValueError(f"Each entry must be a 5‑digit number (got {t!r}).")
        out.append(t)
    if len(out) < need:
        raise ValueError(f"Need {need} winners (most recent first); got {len(out)}.")
    return out[:need]

def heat_order(block: List[List[str]]) -> List[str]:
    c = Counter(d for r in block for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))  # hottest → coldest

def rank_map(order: List[str]) -> Dict[str, str]:
    # return letter by digit (A hottest)
    return {d: LETTERS[i] for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo:hi+1]

def fmt_list(xs: List[str]) -> str:
    return "[" + ",".join(str(int(d)) for d in xs) + "]"

# ───────────────────── Variable derivation (builder only) ─────────────────────

def derive_variables(last13: List[str], last20: List[str] | None = None) -> Tuple[Dict, Dict]:
    rows13 = [list(s) for s in last13]  # MR → older
    most_recent = rows13[0]
    prev        = rows13[1]

    # Heatmaps (current: last10 incl MR; previous: the 10 before MR)
    curr10 = rows13[0:10]
    prev10 = rows13[1:11]
    order_curr = heat_order(curr10)
    order_prev = heat_order(prev10)

    digit_curr_letters = rank_map(order_curr)  # {'0': 'C', ...}
    digit_prev_letters = rank_map(order_prev)

    prev_core_letters = sorted({digit_prev_letters[d] for d in most_recent}, key=lambda L: LETTERS.index(L))
    # core letters for current map (used by some size checks)
    curr_core_letters = sorted({digit_curr_letters[d] for d in most_recent}, key=lambda L: LETTERS.index(L))

    # ring: ±1 around each previous core letter (by alphabet index)
    ring_letters = sorted({L for C in prev_core_letters for L in neighbors(C, 1)}, key=lambda L: LETTERS.index(L))
    ring_digits  = sorted([d for d in DIGITS if digit_curr_letters[d] in set(ring_letters)], key=int)

    # loser_7_9 = 3 coldest digits in current 10
    loser_7_9 = order_curr[-3:]

    # cooled = 3 coldest in previous 10 (example definition used in prior runs)
    cooled_digits = order_prev[-3:]

    # new_core_digits (current letters for MR seed digits)
    new_core_digits = sorted({d for d in DIGITS if digit_curr_letters[d] in set(curr_core_letters)}, key=int)

    # hot7 sets
    hot7_last10 = order_curr[:7]
    hot7_last20 = []
    if last20 and len(last20) >= 20:
        block20 = [list(s) for s in last20[:20]]
        hot7_last20 = [d for d,_ in Counter(d for r in block20 for d in r).most_common(7)]

    seed_digits = most_recent
    prev_digits = prev
    prev2_digits = rows13[2] if len(rows13) > 2 else []

    # sums
    seed_sum = sum(int(x) for x in seed_digits)
    prev_sum = sum(int(x) for x in prev_digits)

    # union/due over last 2 draws
    union_last2 = sorted(set(prev_digits) | set(prev2_digits), key=int)
    due_last2   = sorted(set(DIGITS) - set(prev_digits) - set(prev2_digits), key=int)

    # mirrors
    prev_mirror_digits = sorted({str(MIRROR[int(d)]) for d in prev_digits}, key=int)
    seed_mirror_digits = sorted({str(MIRROR[int(d)]) for d in seed_digits}, key=int)

    dyn = dict(
        seed_digits=seed_digits,
        prev_digits=prev_digits,
        prev2_digits=prev2_digits,
        seed_sum=seed_sum,
        prev_sum=prev_sum,
        union_last2=union_last2,
        due_last2=due_last2,
        prev_mirror_digits=prev_mirror_digits,
        seed_mirror_digits=seed_mirror_digits,
        digit_curr_letters=digit_curr_letters,
        digit_prev_letters=digit_prev_letters,
        prev_core_letters=prev_core_letters,
        curr_core_letters=curr_core_letters,
        ring_digits=ring_digits,
        loser_7_9=loser_7_9,
        cooled_digits=cooled_digits,
        new_core_digits=new_core_digits,
        hot7_last10=hot7_last10,
        hot7_last20=hot7_last20,
    )

    meta = dict(order_curr="".join(order_curr), order_prev="".join(order_prev))
    return dyn, meta

# ───────────────────────── CSV readers/writers ─────────────────────────

def read_csv_loose(text: str) -> pd.DataFrame:
    text = normalize_quotes(text)
    last_err = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", quotechar='"', escapechar='\\', dtype=str, on_bad_lines="skip")
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

# ───────────────────────── Expression resolver ─────────────────────────

def _bool_from_letter_eq(dmap: Dict[str,str], txt: str, var: str) -> str:
    # replace: digit_prev_letters['8'] == 'F'  →  True
    pat = re.compile(rf"{var}\\s*\\[\\s*'(?P<d>\d)'\\s*\\]\\s*==\\s*'(?P<L>[A-J])'")
    return pat.sub(lambda m: str(dmap.get(m.group('d')) == m.group('L')), txt)


def resolve_expression(expr: str, dyn: Dict) -> str:
    x = normalize_quotes(expr)

    # Inline set/list variables with numeric literals
    list_vars = {
        "cooled_digits":      dyn["cooled_digits"],
        "new_core_digits":    dyn["new_core_digits"],
        "loser_7_9":          dyn["loser_7_9"],
        "ring_digits":        dyn["ring_digits"],
        "hot7_last10":        dyn["hot7_last10"],
        "hot7_last20":        dyn["hot7_last20"],
        "seed_digits":        dyn["seed_digits"],
        "prev_digits":        dyn["prev_digits"],
        "union_last2":        dyn["union_last2"],
        "due_last2":          dyn["due_last2"],
        "prev_mirror_digits": dyn["prev_mirror_digits"],
        "seed_mirror_digits": dyn["seed_mirror_digits"],
    }

    for name, vals in list_vars.items():
        literal = fmt_list(vals)
        # support both "in var" and bare occurrences
        x = re.sub(rf"\bin\s+{name}\b", " in " + literal, x)
        x = re.sub(rf"\b{re.escape(name)}\b", literal, x)

    # Inline scalars
    x = re.sub(r"\bseed_sum\b", str(dyn["seed_sum"]), x)
    x = re.sub(r"\bprev_sum\b", str(dyn["prev_sum"]), x)

    # Letter equality to boolean
    x = _bool_from_letter_eq(dyn["digit_prev_letters"], x,  "digit_prev_letters")
    x = _bool_from_letter_eq(dyn["digit_curr_letters"], x,  "digit_current_letters")

    # len(core_letters) variants → actual numbers
    x = re.sub(r"len\(\s*prev_core_letters\s*\)",  str(len(dyn["prev_core_letters"])),  x)
    x = re.sub(r"len\(\s*core_letters\s*\)",        str(len(dyn["curr_core_letters"])), x)

    return x


def build_tester_csv(df_three: pd.DataFrame, dyn: Dict) -> pd.DataFrame:
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
        rows.append([name, desc, "True", "True", exp_resolved, "", "", "", "", "", "", "", "", "", ""])
    return pd.DataFrame(rows, columns=cols15)

# ─────────────────────────────── UI (unchanged) ───────────────────────────────

st.title("Loser List (Least → Most Likely) — Tester‑ready Export")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    winners_txt = st.text_area(
        "Paste last 13 winners (most recent first)",
        height=160,
        placeholder="27500\n28825\n43769\n… (13 total)")
    # OPTIONAL: last‑20 input (no layout change — same column, under winners)
    last20_txt = st.text_area("(Optional) Paste last 20 winners (MR first)", height=120, placeholder="Paste 20 if you want hot7_last20; else leave blank")
with c2:
    st.markdown("**Paste Filters (3‑col or 5‑col)**")
    csv_in = st.text_area("CSV content", height=280,
                          placeholder="name,description,expression\nLL002,Eliminate if loser 7–9 count == 5, sum(1 for d in combo_digits if d in loser_7_9) == 5")
with c3:
    st.info("Builder resolves variables to numbers (loser_7_9, ring_digits, hot7_last10/20, union_last2, due_last2, mirrors, sums, letter checks). Tester receives only concrete expressions.")

if winners_txt.strip() and csv_in.strip():
    try:
        last13 = parse_winners_text(winners_txt, need=13)
        last20 = None
        if last20_txt.strip():
            last20 = parse_winners_text(last20_txt, need=20)
        dyn, meta = derive_variables(last13, last20)

        df_in  = read_csv_loose(csv_in)
        three  = to_three_cols(df_in)
        out15  = build_tester_csv(three, dyn)

        st.subheader("Tester‑ready CSV (copy/paste)")
        st.dataframe(out15.head(50), use_container_width=True)
        st.download_button("Download tester CSV", data=out15.to_csv(index=False).encode("utf-8"), file_name="filters_tester_ready.csv", mime="text/csv")

    except Exception as e:
        st.error(str(e))
else:
    st.caption("Paste 13 winners and your filters to build the export. (Optional: paste 20 to enable hot7_last20.)")
