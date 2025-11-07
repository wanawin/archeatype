# streamlit page: "loserlist w filters"
# NOTE: keep this file self-contained; the "tester" app reads the CSV this builds.
from __future__ import annotations

import re
import csv
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any

import streamlit as st

###############################################################################
# Constants / shared “context” helpers
###############################################################################

DIGITS: List[int] = list(range(10))  # 0..9


@dataclass
class PageState:
    seed_1: str = ""
    seed_2: str = ""
    seed_3: str = ""
    seed_4: str = ""
    hot_digits: List[int] = None
    cold_digits: List[int] = None
    due_digits: List[int] = None
    pad_4_to_5: bool = True


def normalize_quotes(s: str) -> str:
    # make sure we’re always working with straight quotes in pasted CSV expressions
    return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")


def parse_digits_text(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    items = [t.strip() for t in s.replace(";", ",").split(",")]
    out: List[int] = []
    for t in items:
        if t == "":
            continue
        if t.isdigit():
            out.append(int(t))
        else:
            # tolerate accidental quotes or whitespace
            t = t.strip("'\" ")
            if t.isdigit():
                out.append(int(t))
    return out


def pad_to_5(d: List[int]) -> List[int]:
    return ([0] * (5 - len(d))) + d if len(d) < 5 else d[:5]


###############################################################################
# “Core letters” mapping and common derived sets (same as your working version)
###############################################################################

# Example core-letter mapping. Keep this exactly as in your current app.
# (This is a fixed mapping you and I have been using; do not modify.)
DIGIT_TO_LETTER: Dict[int, str] = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
}

LETTER_TO_DIGITS: Dict[str, List[int]] = {
    "A": [0], "B": [1], "C": [2], "D": [3], "E": [4],
    "F": [5], "G": [6], "H": [7], "I": [8], "J": [9],
}


def letter_of(d: int) -> str:
    return DIGIT_TO_LETTER.get(int(d), "?")


def fmt_list(xs: List[int]) -> str:
    # Always output square-bracket list with ints: [0,1,2]
    return "[" + ",".join(str(int(v)) for v in xs) + "]"


def compute_context_from_inputs(seed1: str,
                                seed2: str,
                                seed3: str,
                                seed4: str,
                                hot_digits: List[int],
                                cold_digits: List[int],
                                due_digits: List[int],
                                pad_4_to_5_flag: bool) -> Dict[str, Any]:
    """
    Build the evaluation/translation context for the CSV expressions.
    """
    # seed parsing (seed_1 is required)
    def _parse_seed(s: str) -> List[int]:
        s = (s or "").strip()
        if not s:
            return []
        # accept comma-separated or raw “digits”
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
        return [int(ch) for ch in s if ch.isdigit()]

    seed_digits_raw = _parse_seed(seed1)
    if pad_4_to_5_flag and len(seed_digits_raw) == 4:
        seed_digits = pad_to_5(seed_digits_raw)
    else:
        seed_digits = seed_digits_raw[:5]

    # seed+1 set is "unique digits from seed plus +1 unique"
    plus_ones = sorted({(d + 1) % 10 for d in seed_digits})
    p_plus_ones = plus_ones  # same list for p1..p5 replacement when needed

    # carry candidates (top-2) – you already feed these; keep as-is
    # By default just example placeholders; replace in UI before export if needed.
    carry_top2 = []  # [best1, best2] if present

    # union2: you’ve been using this as “top-2 carry ∪ seed+1” (and sometimes more)
    # We’ll keep a single list “union2”; CSV can reference u1..u7 which we expand to this.
    union2 = []  # to be filled by UI if you provide it; kept as a single set/list for expansion

    # union-digits used by UNI2/UNI3 (seed ∪ seed+1). We construct it here.
    union_digits = sorted(set(seed_digits) | set(plus_ones))

    # core-letter sets
    prev_core_letters: List[str] = []
    curr_core_letters: List[str] = []

    # per-digit “current letter” lookup
    digit_current_letters = {d: letter_of(d) for d in DIGITS}

    # some scalars some filters reference
    ctx = {
        "seed_digits": seed_digits,
        "seed_sum": sum(seed_digits),
        "prev_sum": sum(_parse_seed(seed2)) if seed2 else 0,
        "p1": p_plus_ones, "p2": p_plus_ones, "p3": p_plus_ones, "p4": p_plus_ones, "p5": p_plus_ones,
        "carry_top2": carry_top2,
        "union2": union2,
        "union_digits": union_digits,
        "prev_core_letters": prev_core_letters,
        "curr_core_letters": curr_core_letters,
        "digit_current_letters": digit_current_letters,
        "core_size_flags": {},  # keep for compatibility
    }
    return ctx


###############################################################################
# CSV builder (UI)
###############################################################################

st.set_page_config(layout="wide", page_title="loserlist w filters")

st.title("Loser List → 15-column Filter File (Builder)")

st.write("Paste seeds, choose options, then build a 15-column CSV your tester parses.")

with st.expander("Inputs", expanded=True):
    seed1 = st.text_area("Draw 1-back (required):", placeholder="e.g. 40306")
    seed2 = st.text_input("Draw 2-back (optional):", value="")
    seed3 = st.text_input("Draw 3-back (optional):", value="")
    seed4 = st.text_input("Draw 4-back (optional):", value="")

    c1, c2, c3 = st.columns(3)
    with c1:
        pad_4 = st.checkbox("Pad 4-digit seeds to 5 (left-zero)", value=True)
    with c2:
        hot_s = st.text_input("Hot digits (comma-separated):", value="")
    with c3:
        cold_s = st.text_input("Cold digits (comma-separated):", value="")

    due_s = st.text_input("Due digits (comma-separated, optional):", value="")

    st.info("Click **Compute** to populate context for CSV translation.")
    compute_btn = st.button("Compute")

if compute_btn:
    st.success("Computed. Scroll down to CSV builder.")

###############################################################################
# CSV paste → 15-col builder
###############################################################################

st.markdown("### CSV → 15-column Filter File")

csv_paste = st.text_area(
    "Paste filters CSV (3-col or 5-col)",
    height=230,
    placeholder="id,name,enabled,applicable_if,expression,..."
)

def build_tester_csv_from_paste(pasted: str,
                                ctx: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Accept either 3-col (id,name,expression) or 5-col (id,name,enabled,applicable_if,expression)
    and return **15-col** CSV text with your exact header.
    Only translates the expression text; all other columns pass through.
    """
    header_15 = [
        "id","name","enabled","applicable_if","expression",
        "Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9",
        "Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14",
    ]
    out_rows: List[List[str]] = []
    errors: List[str] = []

    pasted = (pasted or "").strip()
    if not pasted:
        return "", ["No columns to parse from file"]

    # Robust sniff for commas; allow stray whitespace
    reader = csv.reader(io.StringIO(pasted))
    for row in reader:
        if not row or all(not (cell or "").strip() for cell in row):
            continue
        cells = [normalize_quotes(c) for c in row]
        # 3-col: id,name,expression
        # 5-col: id,name,enabled,applicable_if,expression
        # Anything longer → we still map into 15 columns and ignore the extra.
        if len(cells) < 3:
            errors.append(f"Row too short: {cells}")
            continue

        if len(cells) == 3:
            f_id, f_name, f_expr = cells
            enabled = "TRUE"
            applicable_if = ""
        else:
            # At least 5 cols → map the first five into the 15-col schema
            f_id = cells[0]
            f_name = cells[1]
            enabled = (cells[2] if len(cells) > 2 and cells[2] != "" else "TRUE")
            applicable_if = (cells[3] if len(cells) > 3 else "")
            f_expr = (cells[4] if len(cells) > 4 else "")

        # translate expression using current context
        translated = resolve_expression(f_expr, ctx)

        out_rows.append([
            f_id, f_name, enabled, applicable_if, translated,
            "", "", "", "", "", "", "", "", "", "",
        ])

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header_15)
    for r in out_rows:
        w.writerow(r)
    return buf.getvalue(), errors


###############################################################################
# Expression translation (ONLY CHANGE IS INSIDE THIS FUNCTION)
###############################################################################

def resolve_expression(expr: str, ctx: Dict) -> str:
    x = (normalize_quotes(expr or "")).strip()

    # --- 0) normalize quoted digits -> bare ints, quoted sets like ['0','1'] -> [0,1]
    x = re.sub(r"'([0-9])'", r"\1", x)

    # --- 1) digit_current_letters[VAR] in ['A','B',...]  ->  VAR in [digits that map to those letters]
    pat_letters_membership = re.compile(
        r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*\[(.*?)\]"
    )
    def sub_letter_membership(m):
        var_d = m.group(1)
        raw   = m.group(2)
        letters_raw = [tok.strip() for tok in raw.split(",") if tok.strip()]
        letters = [s.strip("'\"") for s in letters_raw]
        allowed = [d for d in DIGITS if ctx["digit_current_letters"].get(d) in letters]
        return f"{var_d} in {fmt_list(allowed)}"
    x = pat_letters_membership.sub(sub_letter_membership, x)

    # --- 2) digit_current_letters[VAR] in prev/cur core letters -> VAR in [digits]
    pat_in_prev_core = re.compile(r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*prev_core_letters")
    if ctx.get("prev_core_letters"):
        allowed_prev = [d for d in DIGITS if ctx["digit_current_letters"].get(d) in set(ctx["prev_core_letters"]) ]
        x = pat_in_prev_core.sub(lambda m: f"{m.group(1)} in {fmt_list(allowed_prev)}", x)

    pat_in_cur_core = re.compile(r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*core_letters")
    if ctx.get("curr_core_letters"):
        allowed_cur = [d for d in DIGITS if ctx["digit_current_letters"].get(d) in set(ctx["curr_core_letters"]) ]
        x = pat_in_cur_core.sub(lambda m: f"{m.group(1)} in {fmt_list(allowed_cur)}", x)

    # --- 3) Named digit sets -> concrete lists
    list_vars: Dict[str, List[int]] = {
        # seed digits (padded to 5)
        "s1": ctx.get("seed_digits", [])[:1] or [],
        "s2": ctx.get("seed_digits", [])[1:2] or [],
        "s3": ctx.get("seed_digits", [])[2:3] or [],
        "s4": ctx.get("seed_digits", [])[3:4] or [],
        "s5": ctx.get("seed_digits", [])[4:5] or [],

        # seed+1 set
        "p1": ctx.get("p1", []), "p2": ctx.get("p2", []), "p3": ctx.get("p3", []),
        "p4": ctx.get("p4", []), "p5": ctx.get("p5", []),

        # carry top-2 (c1,c2) and union2 (u1..u7)
        "c1": ctx.get("carry_top2", [])[:1] or [],
        "c2": ctx.get("carry_top2", [])[1:2] or [],
        "u1": ctx.get("union2", []),
        "u2": ctx.get("union2", []),
        "u3": ctx.get("union2", []),
        "u4": ctx.get("union2", []),
        "u5": ctx.get("union2", []),
        "u6": ctx.get("union2", []),
        "u7": ctx.get("union2", []),

        # uppercase UNION_DIGITS (used by UNI2/UNI3)
        "UNION_DIGITS": ctx.get("union_digits", []),
    }
    # replace standalone names and "in NAME" with their list literal
    for name, arr in list_vars.items():
        lit = fmt_list(arr)
        x = re.sub(rf"\bin\s+{name}\b", " in " + lit, x)
        x = re.sub(rf"\b{name}\b", lit, x)

    # --- 4) replace any `sum(1 for d in combo_digits if d in [...])`
    #        with    `sum(int(d) in {...} for d in combo_digits)`
    def _rewrite_sum_membership(text: str) -> str:
        # square brackets -> set
        text = re.sub(
            r"sum\s*\(\s*1\s*for\s+d\s+in\s+combo_digits\s+if\s+d\s+in\s*\[\s*([^\]]*?)\s*\]\s*\)",
            lambda m: f"sum(int(d) in {{{m.group(1)}}} for d in combo_digits)",
            text,
        )
        # parentheses version (after earlier NAME replacement may leave (1,2,3))
        text = re.sub(
            r"sum\s*\(\s*1\s*for\s+d\s+in\s+combo_digits\s+if\s+d\s+in\s*\(\s*([^\)]*?)\s*\)\s*\)",
            lambda m: f"sum(int(d) in {{{m.group(1)}}} for d in combo_digits)",
            text,
        )
        return text
    x = _rewrite_sum_membership(x)

    # --- 5) letter-set membership literals to booleans
    def letter_contains(txt: str, varname: str, letters: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + re.escape(varname))
        return p.sub(lambda mm: "True" if mm.group(1) in letters else "False", txt)

    x = letter_contains(x, "prev_core_letters", set(ctx["prev_core_letters"]))
    x = letter_contains(x, "core_letters",      set(ctx["curr_core_letters"]))

    # --- 6) scalar replacements
    x = re.sub(r"\bseed_sum\b", str(ctx.get("seed_sum", 0)), x)
    x = re.sub(r"\bprev_sum\b", str(ctx.get("prev_sum", 0)), x)

    # --- 7) core-size flags
    for key, val in (ctx.get("core_size_flags") or {}).items():
        x = re.sub(rf"\b{re.escape(key)}\b", "True" if val else "False", x)

    return x


###############################################################################
# UI: Build CSV
###############################################################################

ctx_for_build = compute_context_from_inputs(
    seed1, seed2, seed3, seed4,
    parse_digits_text(hot_s if 'hot_s' in locals() else ""),
    parse_digits_text(cold_s if 'cold_s' in locals() else ""),
    parse_digits_text(due_s if 'due_s' in locals() else ""),
    pad_4 if 'pad_4' in locals() else True,
)

build_btn = st.button("Build 15-column CSV")

if build_btn:
    csv_text, errs = build_tester_csv_from_paste(csv_paste, ctx_for_build)
    if errs:
        st.error("Some rows could not be parsed:")
        for e in errs:
            st.write("• " + e)
    else:
        st.success("CSV built successfully.")
    if csv_text:
        st.download_button(
            "Download 15-col CSV",
            data=csv_text.encode("utf-8"),
            file_name="loserlist_filters15.csv",
            mime="text/csv",
        )
    st.code(csv_text or "", language="csv")
