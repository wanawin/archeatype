# digits_export_extension.py
from __future__ import annotations
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
            if i-1 >= 0: neigh.add(LETTERS[i-1])
            if i+1 < len(LETTERS): neigh.add(LETTERS[i+1])
    return sorted([d for d, L in digit_current_letters.items() if L in neigh])

def _csv_to_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text), header=0)

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def _replace_sets(expr: str,
                  digits_by_letter: Dict[str, List[str]],
                  prev_core_letters: Set[str],
                  cooled_digits: Set[str],
                  new_core_digits: Set[str],
                  loser_7_9: List[str],
                  ring_digits: List[str]) -> str:
    out = expr

    def _fmt(lst: List[str] | Set[str]) -> str:
        return "[" + ",".join(f"'{d}'" for d in sorted(list(lst))) + "]"

    # Replace set names used in membership tests with digit lists
    out = re.sub(r"\bin\s+cooled_digits\b",     f"in {_fmt(cooled_digits)}", out)
    out = re.sub(r"\bin\s+new_core_digits\b",   f"in {_fmt(new_core_digits)}", out)
    out = re.sub(r"\bin\s+loser_7_9\b",         f"in {_fmt(loser_7_9)}", out)
    out = re.sub(r"\bin\s+ring_digits\b",       f"in {_fmt(ring_digits)}", out)

    # Replace digit_current_letters[d] in ['A','B',...]  →  d in ['x','y',...]
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

    # Replace `'X' in prev_core_letters` / `core_letters` with True/False booleans
    def _letter_in_set(expr_in: str, varname: str, ref_set: Set[str]) -> str:
        pat = re.compile(rf"'([A-J])'\s+in\s+{varname}")
        def repl(mm: re.Match) -> str:
            return "True" if mm.group(1) in ref_set else "False"
        return pat.sub(repl, expr_in)

    out = _letter_in_set(out, "prev_core_letters", prev_core_letters)
    out = _letter_in_set(out, "core_letters", prev_core_letters)  # same source in your app

    return out

def digits_only_transform(csv_text: str,
                          digit_current_letters: Dict[str, str],
                          digit_prev_letters: Dict[str, str],
                          prev_core_letters: Set[str],
                          cooled_digits: Set[str],
                          new_core_digits: Set[str],
                          loser_7_9: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    digits_by_letter_curr = _digits_by_letter(digit_current_letters)
    ring = _ring_digits(prev_core_letters, digit_current_letters)

    df = _csv_to_df(csv_text)
    transformed_rows = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        desc = str(row.get("description", "")).strip()
        expr = str(row.get("expression", ""))
        new_expr = _replace_sets(expr,
                                 digits_by_letter_curr,
                                 prev_core_letters,
                                 cooled_digits,
                                 new_core_digits,
                                 loser_7_9,
                                 ring)
        transformed_rows.append({"name": name, "description": desc, "expression": new_expr})
    out_df = pd.DataFrame(transformed_rows, columns=["name","description","expression"])

    unresolved = []
    bad_tokens = ["LETTER_TO_NUM", "LETTERS[", "LETTERS.", "digit_current_letters[", "ring_digits",
                  "new_core_digits", "cooled_digits", "loser_7_9", "prev_core_letters", "core_letters"]
    for tok in bad_tokens:
        if any(tok in s for s in out_df["expression"].astype(str)):
            unresolved.append(tok)
    return out_df, unresolved

def render_export_panel(filters_csv_text: str,
                        digit_current_letters: Dict[str, str],
                        digit_prev_letters: Dict[str, str],
                        prev_core_letters: Set[str],
                        cooled_digits: Set[str],
                        new_core_digits: Set[str],
                        loser_7_9: List[str]) -> None:
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

    out_df, unresolved = digits_only_transform(
        csv_in, digit_current_letters, digit_prev_letters,
        prev_core_letters, cooled_digits, new_core_digits, loser_7_9,
    )

    st.markdown("**Digits-only CSV (output)** — copy/paste:")
    st.code(out_df.to_csv(index=False), language="csv")

    st.download_button(
        "Download digits-only CSV",
        data=_df_to_csv_bytes(out_df),
        file_name="filters_export_digits_only.csv",
        mime="text/csv",
    )

    if unresolved:
        st.warning(
            "Unresolved tokens found in expressions → " + ", ".join(sorted(set(unresolved))) +
            "\nThese should not appear; please verify the input CSV."
        )
