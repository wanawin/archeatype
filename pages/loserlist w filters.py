# --- BEGIN: digits-only export panel (inline helper) ---
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
            if i - 1 >= 0: neigh.add(LETTERS[i - 1])
            if i + 1 < len(LETTERS): neigh.add(LETTERS[i + 1])
    return sorted([d for d, L in digit_current_letters.items() if L in neigh])

def _robust_csv_to_df(csv_text: str) -> pd.DataFrame:
    """Accepts either 3-col (name,description,expression) or Tester-style CSV with id/name/enabled/.../expression.
    Handles smart quotes, mixed newlines, and common separators; skips truly broken lines."""
    txt = (csv_text or "")
    # normalize quotes/newlines
    txt = (txt.replace("\u201c", '"')
              .replace("\u201d", '"')
              .replace("\u2018", "'")
              .replace("\u2019", "'"))
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    last_err = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(txt),
                sep=sep,
                engine="python",
                quotechar='"',
                escapechar="\\",
                dtype=str,
                on_bad_lines="skip",
            )
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.fillna("")
    cols = set(df.columns)

    def pick(*names):
        for n in names:
            if n in cols:
                return n
        return None

    # If already 3 good columns:
    if {"name", "description", "expression"}.issubset(cols) and "id" not in cols:
        out = df[["name", "description", "expression"]].copy()
        return out[out["expression"].astype(str).str.strip() != ""].reset_index(drop=True)

    # Map Tester-style → 3 columns
    id_col   = pick("id", "code", "filterid")
    name_col = pick("name", "title", "label")
    desc_col = pick("description", "desc", "details")
    expr_col = pick("expression", "expr", "formula")

    if not expr_col:
        raise ValueError("No 'expression' column found. Include a column named 'expression'.")

    name_vals = df[id_col] if id_col else df[name_col] if name_col else pd.Series([""] * len(df))
    desc_vals = (df[name_col] if (id_col and name_col) else
                 df[desc_col] if desc_col else pd.Series([""] * len(df)))

    out = pd.DataFrame({
        "name": name_vals.astype(str).str.strip(),
        "description": desc_vals.astype(str).str.strip(),
        "expression": df[expr_col].astype(str).str.strip(),
    })
    out = out[out["expression"] != ""]
    return out.reset_index(drop=True)

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
        return "[" + ",".join("'{}'".format(d) for d in sorted(list(lst))) + "]"

    # Replace simple set names with digit lists
    out = re.sub(r"\bin\s+cooled_digits\b",   " in {}".format(_fmt(cooled_digits)), out)
    out = re.sub(r"\bin\s+new_core_digits\b", " in {}".format(_fmt(new_core_digits)), out)
    out = re.sub(r"\bin\s+loser_7_9\b",       " in {}".format(_fmt(loser_7_9)), out)
    out = re.sub(r"\bin\s+ring_digits\b",     " in {}".format(_fmt(ring_digits)), out)

    # digit_current_letters[d] in ['A','B',...]  ->  d in ['x','y',...]
    pat = re.compile(r"digit_current_letters\s*\[\s*([a-zA-Z_][\w]*)\s*\]\s*in\s*\[(.*?)\]")
    def _sub(m: re.Match) -> str:
        var_d = m.group(1)
        letters = [tok.strip().strip("'\"") for tok in m.group(2).split(",") if tok.strip()]
        digits = sorted({d for L in letters for d in digits_by_letter.get(L, [])})
        return "{} in [{}]".format(var_d, ",".join("'{}'".format(x) for x in digits))
    out = pat.sub(_sub, out)

    # Replace "'X' in prev_core_letters/core_letters" → True/False
    def _letter_in_set(expr_in: str, varname: str, ref_set: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+{}".format(varname))
        return p.sub(lambda mm: "True" if mm.group(1) in ref_set else "False", expr_in)

    out = _letter_in_set(out, "prev_core_letters", prev_core_letters)
    out = _letter_in_set(out, "core_letters",       prev_core_letters)
    return out

def digits_only_transform(
    csv_text: str,
    digit_current_letters: Dict[str, str],
    digit_prev_letters: Dict[str, str],
    prev_core_letters: Set[str],
    cooled_digits: Set[str],
    new_core_digits: Set[str],
    loser_7_9: List[str],
) -> pd.DataFrame:
    digits_by_letter_curr = _digits_by_letter(digit_current_letters)
    ring = _ring_digits(prev_core_letters, digit_current_letters)
    df_in = _robust_csv_to_df(csv_text)
    rows = []
    for _, row in df_in.iterrows():
        name = str(row.get("name", "")).strip()
        desc = str(row.get("description", "")).strip()
        expr = str(row.get("expression", ""))
        new_expr = _replace_sets(expr, digits_by_letter_curr, prev_core_letters,
                                 cooled_digits, new_core_digits, loser_7_9, ring)
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
    st.subheader("Digits-only Filter Export / Verification Panel")

    # Heatmaps
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Current heatmap (digit → letter)**")
        st.dataframe(pd.DataFrame({"digit": DIGITS, "letter": [digit_current_letters[d] for d in DIGITS]}),
                     use_container_width=True, hide_index=True)
    with colB:
        st.markdown("**Previous heatmap (digit → letter)**")
        st.dataframe(pd.DataFrame({"digit": DIGITS, "letter": [digit_prev_letters[d] for d in DIGITS]}),
                     use_container_width=True, hide_index=True)

    # Sets
    ring = _ring_digits(prev_core_letters, digit_current_letters)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**prev_core_letters**"); st.code(", ".join(sorted(prev_core_letters)) or "∅")
        st.markdown("**loser_7_9 (digits)**"); st.code(", ".join(loser_7_9) or "∅")
    with c2:
        st.markdown("**cooled_digits (digits)**");   st.code(", ".join(sorted(cooled_digits)) or "∅")
        st.markdown("**new_core_digits (digits)**"); st.code(", ".join(sorted(new_core_digits)) or "∅")
    with c3:
        st.markdown("**ring_digits (computed, digits)**"); st.code(", ".join(ring) or "∅")

    st.divider()
    st.markdown("**Digits-only CSV (output)** — copy/paste or download:")
    out_df = digits_only_transform(filters_csv_text, digit_current_letters, digit_prev_letters,
                                   prev_core_letters, cooled_digits, new_core_digits, loser_7_9)
    st.code(out_df.to_csv(index=False), language="csv")
    st.download_button("Download digits-only CSV",
                       data=_df_to_csv_bytes(out_df),
                       file_name="filters_export_digits_only.csv",
                       mime="text/csv")
# --- END: digits-only export panel (inline helper) ---

# ===== Your page logic (unchanged semantics; now form-driven & sticky) =====
from collections import Counter

st.set_page_config(page_title="Loser List (Least → Most Likely)", layout="wide")
st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS: c.setdefault(d, 0)
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
            raise ValueError("Each item must be 5 digits")
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

    tiers = {
        d: (3 if digit_to_letter_curr[d] in core_letters
            else (2 if (digit_to_letter_curr[d] in U and d in due)
                  else (1 if digit_to_letter_curr[d] in U else 0)))
        for d in DIGITS
    }
    ranking = sorted(DIGITS, key=lambda d:(tiers[d], -rank_curr[d], age[d]))
    return ranking, {
        "previous_map_order": "".join(order_prev),
        "current_map_order":  "".join(order_curr),
        "core_letters": core_letters,
        "digit_current_letters": digit_to_letter_curr,
        "digit_prev_letters":    digit_to_letter_prev,
        "rank_curr_map": rank_curr,
        "rank_prev_map": rank_prev,
        "ranking": ranking
    }

# Sidebar
with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries", value=True)
    example_btn = st.button("Load example")

# Winners form (prevents reset while typing)
with st.form("winners_form", clear_on_submit=False):
    if example_btn:
        st.session_state["winners_text"] = "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
    winners_text = st.text_area("13 winners (MR→Oldest)", key="winners_text", height=140)
    compute_clicked = st.form_submit_button("Compute")

if compute_clicked:
    try:
        winners = parse_winners_text(st.session_state["winners_text"], pad4=pad4)[:13]
        ranking, info = loser_list(winners)

        st.session_state["info"] = info
        st.subheader("Loser list (Least → Most Likely)")
        st.code(" ".join(ranking))

        # Derived numeric sets for filters (keep semantics)
        LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
        core_digits      = [LETTER_TO_NUM[L] for L in info["core_letters"]]
        new_core_digits  = [d for d in DIGITS if info["digit_current_letters"][d] not in info["core_letters"]]
        cooled_digits    = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
        loser_7_9        = info["ranking"][7:10]

        st.session_state["core_digits"]     = core_digits
        st.session_state["new_core_digits"] = new_core_digits
        st.session_state["cooled_digits"]   = cooled_digits
        st.session_state["loser_7_9"]       = loser_7_9

        # Always show verification info after Compute
        st.markdown("### Verification: maps and derived sets")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Current heatmap (digit → letter)**")
            st.dataframe(pd.DataFrame({"digit": DIGITS, "letter": [info["digit_current_letters"][d] for d in DIGITS]}),
                         use_container_width=True, hide_index=True)
        with colB:
            st.markdown("**Previous heatmap (digit → letter)**")
            st.dataframe(pd.DataFrame({"digit": DIGITS, "letter": [info["digit_prev_letters"][d] for d in DIGITS]}),
                         use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**prev_core_letters**"); st.code(", ".join(sorted(info["core_letters"])) or "∅")
            st.markdown("**loser_7_9 (digits)**");  st.code(", ".join(loser_7_9) or "∅")
        with c2:
            st.markdown("**cooled_digits (digits)**");   st.code(", ".join(sorted(cooled_digits)) or "∅")
            st.markdown("**new_core_digits (digits)**"); st.code(", ".join(sorted(new_core_digits)) or "∅")
        with c3:
            ring = _ring_digits(set(info["core_letters"]), info["digit_current_letters"])
            st.markdown("**ring_digits (computed, digits)**"); st.code(", ".join(ring) or "∅")

        # Build a small demo CSV (only used if user selects it later)
        filters = [
            ("LL001","Eliminate combos with >=3 digits in [0,9,1,2,4]",
             "sum(1 for d in combo_digits if d in ['0','9','1','2','4']) >= 3"),
            ("LL001A","Eliminate combos with no core digits",
             f"sum(1 for d in combo_digits if d in {core_digits}) == 0"),
            ("LL001B","Eliminate combos with <=2 core digits",
             f"sum(1 for d in combo_digits if d in {core_digits}) <= 2"),
            ("LL002","Eliminate combos with <2 of loser list 7–9",
             f"sum(1 for d in combo_digits if d in {loser_7_9}) < 2"),
            ("LL003","Eliminate combos missing >=3 new-core digits",
             f"sum(1 for d in combo_digits if d in {new_core_digits}) >= 3"),
            ("LL004","Eliminate combos including J(9) unless prev had J",
             "('9' in combo_digits) and not ('9' in seed_digits)"),
            ("LL004R","Eliminate combos with >=2 new-core digits",
             f"sum(1 for d in combo_digits if d in {new_core_digits}) >= 2"),
            ("LL005B","Eliminate combos missing loser list 7–9 entirely",
             f"sum(1 for d in combo_digits if d in {loser_7_9}) == 0"),
            ("LL009","Eliminate if cooled digit repeats",
             f"any(combo_digits.count(d)>1 for d in {cooled_digits})"),
        ]
        csv_lines = ["name,description,expression"]
        for name,desc,expr in filters:
            csv_lines.append(f'{name},"{desc}","{expr}"')
        st.session_state["csv_text_small"] = "\n".join(csv_lines)

    except Exception as e:
        st.error(str(e))

# CSV source form (separate submit; stable while typing)
if "info" in st.session_state:
    st.markdown("### CSV Source for Digits-Only Export")
    with st.form("csv_form", clear_on_submit=False):
        source = st.radio("Choose source", ["Use my MEGA CSV (paste below)", "Use small demo list"], index=0)
        st.text_area("Paste your MEGA CSV (name,description,expression)",
                     key="mega_csv", height=180, value=st.session_state.get("mega_csv",""))
        build_clicked = st.form_submit_button("Build digits-only CSV")

    if build_clicked:
        csv_source_text = (st.session_state.get("mega_csv","").strip()
                           if source.startswith("Use my MEGA")
                           else st.session_state.get("csv_text_small","name,description,expression"))
        render_export_panel(
            filters_csv_text=csv_source_text,
            digit_current_letters=st.session_state["info"]["digit_current_letters"],
            digit_prev_letters=st.session_state["info"]["digit_prev_letters"],
            prev_core_letters=set(st.session_state["info"]["core_letters"]),
            cooled_digits=set(st.session_state["cooled_digits"]),
            new_core_digits=set(st.session_state["new_core_digits"]),
            loser_7_9=list(st.session_state["loser_7_9"]),
        )
else:
    st.info("Enter winners and click **Compute** first. Then you can paste your MEGA CSV and build the digits-only export.")
