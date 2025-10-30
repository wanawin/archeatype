# loserlist_w_filters_builder.py
# Loser List (Least → Most Likely) — Tester-ready Export (numbers-only expressions)

import io, re
from typing import Dict, List, Set
from collections import Counter
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

# ----------------------------- Helpers -----------------------------
def normalize_quotes(text: str) -> str:
    if not text: return ""
    return (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\r\n", "\n").replace("\r", "\n"))

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    txt = normalize_quotes(txt or "")
    tokens = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
    out = []
    for t in tokens:
        if not t.isdigit(): raise ValueError(f"Non-digit token: {t!r}")
        if len(t) == 4 and pad4: t = t.zfill(5)
        if len(t) != 5: raise ValueError("Each entry must be exactly 5 digits.")
        out.append(t)
    return out

def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS: c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter); lo, hi = max(0, i-span), min(9, i+span)
    return LETTERS[lo:hi+1]

def _read_csv_loose(text: str) -> pd.DataFrame:
    text = normalize_quotes(text)
    last_err = None
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                             quotechar='"', escapechar="\\", dtype=str, on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df.fillna("")
        except Exception as e:
            last_err = e
    raise last_err

def to_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(c.lower() for c in df.columns)
    m    = {c.lower(): c for c in df.columns}

    if {"name","description","expression"}.issubset(cols) and "id" not in cols:
        out = df[[m["name"], m["description"], m["expression"]]].copy()
        out.columns = ["name","description","expression"]
        out = out[out["expression"].astype(str).strip() != ""]
        out = out[out["name"].str.lower() != "name"]
        return out.reset_index(drop=True)

    need = {"id","name","enabled","applicable_if","expression"}
    if need.issubset(cols):
        out = df[[m["id"], m["name"], m["enabled"], m["applicable_if"], m["expression"]]].copy()
        out.columns = ["id","name","enabled","applicable_if","expression"]
        out3 = pd.DataFrame({
            "name":        out["id"].astype(str),
            "description": out["name"].astype(str),
            "expression":  out["expression"].astype(str),
        })
        out3 = out3[out3["expression"].astype(str).strip() != ""]
        out3 = out3[out3["name"].str.lower() != "name"]
        return out3.reset_index(drop=True)

    raise ValueError("CSV must be 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression).")

def fmt_digits_list(xs: List[str]) -> str:
    return "[" + ",".join(str(int(d)) for d in xs) + "]"

# ----------------------------- Core mapping (for panels) -----------------------------
def loser_list(last13: List[str]):
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]
    order_prev, order_curr = heat_order(prev10), heat_order(curr10)
    rank_prev,  rank_curr  = rank_of_digit(order_prev), rank_of_digit(order_curr)
    digit_prev_letters  = {d: LETTERS[rank_prev[d]  - 1] for d in DIGITS}
    digit_curr_letters  = {d: LETTERS[rank_curr[d]  - 1] for d in DIGITS}
    most_recent = rows[0]
    core_letters_prevmap = sorted({digit_prev_letters[d] for d in most_recent},
                                  key=lambda L: LETTERS.index(L))
    ring_letters = set()
    for L in core_letters_prevmap: ring_letters.update(neighbors(L, 1))
    def tier(d: str) -> int:
        L = digit_curr_letters[d]
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

def ring_digits_from_letters(ring_letters: Set[str], digit_curr_letters: Dict[str,str]) -> List[str]:
    return [d for d in DIGITS if digit_curr_letters.get(d) in ring_letters]

# ----------------------------- Numeric export context -----------------------------
def compute_numeric_context(last13: List[str], last20_opt: List[str]) -> Dict:
    rows = [list(s) for s in last13]
    seed_digits = rows[0]; prev_digits = rows[1]; prev2_digits = rows[2] if len(rows) > 2 else []

    curr10, prev10 = rows[0:10], rows[1:11]
    order_curr, order_prev = heat_order(curr10), heat_order(prev10)
    rank_curr,  rank_prev  = rank_of_digit(order_curr), rank_of_digit(order_prev)
    digit_current_letters = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}
    digit_prev_letters    = {d: LETTERS[rank_prev[d]  - 1] for d in DIGITS}

    prev_core_letters = sorted({digit_prev_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))
    # digits whose CURRENT letter is in the previous core letters
    prev_core_currentmap_digits = [d for d in DIGITS if digit_current_letters[d] in set(prev_core_letters)]

    # ring
    ring_letters = set()
    for L in prev_core_letters: ring_letters.update(neighbors(L, 1))
    ring_digits = [d for d in DIGITS if digit_current_letters[d] in ring_letters]

    # cooled / new-core for convenience
    cooled_digits   = [d for d in DIGITS if rank_curr[d] > rank_prev[d]]
    curr_core_letters = sorted({digit_current_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))
    new_core_digits = [d for d in DIGITS if digit_current_letters[d] in set(curr_core_letters)]

    # loser 7–9 (cold end of current order)
    loser_ranking = sorted(
        DIGITS,
        key=lambda d: (
            0 if digit_current_letters[d] in prev_core_letters else
            1 if digit_current_letters[d] in ring_letters else
            2,
            rank_curr[d]
        )
    )
    loser_7_9 = loser_ranking[7:10]

    # hot sets
    hot7_last10 = list(order_curr[:7])
    hot7_last20: List[str] = []
    if last20_opt and len(last20_opt) >= 20:
        c = Counter(d for s in last20_opt[:20] for d in s)
        for d in DIGITS: c.setdefault(d, 0)
        hot7_last20 = [d for d, _ in c.most_common(7)]

    # mirrors
    prev_mirror_digits = sorted({str(MIRROR[int(d)]) for d in prev_digits}, key=int)
    seed_mirror_digits = sorted({str(MIRROR[int(d)]) for d in seed_digits}, key=int)

    # union / due (last two draws)
    union_last2 = sorted(set(prev_digits) | set(prev2_digits), key=int)
    due_last2   = sorted(set(DIGITS) - set(prev_digits) - set(prev2_digits), key=int)

    # edge digits (A–C / H–J by CURRENT map)
    edge_AC_digits = [d for d in DIGITS if digit_current_letters[d] in {"A","B","C"}]
    edge_HJ_digits = [d for d in DIGITS if digit_current_letters[d] in {"H","I","J"}]

    # core-size flags
    core_size = len(prev_core_letters)
    core_size_flags = {
        "core_size_eq_2":   core_size == 2,
        "core_size_eq_5":   core_size == 5,
        "core_size_in_2_5": core_size in {2,5},
        "core_size_in_235": core_size in {2,3,5},
    }

    seed_sum = sum(int(x) for x in seed_digits)
    prev_sum = sum(int(x) for x in prev_digits)

    return dict(
        seed_digits=seed_digits, prev_digits=prev_digits, prev2_digits=prev2_digits,
        seed_sum=seed_sum, prev_sum=prev_sum,
        digit_current_letters=digit_current_letters, digit_prev_letters=digit_prev_letters,
        prev_core_letters=prev_core_letters, curr_core_letters=curr_core_letters,
        ring_digits=ring_digits, new_core_digits=new_core_digits, cooled_digits=cooled_digits,
        loser_7_9=loser_7_9, hot7_last10=hot7_last10, hot7_last20=hot7_last20,
        prev_mirror_digits=prev_mirror_digits, seed_mirror_digits=seed_mirror_digits,
        union_last2=union_last2, due_last2=due_last2,
        prev_core_currentmap_digits=prev_core_currentmap_digits,
        edge_AC_digits=edge_AC_digits, edge_HJ_digits=edge_HJ_digits,
        current_map_order="".join(order_curr), previous_map_order="".join(order_prev),
        core_size_flags=core_size_flags,
    )

# ----------------------------- Resolver (numbers only) -----------------------------
def resolve_expression(expr: str, ctx: Dict) -> str:
    x = (normalize_quotes(expr or "")).strip()
    x = re.sub(r"'([0-9])'", r"\1", x)

    # letter equality → True/False
    def eval_letter_eq(txt: str, which: str, letters_map: Dict[str, str]) -> str:
        pat = re.compile(rf"{which}\s*\[\s*([0-9])\s*\]\s*([=!]=)\s*'([A-J])'")
        def _sub(m):
            d, op, L = m.group(1), m.group(2), m.group(3)
            ok = (letters_map.get(d) == L)
            return "True" if (ok and op == "==") or ((not ok) and op == "!=") else "False"
        return pat.sub(_sub, txt)
    x = eval_letter_eq(x, "digit_prev_letters", ctx.get("digit_prev_letters", {}))
    x = eval_letter_eq(x, "digit_current_letters", ctx.get("digit_current_letters", {}))

    # digit_current_letters[var] in ['A','B',...'] → var in [digit list]
    pat_letters_membership = re.compile(r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*\[(.*?)\]")
    def sub_letter_membership(m):
        var_d = m.group(1)
        letters = [tok.strip().strip("'\"") for tok in m.group(2).split(",") if tok.strip()]
        allowed = [d for d in DIGITS if ctx["digit_current_letters"].get(d) in letters]
        return f"{var_d} in {fmt_digits_list(allowed)}"
    x = pat_letters_membership.sub(sub_letter_membership, x)

    # digit_current_letters[var] in prev_core_letters → var in prev_core_currentmap_digits
    pat_in_prev_core = re.compile(r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*prev_core_letters")
    if ctx.get("prev_core_currentmap_digits") is not None:
        allowed = fmt_digits_list(ctx["prev_core_currentmap_digits"])
        x = pat_in_prev_core.sub(lambda m: f"{m.group(1)} in {allowed}", x)

    # Replace list vars (both “in var” and bare “var”)
    list_vars = {
        "cooled_digits":      ctx["cooled_digits"],
        "new_core_digits":    ctx["new_core_digits"],
        "loser_7_9":          ctx["loser_7_9"],
        "ring_digits":        ctx["ring_digits"],
        "hot7_last10":        ctx["hot7_last10"],
        "hot7_last20":        ctx["hot7_last20"],
        "seed_digits":        ctx["seed_digits"],
        "prev_digits":        ctx["prev_digits"],
        "prev_mirror_digits": ctx["prev_mirror_digits"],
        "seed_mirror_digits": ctx["seed_mirror_digits"],
        "union_last2":        ctx["union_last2"],
        "due_last2":          ctx["due_last2"],
        "prev_core_currentmap_digits": ctx["prev_core_currentmap_digits"],
        "edge_AC_digits":     ctx["edge_AC_digits"],
        "edge_HJ_digits":     ctx["edge_HJ_digits"],
    }
    for name, arr in list_vars.items():
        lit = fmt_digits_list(arr)
        x = re.sub(rf"\bin\s+{name}\b", " in " + lit, x)
        x = re.sub(rf"\b{name}\b", lit, x)

    # Replace letter-set contains → True/False
    def letter_contains(txt: str, varname: str, letters: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + re.escape(varname))
        return p.sub(lambda mm: "True" if mm.group(1) in letters else "False", txt)
    x = letter_contains(x, "prev_core_letters", set(ctx["prev_core_letters"]))
    x = letter_contains(x, "core_letters",      set(ctx["curr_core_letters"]))

    # Scalars
    x = re.sub(r"\bseed_sum\b", str(ctx.get("seed_sum", 0)), x)
    x = re.sub(r"\bprev_sum\b", str(ctx.get("prev_sum", 0)), x)

    # Core-size booleans → True/False
    for key, val in (ctx.get("core_size_flags") or {}).items():
        x = re.sub(rf"\b{re.escape(key)}\b", "True" if val else "False", x)
    return x

def build_tester_csv_from_paste(pasted_text: str, ctx: Dict) -> pd.DataFrame:
    df3 = to_three_cols(_read_csv_loose(pasted_text))
    resolved_expr = [resolve_expression(expr=r["expression"], ctx=ctx) for _, r in df3.iterrows()]
    out = pd.DataFrame({
        "id":            df3["name"].astype(str),
        "name":          df3["description"].astype(str),
        "enabled":       ["TRUE"] * len(df3),
        "applicable_if": [""     ] * len(df3),
        "expression":    resolved_expr,
    })
    for i in range(5, 15):
        out[f"Unnamed: {i}"] = ""
    return out

# ----------------------------- UI -----------------------------
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
    seed_digits = list(last13[0]) if last13 else []
    prev_digits = list(last13[1]) if len(last13) > 1 else []
    prev_mirror_digits = [str(MIRROR[int(d)]) for d in prev_digits] if prev_digits else []

    core_letters_prevmap = info["core_letters_prevmap"]
    ring_letters = set()
    for L in core_letters_prevmap: ring_letters.update(neighbors(L, 1))
    ring_digits = ring_digits_from_letters(ring_letters, info["digit_current_letters"])

    cooled_digits   = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
    new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in core_letters_prevmap]

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
    order_curr_str = info.get("current_map_order", "0123456789")
    order_prev_str = info.get("previous_map_order", "0123456789")
    hot7_last10    = list(order_curr_str[:7])

    st.subheader("Resolved variables (this run)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**seed_digits**"); st.code(", ".join(seed_digits) or "∅")
        st.markdown("**prev_digits**"); st.code(", ".join(prev_digits) or "∅")
        st.markdown("**prev_mirror_digits**"); st.code(", ".join(prev_mirror_digits) or "∅")
        st.markdown("**current_map_order**");  st.code(order_curr_str)
        st.markdown("**previous_map_order**"); st.code(order_prev_str)
    with c2:
        st.markdown("**loser_7_9**");     st.code(", ".join(loser_7_9) or "∅")
        st.markdown("**ring_digits**");   st.code(", ".join(ring_digits) or "∅")
        st.markdown("**edge_AC_digits**"); st.code(", ".join([d for d in DIGITS if info['digit_current_letters'][d] in {'A','B','C'}]) or "∅")
        st.markdown("**edge_HJ_digits**"); st.code(", ".join([d for d in DIGITS if info['digit_current_letters'][d] in {'H','I','J'}]) or "∅")
    with c3:
        st.markdown("**new_core_digits**"); st.code(", ".join(new_core_digits) or "∅")
        st.markdown("**cooled_digits**");   st.code(", ".join(cooled_digits) or "∅")
        # Show union/due once we have prev2 in session
        if "union_last2" in st.session_state: st.markdown("**union_last2**"); st.code(", ".join(st.session_state["union_last2"]) or "∅")
        if "due_last2"   in st.session_state: st.markdown("**due_last2**");   st.code(", ".join(st.session_state["due_last2"])   or "∅")
    with c4:
        st.markdown("**hot7_last10**"); st.code(", ".join(hot7_last10) or "∅")
        if "hot7_last20" in st.session_state: st.markdown("**hot7_last20**"); st.code(", ".join(st.session_state["hot7_last20"]) or "∅")
        if "prev_core_currentmap_digits" in st.session_state:
            st.markdown("**prev_core_currentmap_digits**"); st.code(", ".join(st.session_state["prev_core_currentmap_digits"]) or "∅")
        if "core_size_flags" in st.session_state:
            cs = st.session_state["core_size_flags"]
            st.markdown("**core_size_flags**"); st.code(", ".join([f"{k}={cs[k]}" for k in ["core_size_eq_2","core_size_eq_5","core_size_in_2_5","core_size_in_235"]]))

    # keep in session for export
    st.session_state.update({
        "seed_digits": seed_digits,
        "prev_digits": prev_digits,
        "prev_mirror_digits": prev_mirror_digits,
        "loser_7_9": loser_7_9,
        "ring_digits": ring_digits,
        "hot7_last10": hot7_last10,
    })

if compute:
    try:
        last13 = parse_winners_text(st.session_state.get("winners_text",""), pad4=st.session_state.get("pad4", True))
        if len(last13) < 13: st.error("Please provide at least 13 winners (MR→Oldest).")
        else:
            _, info = loser_list(last13)
            st.session_state["info"] = info
            st.session_state["last13"] = last13
            last20_opt = parse_winners_text(st.session_state.get("winners20_text",""),
                                            pad4=st.session_state.get("pad4", True)) if st.session_state.get("winners20_text","").strip() else []
            st.session_state["last20"] = last20_opt

            # also pre-compute numeric export context and cache key pieces for the panel
            ctx = compute_numeric_context(last13, last20_opt)
            for k in ["union_last2","due_last2","hot7_last20","prev_core_currentmap_digits","core_size_flags"]:
                st.session_state[k] = ctx.get(k, [])

            render_context_panels(info, last13, last20_opt)
    except Exception as e:
        st.error(str(e))

# Re-render if already computed
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
            last13 = st.session_state["last13"]
            last20 = st.session_state.get("last20", [])
            ctx = compute_numeric_context(last13, last20)

            tester_df = build_tester_csv_from_paste(pasted_text=mega_csv, ctx=ctx)

            # --- OPTIONAL ready-made rows you said you want available ---
            # These resolve to numbers/booleans by the resolver at export time.
            ready = [
                # CF004b: if core_size == 2, need ≥3 of prev_core_currentmap_digits
                ("CF004b_only2",
                 "ELIMINATE if combo has FEWER THAN 3 of {seed-bucket digits}",
                 "sum(1 for d in combo_digits if d in prev_core_currentmap_digits) < 3"),
                # CF004d: if core_size == 5, need ≥2
                ("CF004d_only5",
                 "ELIMINATE if combo has FEWER THAN 2 of {seed-bucket digits}",
                 "sum(1 for d in combo_digits if d in prev_core_currentmap_digits) < 2"),
                # CF004e: mixed rule — apply threshold by core-size branch
                ("CF004e_2or5",
                 "ELIMINATE if (<2-bucket needs ≥3> OR <5-bucket needs ≥2>) of {seed-bucket digits}",
                 "(core_size_eq_2 and (sum(1 for d in combo_digits if d in prev_core_currentmap_digits) < 3)) or "
                 "(core_size_eq_5 and (sum(1 for d in combo_digits if d in prev_core_currentmap_digits) < 2))"),
                # Edge extremes (A–C / H–J); the lists resolve to numbers
                ("CF005a_run_extremes",
                 "ELIMINATE if combo has ≥3 of {edge-run (A–C) OR (H–J)}",
                 "(sum(1 for d in combo_digits if d in edge_AC_digits) >= 3) or "
                 "(sum(1 for d in combo_digits if d in edge_HJ_digits) >= 3)"),
            ]
            add_df = pd.DataFrame({
                "id": [r[0] for r in ready],
                "name": [r[1] for r in ready],
                "enabled": ["TRUE"]*len(ready),
                "applicable_if": ["" for _ in ready],
                "expression": [r[2] for r in ready],
            })
            for i in range(5, 15): add_df[f"Unnamed: {i}"] = ""

            tester_df = pd.concat([add_df, tester_df], ignore_index=True)

            st.markdown("### Tester-ready CSV (copy/paste)")
            csv_text = tester_df.to_csv(index=False)
            st.code(csv_text, language="csv")
            st.download_button("Download tester CSV",
                               data=csv_text.encode("utf-8"),
                               file_name="filters_for_tester.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(str(e))
else:
    st.info("Enter winners and click **Compute** first.")
