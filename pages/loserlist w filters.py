
# loserlist_w_filters_builder.py
# Loser List (Least → Most Likely) — Tester-ready Export (with Heatmaps & Full Loser List restored)

import io
import re
from typing import Dict, List, Set
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

# -----------------------------
# Utilities
# -----------------------------
def normalize_quotes(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return (
        s.replace("\u201c", '"').replace("\u201d", '"')
         .replace("\u2018", "'").replace("\u2019", "'")
         .replace("\r\n", "\n").replace("\r", "\n")
    )

def parse_winners_text(txt: str, pad4: bool = False) -> List[str]:
    txt = normalize_quotes(txt or "")
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

def heat_order(last10: List[List[str]]) -> List[str]:
    c = Counter(d for s in last10 for d in s)
    # Ensure all digits present (even if zero frequency), keep stable int order on ties
    for d in DIGITS:
        c.setdefault(d, 0)
    return [d for d, _ in sorted(c.items(), key=lambda kv: (-kv[1], int(kv[0])))]

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo:hi + 1]

def ring_digits_from_letters(letters: Set[str], digit_current_letters: Dict[str,str]) -> List[str]:
    ring = set()
    for L in letters:
        ring.update(neighbors(L, 1))
    return [d for d in DIGITS if digit_current_letters[d] in ring]

def loser_list(last13: List[str]):
    if len(last13) < 13:
        raise ValueError("Need 13 winners (Most Recent → Oldest).")
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]

    order_prev, order_curr = heat_order(prev10), heat_order(curr10)
    rank_prev,  rank_curr  = rank_of_digit(order_prev), rank_of_digit(order_curr)

    digit_prev_letters = {d: LETTERS[rank_prev[d]  - 1] for d in DIGITS}
    digit_current_letters = {d: LETTERS[rank_curr[d]  - 1] for d in DIGITS}

    most_recent = rows[0]
    core_letters_prevmap = sorted({digit_prev_letters[d] for d in most_recent},
                                  key=lambda L: LETTERS.index(L))

    ring_letters = set()
    for L in core_letters_prevmap:
        ring_letters.update(neighbors(L, 1))

    def tier(d: str) -> int:
        L = digit_current_letters[d]
        if L in core_letters_prevmap: return 0
        if L in ring_letters:          return 1
        return 2

    loser_sorted = sorted(DIGITS, key=lambda d: (tier(d), rank_curr[d]))
    return loser_sorted, {
        "order_prev": order_prev, "order_curr": order_curr,
        "rank_prev": rank_prev, "rank_curr": rank_curr,
        "digit_prev_letters": digit_prev_letters,
        "digit_current_letters": digit_current_letters,
        "core_letters_prevmap": core_letters_prevmap,
        "ring_letters": sorted(ring_letters, key=lambda L: LETTERS.index(L))
    }

# -----------------------------
# CSV helpers (loose reader)
# -----------------------------
def _read_csv_loose(text: str) -> pd.DataFrame:
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

    # 3-col CSV
    if {"name","description","expression"}.issubset(cols) and "id" not in cols:
        out = df[[lower_map["name"], lower_map["description"], lower_map["expression"]]].copy()
        out.columns = ["name","description","expression"]
        out = out[out["expression"].astype(str).str.strip() != ""]
        # Harden header-row stripping
        out["name_norm"] = out["name"].astype(str).str.strip().str.strip('"').str.lower()
        out = out[out["name_norm"] != "name"].drop(columns=["name_norm"])
        return out.reset_index(drop=True)

    # 5-col CSV -> collapse to 3 col
    needed = {"id","name","enabled","applicable_if","expression"}
    if needed.issubset(cols):
        out = df[[lower_map["id"], lower_map["name"], lower_map["enabled"],
                  lower_map["applicable_if"], lower_map["expression"]]].copy()
        out.columns = ["id","name","enabled","applicable_if","expression"]
        out3 = pd.DataFrame({
            "name":        out["id"].astype(str),
            "description": out["name"].astype(str),
            "expression":  out["expression"].astype(str),
        })
        out3 = out3[out3["expression"].astype(str).str.strip() != ""]
        # Harden header-row stripping
        out3["name_norm"] = out3["name"].astype(str).str.strip().str.strip('"').str.lower()
        out3 = out3[out3["name_norm"] != "name"].drop(columns=["name_norm"])
        return out3.reset_index(drop=True)

    raise ValueError("CSV must be 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression).")

def fmt_digits_list(xs: List[str]) -> str:
    return "[" + ",".join(str(int(d)) for d in xs) + "]"

# -----------------------------
# Export context (numbers only)
# -----------------------------
def compute_numeric_context(last13: List[str], last20_opt: List[str]) -> Dict:
    rows = [list(s) for s in last13]
    seed_digits = rows[0]
    prev_digits = rows[1]
    prev2_digits = rows[2] if len(rows) > 2 else []

    curr10 = rows[0:10]
    prev10 = rows[1:11]

    order_curr = heat_order(curr10)
    order_prev = heat_order(prev10)
    rank_curr  = rank_of_digit(order_curr)
    rank_prev  = rank_of_digit(order_prev)

    digit_current_letters = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}
    digit_prev_letters    = {d: LETTERS[rank_prev[d]  - 1] for d in DIGITS}

    # transitions
    trans_FI = [d for d in DIGITS if digit_prev_letters.get(d) == 'F' and digit_current_letters.get(d) == 'I']
    trans_GI = [d for d in DIGITS if digit_prev_letters.get(d) == 'G' and digit_current_letters.get(d) == 'I']

    prev_core_letters = sorted({digit_prev_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))
    curr_core_letters = sorted({digit_current_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))

    prev_core_currentmap_digits = [d for d in DIGITS if digit_current_letters[d] in set(prev_core_letters)]

    cooled_digits   = [d for d in DIGITS if rank_curr[d] > rank_prev[d]]
    new_core_digits = [d for d in DIGITS if digit_current_letters[d] in set(curr_core_letters)]

    loser_7_9 = order_curr[-3:]

    hot7_last10 = order_curr[:7]
    hot7_last20: List[str] = []
    if last20_opt and len(last20_opt) >= 20:
        c = Counter(d for s in last20_opt[:20] for d in s)
        for d in DIGITS:
            c.setdefault(d, 0)
        hot7_last20 = [d for d, _ in c.most_common(7)]

    prev_mirror_digits = sorted({str(MIRROR[int(d)]) for d in prev_digits}, key=int)
    seed_mirror_digits = sorted({str(MIRROR[int(d)]) for d in seed_digits}, key=int)

    union_last2 = sorted(set(prev_digits) | set(prev2_digits), key=int)
    due_last2   = sorted(set(DIGITS) - set(prev_digits) - set(prev2_digits), key=int)

    # Added: derived sets for UNION_DIGITS / UNION2
    seed_plus1  = sorted({str((int(d)+1) % 10) for d in seed_digits}, key=int)
    seed_minus1 = sorted({str((int(d)-1) % 10) for d in seed_digits}, key=int)
    union_digits = sorted(set(seed_digits) | set(seed_plus1), key=int)
    carried_now = [d for d in DIGITS if d in seed_digits and d in prev_digits]
    # choose two lowest current ranks (i.e., hottest now)
    carry_top2  = sorted(carried_now, key=lambda d: rank_curr[d])[:2]
    union2 = sorted(set(carry_top2) | set(seed_plus1), key=int)

    core_size = len(prev_core_letters)
    core_size_flags = {
        "core_size_eq_2":   core_size == 2,
        "core_size_eq_5":   core_size == 5,
        "core_size_in_2_5": core_size in {2,5},
        "core_size_in_235": core_size in {2,3,5},
    }

    seed_sum = sum(int(x) for x in seed_digits)
    prev_sum = sum(int(x) for x in prev_digits)

    # loser ranking (for display section)
    ring_letters_set = set(ring_digits_from_letters(set(prev_core_letters), digit_current_letters))
    def tier_for_display(d: str) -> int:
        L = digit_current_letters[d]
        if L in prev_core_letters: return 0
        if d in ring_letters_set:   return 1
        return 2

    loser_ranking = ", ".join(sorted(DIGITS, key=lambda d: (tier_for_display(d), rank_curr[d])))

    return dict(
        seed_digits=seed_digits, prev_digits=prev_digits, prev2_digits=prev2_digits,
        digit_prev_letters=digit_prev_letters, digit_current_letters=digit_current_letters,
        order_prev=order_prev, order_curr=order_curr,
        rank_prev=rank_prev, rank_curr=rank_curr,
        prev_core_letters=prev_core_letters, curr_core_letters=curr_core_letters,
        cooled_digits=cooled_digits, new_core_digits=new_core_digits,
        ring_digits=sorted(ring_digits_from_letters(set(prev_core_letters), digit_current_letters)),
        loser_7_9=loser_7_9, hot7_last10=hot7_last10, hot7_last20=hot7_last20,
        prev_mirror_digits=prev_mirror_digits, seed_mirror_digits=seed_mirror_digits,
        union_last2=union_last2, due_last2=due_last2, seed_plus1=seed_plus1, seed_minus1=seed_minus1, union_digits=union_digits, carry_top2=carry_top2, union2=union2,
        prev_core_currentmap_digits=prev_core_currentmap_digits,
        core_size_flags=core_size_flags,
        trans_FI=trans_FI, trans_GI=trans_GI,
        seed_sum=seed_sum, prev_sum=prev_sum,
        loser_ranking=loser_ranking
    )

# -----------------------------
# Resolver (numbers only)
# -----------------------------
def resolve_expression(expr: str, ctx: Dict) -> str:
    x = (normalize_quotes(expr or "")).strip()

    # Replace quoted single digits -> bare ints
    x = re.sub(r"'([0-9])'", r"\\1", x)

    # Resolve letter comparisons into booleans
    def eval_letter_eq(txt: str, which: str, letters_map: Dict[str, str]) -> str:
        pat = re.compile(rf"{which}\\s*\\[\\s*([0-9])\\s*\\]\\s*([=!]=)\\s*'([A-J])'")
        def _sub(m):
            d, op, L = m.group(1), m.group(2), m.group(3)
            actual = letters_map.get(d)
            ok = (actual == L)
            return "True" if (ok and op == "==") or ((not ok) and op == "!=") else "False"
        return pat.sub(_sub, txt)

    x = eval_letter_eq(x, "digit_prev_letters",    ctx.get("digit_prev_letters", {}))
    x = eval_letter_eq(x, "digit_current_letters", ctx.get("digit_current_letters", {}))

    # Macro replacements for CSV tokens
    macro_map = {
        "{seed}": "seed_digits",
        "{seed+1}": "seed_plus1",
        "{seed-1}": "seed_minus1",
        "{seed(seed+1)}": "union_digits",
        "{UNION_DIGITS}": "union_digits",
        "{CARRY_TOP2}": "carry_top2",
        "{UNION2}": "union2",
    }
    for _k, _v in macro_map.items():
        x = x.replace(_k, _v)

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
        "seed_plus1":         ctx.get("seed_plus1", []),
        "seed_minus1":        ctx.get("seed_minus1", []),
        "union_digits":       ctx.get("union_digits", []),
        "carry_top2":         ctx.get("carry_top2", []),
        "union2":             ctx.get("union2", []),
        "prev_core_currentmap_digits": ctx["prev_core_currentmap_digits"],
        "trans_FI":           ctx.get("trans_FI", []),
        "trans_GI":           ctx.get("trans_GI", []),
    }
    for name, arr in list_vars.items():
        lit = fmt_digits_list(arr)
        x = re.sub(rf"\\bin\\s+{name}\\b", " in " + lit, x)
        x = re.sub(rf"\\b{name}\\b", lit, x)

    # Numeric scalars
    x = re.sub(r"\\bseed_sum\\b", str(ctx.get("seed_sum", 0)), x)
    x = re.sub(r"\\bprev_sum\\b", str(ctx.get("prev_sum", 0)), x)

    # Boolean flags
    for key, val in (ctx.get("core_size_flags") or {}).items():
        x = re.sub(rf"\\b{re.escape(key)}\\b", "True" if val else "False", x)

    return x

def build_tester_csv_from_paste(pasted_text: str, ctx: Dict) -> pd.DataFrame:
    df3 = to_three_cols(_read_csv_loose(pasted_text))
    resolved_expr = [resolve_expression(expr=r["expression"], ctx=ctx) for _, r in df3.iterrows()]
    out = pd.DataFrame({
        "name": df3["name"],
        "description": df3["description"],
        "expression": resolved_expr
    })
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚡ Loser List → Tester-ready Export")

with st.form("winners_form"):
    winners_text  = st.text_area("13 winners (Most Recent → Oldest)", key="winners_text", height=120)
    winners20_txt = st.text_area("Optional: Last 20 winners (Most Recent → Oldest)", key="winners20_text", height=100,
                                 help="Needed only for *Last-20 Hot* filters.")
    pad4 = st.checkbox("Pad 4-digit inputs to 5 (left zero)", value=True, key="pad4")
    compute = st.form_submit_button("Compute")

def render_context_panels(ctx: Dict, last13: List[str], last20_opt: List[str]):
    seed_digits = ctx["seed_digits"]
    prev_digits = ctx["prev_digits"]

    st.subheader("Resolved variables (this run)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**seed_digits**")
        st.code(", ".join(seed_digits) or "∅")
        st.markdown("**prev_digits**")
        st.code(", ".join(prev_digits) or "∅")
        st.markdown("**prev2_digits**")
        st.code(", ".join(ctx.get("prev2_digits", [])) or "∅")
    with c2:
        st.markdown("**union_last2 (prev ∪ prev2)**")
        st.code(", ".join(ctx["union_last2"]) or "∅")
        st.markdown("**due_last2 (not in prev/prev2)**")
        st.code(", ".join(ctx["due_last2"]) or "∅")
        st.markdown("**seed±1 / unions**")
        st.code(f"seed+1: {', '.join(ctx.get('seed_plus1', [])) or '∅'}")
        st.code(f"seed-1: {', '.join(ctx.get('seed_minus1', [])) or '∅'}")
    with c3:
        st.markdown("**union_digits (seed ∪ seed+1)**")
        st.code(", ".join(ctx.get("union_digits", [])) or "∅")
        st.markdown("**carry_top2 ∪ seed+1 (UNION2)**")
        st.code(", ".join(ctx.get('union2', [])) or "∅")
        st.markdown("**prev_mirror_digits**")
        st.code(", ".join(ctx["prev_mirror_digits"]) or "∅")
    with c4:
        st.markdown("**ring_digits (from prev core)**")
        st.code(", ".join(ctx["ring_digits"]) or "∅")
        st.markdown("**hot7_last10**")
        st.code(", ".join(ctx["hot7_last10"]) or "∅")
        st.markdown("**hot7_last20**")
        st.code(", ".join(ctx["hot7_last20"]) or "∅")

    # Heat maps
    st.markdown("---")
    st.subheader("Heat maps (Prev ↔ Current) + Transitions")
    df_heat = pd.DataFrame({
        "digit": DIGITS,
        "prev_letter": [ctx["digit_prev_letters"][d]   for d in DIGITS],
        "prev_rank":   [ctx["rank_prev"][d]            for d in DIGITS],
        "curr_letter": [ctx["digit_current_letters"][d] for d in DIGITS],
        "curr_rank":   [ctx["rank_curr"][d]            for d in DIGITS],
    })
    st.dataframe(df_heat, use_container_width=True, height=280)

    st.caption(f"trans_FI (prev F → curr I): {', '.join(ctx['trans_FI']) or '∅'}")
    st.caption(f"trans_GI (prev G → curr I): {', '.join(ctx['trans_GI']) or '∅'}")

    # Full Loser List (ordered view)
    st.markdown("---")
    st.subheader("Full Loser List (ordered view)")
    st.code(", ".join(ctx["loser_ranking"]), language="text")

if compute:
    try:
        last13 = parse_winners_text(st.session_state.get("winners_text",""), pad4=st.session_state.get("pad4", True))
        if len(last13) < 13:
            st.error("Please provide at least 13 winners (MR→Oldest).")
        else:
            ranking, info = loser_list(last13)
            last20_opt = parse_winners_text(st.session_state.get("winners20_text",""),
                                            pad4=st.session_state.get("pad4", True)) if st.session_state.get("winners20_text") else []
            st.success("Loser list computed.")
            ctx = compute_numeric_context(last13, last20_opt)
            render_context_panels(ctx, last13, last20_opt)

            # Tester CSV builder
            st.markdown("---")
            st.subheader("CSV → Tester Export")
            mega_csv = st.text_area("Paste filters CSV (3-col or 5-col)", height=200, key="mega_csv")
            try:
                tester_df = build_tester_csv_from_paste(mega_csv, ctx)
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
    except Exception as e:
        st.error(str(e))
else:
    st.info("Enter winners and click **Compute** first.")
