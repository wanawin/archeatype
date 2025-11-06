# Loser List (Least → Most Likely) — Tester-ready Export
# (with heat maps, loser lists, and numbers-only resolver)

import io, re
from typing import Dict, List, Set, Tuple
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loser List → Tester Export", layout="wide")

LETTERS = list("ABCDEFGHIJ")
DIGITS  = list("0123456789")
MIRROR  = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}

# -----------------------------
# Helpers
# -----------------------------
def normalize_quotes(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("\u201c", '"').replace("\u201d", '"')
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

def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))  # hottest→coldest

def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i+1 for i, d in enumerate(order)}

def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i-span), min(9, i+span)
    return LETTERS[lo:hi+1]

def digits_for_letters_currentmap(letters: Set[str], digit_current_letters: Dict[str,str]) -> List[str]:
    return [d for d in DIGITS if digit_current_letters.get(d) in letters]

# -----------------------------
# Core analytics (maps, lists)
# -----------------------------
def compute_maps(last13: List[str]) -> Tuple[Dict, Dict]:
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]

    order_prev = heat_order(prev10)
    order_curr = heat_order(curr10)

    def pack(order, rows10):
        cnt = Counter(d for r in rows10 for d in r)
        for d in DIGITS:
            cnt.setdefault(d, 0)
        rank = rank_of_digit(order)
        d2L = {d: LETTERS[rank[d]-1] for d in DIGITS}
        return dict(order=order, rank=rank, counts=cnt, digit_letters=d2L)

    return pack(order_prev, prev10), pack(order_curr, curr10)

def loser_list(last13: List[str]) -> Tuple[List[str], Dict]:
    if len(last13) < 13:
        raise ValueError("Need 13 winners (Most Recent → Oldest).")
    rows = [list(s) for s in last13]
    info_prev, info_curr = compute_maps(last13)

    # Most-recent draw
    seed_digits = rows[0]          # list[str] len=5
    prev_digits = rows[1]
    prev2_digits = rows[2] if len(rows) > 2 else []

    digit_prev_letters  = info_prev["digit_letters"]
    digit_curr_letters  = info_curr["digit_letters"]

    # Previous-map core letters based on MR digits
    prev_core_letters = sorted({digit_prev_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))

    ring_letters = set()
    for L in prev_core_letters:
        ring_letters.update(neighbors(L, 1))
    ring_digits = digits_for_letters_currentmap(ring_letters, digit_curr_letters)

    loser_7_9 = info_curr["order"][-3:]
    curr_core_letters = sorted({digit_curr_letters[d] for d in seed_digits},
                               key=lambda L: LETTERS.index(L))
    new_core_digits = digits_for_letters_currentmap(set(curr_core_letters), digit_curr_letters)
    cooled_digits = [d for d in DIGITS if info_curr["rank"][d] > info_prev["rank"][d]]
    hot7_last10 = info_curr["order"][:7]

    prev_mirror_digits = sorted({str(MIRROR[int(d)]) for d in prev_digits}, key=int)
    union_last2 = sorted(set(prev_digits) | set(prev2_digits), key=int)
    due_last2   = sorted(set(DIGITS) - set(prev_digits) - set(prev2_digits), key=int)

    prev_core_currentmap_digits = digits_for_letters_currentmap(set(prev_core_letters), digit_curr_letters)
    edge_AC = digits_for_letters_currentmap(set("ABC"), digit_curr_letters)
    edge_HJ = digits_for_letters_currentmap(set("HIJ"), digit_curr_letters)

    seed_sum = sum(int(x) for x in seed_digits)
    prev_sum = sum(int(x) for x in prev_digits)

    core_size = len(prev_core_letters)
    core_size_flags = {
        "core_size_eq_2":   core_size == 2,
        "core_size_eq_5":   core_size == 5,
        "core_size_in_2_5": core_size in {2,5},
        "core_size_in_235": core_size in {2,3,5},
    }

    # ---------- minimal NEW definitions / aliases ----------
    # seed+1 (mod 10), top-2 carry from last two draws, and their union
    seedp1 = sorted({str((int(d) + 1) % 10) for d in seed_digits}, key=int)

    counts_last2 = Counter(prev_digits + prev2_digits)
    for d in DIGITS:
        counts_last2.setdefault(d, 0)
    carry2_order = sorted(DIGITS, key=lambda d: (-counts_last2[d], int(d)))
    carry2 = carry2_order[:2]

    union2 = sorted(set(carry2) | set(seedp1), key=int)

    # positional single-digit lists for seed and seed+1
    seed_pos  = [seed_digits[i] for i in range(5)]
    p1_pos    = [str((int(x) + 1) % 10) for x in seed_pos]
    # union of seed and seed+1 (set)
    union_digits = sorted(set(seed_pos) | set(p1_pos), key=int)
    # -------------------------------------------------------

    ctx = dict(
        # basic sets
        seed_digits=seed_digits, prev_digits=prev_digits, prev2_digits=prev2_digits,
        prev_mirror_digits=prev_mirror_digits, union_last2=union_last2, due_last2=due_last2,

        # maps
        digit_prev_letters=digit_prev_letters, digit_current_letters=digit_curr_letters,
        prev_core_letters=prev_core_letters, curr_core_letters=curr_core_letters,
        prev_core_currentmap_digits=prev_core_currentmap_digits,

        # groups
        ring_digits=ring_digits, new_core_digits=new_core_digits, cooled_digits=cooled_digits,
        loser_7_9=loser_7_9, hot7_last10=hot7_last10,

        # edges
        edge_AC=edge_AC, edge_HJ=edge_HJ,

        # sums / flags
        seed_sum=seed_sum, prev_sum=prev_sum, core_size_flags=core_size_flags,

        # NEW exposed lists used by your filters
        seedp1=seedp1,          # seed + 1 (set)
        carry2=carry2,          # top-2 carry
        union2=union2,          # carry2 ∪ seedp1
        seed_pos=seed_pos,      # positional seed digits
        p1_pos=p1_pos,          # positional seed+1 digits
        union_digits=union_digits,  # seed ∪ seed+1

        # heat maps for UI
        current_map_order="".join(info_curr["order"]),
        previous_map_order="".join(info_prev["order"]),
        current_counts=info_curr["counts"],
        previous_counts=info_prev["counts"],
        current_rank=info_curr["rank"],
        previous_rank=info_prev["rank"],
    )

    loser_ranking = list(reversed(info_curr["order"]))
    return loser_ranking, ctx

# -----------------------------
# CSV ingest (3-col / 5-col)
# -----------------------------
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
    cmap = {str(c).strip().lower(): c for c in df.columns}
    cols = set(cmap.keys())

    if {"name","description","expression"}.issubset(cols) and "id" not in cols:
        out = df[[cmap["name"], cmap["description"], cmap["expression"]]].copy()
        out.columns = ["name","description","expression"]
        out["name"]        = out["name"].astype(str)
        out["description"] = out["description"].astype(str)
        out["expression"]  = out["expression"].astype(str)
        mask = (out["name"].str.strip().str.lower() != "name") & (out["expression"].str.strip() != "")
        return out.loc[mask].reset_index(drop=True)

    need5 = {"id","name","enabled","applicable_if","expression"}
    if need5.issubset(cols):
        out = df[[cmap["id"], cmap["name"], cmap["enabled"], cmap["applicable_if"], cmap["expression"]]].copy()
        out.columns = ["id","name","enabled","applicable_if","expression"]
        out = out.astype({"id":str,"name":str,"enabled":str,"applicable_if":str,"expression":str})
        out3 = pd.DataFrame({
            "name":        out["id"],
            "description": out["name"],
            "expression":  out["expression"],
        })
        mask = (out3["name"].str.strip().str.lower() != "name") & (out3["expression"].str.strip() != "")
        return out3.loc[mask].reset_index(drop=True)

    raise ValueError("CSV must be 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression).")

def fmt_list(xs: List[str]) -> str:
    return "[" + ",".join(str(int(d)) for d in xs) + "]"

# -----------------------------
# Resolver
# -----------------------------
def resolve_expression(expr: str, ctx: Dict) -> str:
    x = (normalize_quotes(expr or "")).strip()

    # quoted digits → bare ints
    x = re.sub(r"'([0-9])'", r"\1", x)

    # digit_current_letters[var] in ['A','B',...']  --> var in [digit list]
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

    # digit_current_letters[var] in prev_core_letters  --> var in [digits]
    pat_in_prev_core = re.compile(r"digit_current_letters\s*\[\s*([A-Za-z_]\w*)\s*\]\s*in\s*prev_core_letters")
    if ctx.get("prev_core_currentmap_digits") is not None:
        allowed = fmt_list(ctx["prev_core_currentmap_digits"])
        x = pat_in_prev_core.sub(lambda m: f"{m.group(1)} in {allowed}", x)

    # Base list variables (both “in var” and bare “var”)
    list_vars = {
        "cooled_digits":      ctx["cooled_digits"],
        "new_core_digits":    ctx["new_core_digits"],
        "loser_7_9":          ctx["loser_7_9"],
        "ring_digits":        ctx["ring_digits"],
        "hot7_last10":        ctx["hot7_last10"],
        "hot7_last20":        ctx.get("hot7_last20", []),  # NEW

        "seed_digits":        ctx["seed_digits"],
        "prev_digits":        ctx["prev_digits"],
        "prev_mirror_digits": ctx["prev_mirror_digits"],
        "union_last2":        ctx["union_last2"],
        "due_last2":          ctx["due_last2"],
        "prev_core_currentmap_digits": ctx["prev_core_currentmap_digits"],
        "edge_AC":            ctx["edge_AC"],
        "edge_HJ":            ctx["edge_HJ"],

        # Shorthand sets/aliases in your filters
        "s1": [ctx["seed_pos"][0]],  # positional seed digits
        "s2": [ctx["seed_pos"][1]],
        "s3": [ctx["seed_pos"][2]],
        "s4": [ctx["seed_pos"][3]],
        "s5": [ctx["seed_pos"][4]],

        # Keep p1 = full seed+1 SET; p2..p5 = positional seed+1 singletons
        "p1":                 ctx.get("seedp1", []),
        "p2":                [ctx["p1_pos"][1]],
        "p3":                [ctx["p1_pos"][2]],
        "p4":                [ctx["p1_pos"][3]],
        "p5":                [ctx["p1_pos"][4]],

        # carry/union aliases
        "c1":                 ctx.get("carry2", []),
        "c2":                 ctx.get("carry2", []),
        "u1":                 ctx.get("union2", []),
        "u2":                 ctx.get("union2", []),
        "u3":                 ctx.get("union2", []),
        "u4":                 ctx.get("union2", []),
        "u5":                 ctx.get("union2", []),
        "u6":                 ctx.get("union2", []),
        "u7":                 ctx.get("union2", []),

        # Uppercase union of seed and seed+1 (used by UNI2/UNI3)
        "UNION_DIGITS":       ctx.get("union_digits", []),
    }

    # do the replacements
    for name, arr in list_vars.items():
        lit = fmt_list(arr)
        x = re.sub(rf"\bin\s+{name}\b", " in " + lit, x)
        x = re.sub(rf"\b{name}\b", lit, x)

    # Letter-set contains → True/False
    def letter_contains(txt: str, varname: str, letters: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + re.escape(varname))
        return p.sub(lambda mm: "True" if mm.group(1) in letters else "False", txt)

    x = letter_contains(x, "prev_core_letters", set(ctx["prev_core_letters"]))
    x = letter_contains(x, "core_letters",      set(ctx["curr_core_letters"]))

    # Scalar sums
    x = re.sub(r"\bseed_sum\b", str(ctx.get("seed_sum", 0)), x)
    x = re.sub(r"\bprev_sum\b", str(ctx.get("prev_sum", 0)), x)

    # Core-size flags
    for key, val in (ctx.get("core_size_flags") or {}).items():
        x = re.sub(rf"\b{re.escape(key)}\b", "True" if val else "False", x)

    return x

def resolve_text_placeholders(text: str, ctx: Dict) -> str:
    t = normalize_quotes(text or "")
    repls = {
        "{seed-bucket digits}": fmt_list(ctx["prev_core_currentmap_digits"]),
        "{edge-AC}":            fmt_list(ctx["edge_AC"]),
        "{edge-HJ}":            fmt_list(ctx["edge_HJ"]),
        "{loser_7_9}":          fmt_list(ctx["loser_7_9"]),
        "{ring_digits}":        fmt_list(ctx["ring_digits"]),
        "{hot7_last10}":        fmt_list(ctx["hot7_last10"]),
        "{union_last2}":        fmt_list(ctx["union_last2"]),
        "{due_last2}":          fmt_list(ctx["due_last2"]),
        "{prev_mirror_digits}": fmt_list(ctx["prev_mirror_digits"]),
        "{SEEDP1}":             fmt_list(ctx.get("seedp1", [])),
        "{CARRY2}":             fmt_list(ctx.get("carry2", [])),
        "{UNION2}":             fmt_list(ctx.get("union2", [])),
        "{UNION_DIGITS}":       fmt_list(ctx.get("union_digits", [])),  # NEW
    }
    for k, v in repls.items():
        t = t.replace(k, v)
    return t

def build_tester_csv_from_paste(pasted_text: str, ctx: Dict) -> pd.DataFrame:
    df3 = to_three_cols(_read_csv_loose(pasted_text))

    resolved_names = []
    resolved_descs = []
    resolved_exprs = []
    for _, r in df3.iterrows():
        nm  = str(r["name"])
        dsc = str(r["description"])
        exp = str(r["expression"])

        nm_res  = resolve_text_placeholders(nm, ctx)
        dsc_res = resolve_text_placeholders(dsc, ctx)
        exp_res = resolve_expression(exp, ctx)

        resolved_names.append(nm_res)
        resolved_descs.append(dsc_res)
        resolved_exprs.append(exp_res)

    out = pd.DataFrame({
        "id":            resolved_names,
        "name":          resolved_descs,
        "enabled":       ["TRUE"] * len(df3),
        "applicable_if": [""     ] * len(df3),
        "expression":    resolved_exprs,
    })
    for i in range(5, 15):
        out[f"Unnamed: {i}"] = ""
    return out

# -----------------------------
# UI
# -----------------------------
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
                                 help="Needed only for *Last-20 Hot* filters (e.g., xFH5).")
    compute = st.form_submit_button("Compute")

def counts_frame(counts: Counter) -> pd.DataFrame:
    return pd.DataFrame({"digit": [int(d) for d in DIGITS],
                         "count": [int(counts[str(d)]) for d in range(10)]})

def render_heat_maps(ctx: Dict):
    st.subheader("Heat Maps (last 10 draws)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current Map (MR → MR-9)**")
        st.write("Order (hot→cold):", " ".join(ctx["current_map_order"]))
        df = counts_frame(ctx["current_counts"])
        st.bar_chart(df.set_index("digit"))
    with c2:
        st.markdown("**Previous Map (MR-1 → MR-10)**")
        st.write("Order (hot→cold):", " ".join(ctx["previous_map_order"]))
        dfp = counts_frame(ctx["previous_counts"])
        st.bar_chart(dfp.set_index("digit"))

def render_loser_lists(loser_ranking: List[str], ctx: Dict):
    st.subheader("Loser List / Rankings (current map)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Coldest → Hottest (digits)**")
        st.code(", ".join(loser_ranking))
    with col2:
        st.markdown("**Loser 7–9 (coldest 3)**")
        st.code(", ".join(ctx["loser_7_9"]) or "∅")
    with col3:
        st.markdown("**Edge groups (A–C / H–J)**")
        st.write("A–C:", ", ".join(ctx["edge_AC"]) or "∅")
        st.write("H–J:", ", ".join(ctx["edge_HJ"]) or "∅")

def render_resolved_variables(ctx: Dict):
    st.subheader("Resolved variables (this run)")
    rows = [
        ("seed_digits",           ctx["seed_digits"]),
        ("prev_digits",           ctx["prev_digits"]),
        ("prev_mirror_digits",    ctx["prev_mirror_digits"]),
        ("loser_7_9",             ctx["loser_7_9"]),
        ("ring_digits",           ctx["ring_digits"]),
        ("new_core_digits",       ctx["new_core_digits"]),
        ("cooled_digits",         ctx["cooled_digits"]),
        ("hot7_last10",           ctx["hot7_last10"]),
        ("hot7_last20",           ctx.get("hot7_last20", [])),
        ("union_last2",           ctx["union_last2"]),
        ("due_last2",             ctx["due_last2"]),
        ("seed-bucket digits",    ctx["prev_core_currentmap_digits"]),
        ("edge_AC",               ctx["edge_AC"]),
        ("edge_HJ",               ctx["edge_HJ"]),
        ("seed+1 (set)",          ctx["seedp1"]),
        ("carry2",                ctx["carry2"]),
        ("union2",                ctx["union2"]),
        ("UNION_DIGITS",          ctx["union_digits"]),
    ]
    for i in range(0, len(rows), 4):
        cols = st.columns(4)
        for (label, vals), c in zip(rows[i:i+4], cols):
            with c:
                st.markdown(f"**{label}**")
                st.code(", ".join(vals) if isinstance(vals, list) and vals else "∅")

if compute:
    try:
        last13 = parse_winners_text(st.session_state.get("winners_text",""), pad4=st.session_state.get("pad4", True))
        if len(last13) < 13:
            st.error("Please provide at least 13 winners (MR→Oldest).")
        else:
            loser_ranking, ctx = loser_list(last13)

            winners20_text = st.session_state.get("winners20_text","").strip()
            if winners20_text:
                last20 = parse_winners_text(winners20_text, pad4=st.session_state.get("pad4", True))
                c = Counter(d for s in last20[:20] for d in s)
                for d in DIGITS:
                    c.setdefault(d, 0)
                hot7_20 = [d for d,_ in c.most_common(7)]
                ctx["hot7_last20"] = hot7_20
            else:
                ctx["hot7_last20"] = []

            st.session_state["ctx"] = ctx
            st.session_state["loser_ranking"] = loser_ranking

            render_heat_maps(ctx)
            render_loser_lists(loser_ranking, ctx)
            render_resolved_variables(ctx)

    except Exception as e:
        st.error(str(e))

if "ctx" in st.session_state:
    st.markdown("---")
    st.markdown("### Paste Filters (3-col or 5-col)")
    with st.form("csv_form", clear_on_submit=False):
        mega_csv = st.text_area(
            "CSV content",
            key="mega_csv",
            height=220,
            help="Accepts 3-col (name,description,expression) or 5-col (id,name,enabled,applicable_if,expression)."
        )
        build = st.form_submit_button("Build Tester CSV")

    if build:
        try:
            tester_df = build_tester_csv_from_paste(
                pasted_text=mega_csv,
                ctx=st.session_state["ctx"],
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
