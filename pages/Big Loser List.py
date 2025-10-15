import streamlit as st
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
import io
import hashlib

# ----- Page -----
st.set_page_config(page_title="Loser List Batch Explorer", layout="wide")
st.title("Loser List Batch Explorer — ±1 Neighborhood Method")

DIGITS = list("0123456789")
LETTERS = list("ABCDEFGHIJ")

# =============== Core logic (mirrors your single-winner page) ===============
def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
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

def loser_list_block(last13_mr_to_oldest: List[str]) -> Tuple[Dict, Dict]:
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need 13 winners (MR → Oldest).")
    rows = [list(s) for s in last13_mr_to_oldest]

    # Previous map (for MR winner)
    prev10 = rows[1:11]
    order_prev = heat_order(prev10)
    rank_prev = rank_of_digit(order_prev)

    # Core letters from MR on previous map
    most_recent = rows[0]
    digit_to_letter_prev = {d: LETTERS[rank_prev[d] - 1] for d in DIGITS}
    core_letters = sorted({digit_to_letter_prev[d] for d in most_recent},
                          key=lambda L: LETTERS.index(L))

    # U = union of ±1 neighborhoods of core letters
    U = set()
    for L in core_letters:
        U.update(neighbors(L, 1))
    U_letters = sorted(U, key=lambda L: LETTERS.index(L))

    # Current map (for next draw)
    curr10 = rows[0:10]
    order_curr = heat_order(curr10)
    rank_curr = rank_of_digit(order_curr)
    digit_to_letter_curr = {d: LETTERS[rank_curr[d] - 1] for d in DIGITS}

    # Due (W=2)
    due = due_set(rows[0:2])

    # Age in curr10 (0 if seen in MR)
    age = {d: None for d in DIGITS}
    for back, r in enumerate(curr10):  # rows[0] is MR
        s = set(r)
        for d in DIGITS:
            if age[d] is None and d in s:
                age[d] = back
    for d in DIGITS:
        if age[d] is None:
            age[d] = 9999

    # Tiers
    tiers = {}
    for d in DIGITS:
        Lc = digit_to_letter_curr[d]
        if Lc not in U:
            tiers[d] = 0
        elif Lc not in core_letters:
            tiers[d] = 2 if d in due else 1
        else:
            tiers[d] = 3

    # Loser list (least → most likely)
    def sort_key(d: str):
        return (tiers[d], rank_curr[d], -age[d])
    loser_order = " ".join(sorted(DIGITS, key=sort_key))

    summary = {
        "mr_winner": "".join(most_recent),
        "prev_map_hot_to_cold": "".join(order_prev),
        "curr_map_hot_to_cold": "".join(order_curr),
        "core_letters": ", ".join(core_letters),
        "U_letters": ", ".join(U_letters),
        "due_W2": ", ".join(sorted(due)),
        "loser_list_0_9": loser_order,
    }
    detail = {
        "digit_current_letters": digit_to_letter_curr,
        "digit_tiers": tiers,
        "digit_ages": age,
        "rank_curr_map": rank_curr,
        "due_set": sorted(list(due)),
        "core_letters": core_letters,
        "U_letters": U_letters,
        "prev_map_order": "".join(order_prev),
        "curr_map_order": "".join(order_curr),
        "mr_winner": "".join(most_recent),
    }
    return summary, detail

# =============== Utilities ===============
def parse_winners_blob(text: str, pad4: bool) -> List[str]:
    raw = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    out = []
    for tok in raw:
        if not tok.isdigit():
            raise ValueError(f"Non-digit token found: {tok!r}")
        if len(tok) == 4 and pad4:
            tok = tok.zfill(5)
        if len(tok) != 5:
            raise ValueError(f"Every item must be 5 digits (or 4 with pad). Got: {tok!r}")
        out.append(tok)
    return out

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# =============== Sidebar FORM (no auto-run) ===============
with st.sidebar:
    st.header("Input")
    with st.form("controls", clear_on_submit=False):
        up = st.file_uploader("Upload history (TXT/CSV)", type=["txt", "csv"])
        order = st.radio("History order in file", ["Most-recent → Oldest", "Oldest → Most-recent"], index=0)
        pad4 = st.checkbox("Pad 4-digit items to 5 digits", value=True)
        recompute = st.checkbox("Recompute anyway", value=False,
                                help="Force a fresh run even if input hasn't changed.")
        submitted = st.form_submit_button("Compute")

# Session buckets
ss = st.session_state
if "data_hash" not in ss: ss.data_hash = None
if "df" not in ss: ss.df = None
if "details" not in ss: ss.details = None
if "order" not in ss: ss.order = None
if "pad4" not in ss: ss.pad4 = None

# =============== Run only when Compute is clicked ===============
if submitted:
    if not up:
        st.error("Please upload a TXT/CSV with your winners.")
    else:
        content = up.read()
        this_hash = hash_bytes(content + f"|{order}|{pad4}".encode("utf-8"))
        if (this_hash != ss.data_hash) or recompute:
            try:
                text = content.decode("utf-8")
                winners = parse_winners_blob(text, pad4=pad4)
                # Normalize to MR → Oldest for processing
                if order == "Oldest → Most-recent":
                    winners = list(reversed(winners))
                if len(winners) < 13:
                    st.error(f"Need at least 13 winners to compute one row; got {len(winners)}.")
                else:
                    summaries, details = [], []
                    for i in range(0, len(winners) - 12):
                        window_13 = winners[i:i+13]
                        summary, detail = loser_list_block(window_13)
                        summary["index"] = i
                        summaries.append(summary)
                        details.append(detail)
                    ss.df = pd.DataFrame(summaries, columns=[
                        "index","mr_winner","prev_map_hot_to_cold",
                        "curr_map_hot_to_cold","core_letters","U_letters",
                        "due_W2","loser_list_0_9",
                    ])
                    ss.details = details
                    ss.data_hash = this_hash
                    ss.order = order
                    ss.pad4 = pad4
                    st.success(f"Computed {len(ss.df)} rows.")
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Inputs unchanged — showing cached results.")

# =============== Display (never recomputes) ===============
if ss.df is not None and ss.details is not None:
    st.subheader("Batch table")
    st.dataframe(ss.df, hide_index=True, use_container_width=True)

    st.markdown("### Row detail")
    row_idx = st.number_input("Select row index (0 = most-recent)", min_value=0,
                              max_value=len(ss.df)-1, value=0, step=1)
    d = ss.details[row_idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Previous map (for MR winner)** — hot → cold")
        st.code(d["prev_map_order"])
        st.markdown("**Core letters from MR winner (prev map)**")
        st.code(", ".join(d["core_letters"]))
        st.markdown("**±1 union U (letters)**")
        st.code(", ".join(d["U_letters"]))
    with c2:
        st.markdown("**Current map (for next draw)** — hot → cold")
        st.code(d["curr_map_order"])
        st.markdown("**Due (W=2)**")
        st.code(", ".join(d["due_set"]))

    st.markdown("**Digit classification (selected row)**")
    rows = []
    for digit in DIGITS:
        rows.append({
            "digit": digit,
            "letter_today": d["digit_current_letters"][digit],
            "tier": d["digit_tiers"][digit],
            "age": d["digit_ages"][digit],
            "heat_rank_today (1=hottest)": d["rank_curr_map"][digit],
        })
    df_detail = pd.DataFrame(rows).sort_values(
        ["tier", "heat_rank_today (1=hottest)", "age"],
        ascending=[True, True, True]
    )
    st.dataframe(df_detail, hide_index=True, use_container_width=True)

    # Downloads
    csv_buf = io.StringIO()
    ss.df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download batch table (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="loserlist_batch_table.csv",
        mime="text/csv",
    )

    def as_txt() -> str:
        lines = []
        for _, s in ss.df.sort_values("index").iterrows():
            lines.append(f"Row {int(s['index'])} — MR Winner: {s['mr_winner']}")
            lines.append(f"  Prev map: {s['prev_map_hot_to_cold']}")
            lines.append(f"  Curr map: {s['curr_map_hot_to_cold']}")
            lines.append(f"  Core letters: {s['core_letters']}")
            lines.append(f"  U letters: {s['U_letters']}")
            lines.append(f"  Due (W=2): {s['due_W2']}")
            lines.append(f"  Loser list: {s['loser_list_0_9']}")
            lines.append("")
        return "\n".join(lines)

    st.download_button(
        "Download batch report (TXT)",
        data=as_txt().encode("utf-8"),
        file_name="loserlist_batch_report.txt",
        mime="text/plain",
    )
else:
    st.info("Upload a file and click **Compute** to see results.")

with st.expander("Method refresher"):
    st.markdown("""
- **Previous map**: winners #2..#11 (relative to MR) → map the MR winner there.
- **Core letters**: letters of MR’s digits on the previous map.
- **U**: union of **±1** neighborhoods of those core letters.
- **Current map**: winners #1..#10 (today’s map).
- **Due (W=2)**: digits not seen in last 2 winners.
- **Loser list** sorts digits (least → most likely) by **tier → heat-rank → age**.
""")
