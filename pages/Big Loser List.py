# big_loser_list.py
import io
import re
from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

# -------------------------------
# Helpers: parsing & normalization
# -------------------------------

DIGIT_RE = re.compile(r"^\d+$")
LETTERS = "ABCDEFGHIJ"  # A=rank1 (hottest) ... J=rank10 (coldest)

def read_winners_from_file(uploaded) -> List[str]:
    """
    Accepts CSV or TXT. Extracts tokens that are all-digits length 4 or 5.
    If CSV, inspects all cells; if TXT, splits on non-word boundaries.
    Returns in the textual order of appearance.
    """
    raw = uploaded.getvalue()
    # Try CSV first; if it fails, fallback to text lines
    try:
        df = pd.read_csv(io.BytesIO(raw))
        winners = []
        for _, row in df.iterrows():
            for val in row.values:
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if DIGIT_RE.match(s) and len(s) in (4, 5):
                    winners.append(s)
        if winners:
            return winners
    except Exception:
        pass

    # TXT / generic: pull out digit tokens per line
    text = raw.decode(errors="ignore")
    winners = []
    for line in text.splitlines():
        for token in re.findall(r"\b\d{4,5}\b", line):
            winners.append(token.strip())
    return winners

def normalize_length(items: List[str], pad_4_to_5: bool) -> List[str]:
    out = []
    for s in items:
        if len(s) == 5:
            out.append(s)
        elif len(s) == 4 and pad_4_to_5:
            out.append("0" + s)
        # else drop
    return out

# -----------------------------------
# Heat map & letter/±1 neighborhood
# -----------------------------------

def heat_map_from_window(window: List[str]) -> str:
    """
    Given a list of 5-digit winners, build a frequency map 0..9,
    then return digits sorted hottest->coldest as a string (length 10).
    Tie-breaker: smaller digit first for stability.
    """
    cnt = Counter()
    for w in window:
        for ch in w:
            cnt[int(ch)] += 1
    # Ensure all digits present
    for d in range(10):
        cnt[d] += 0
    # Sort by (-count, digit)
    ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return "".join(str(d) for d, _ in ordered)

def rank_letters_from_map(hot_to_cold: str) -> Dict[int, str]:
    """
    Map digit -> letter rank using the A..J scheme for hot->cold.
    hot_to_cold is a 10-char string of digits with rank order.
    """
    d2letter = {}
    for i, ch in enumerate(hot_to_cold):
        d2letter[int(ch)] = LETTERS[i]  # i=0 -> A ... i=9 -> J
    return d2letter

def neighbors_pm1(letter: str) -> List[str]:
    i = LETTERS.index(letter)
    neigh = [letter]
    if i > 0:
        neigh.append(LETTERS[i - 1])
    if i < 9:
        neigh.append(LETTERS[i + 1])
    # keep unique, keep alpha order A..J
    return sorted(set(neigh), key=lambda L: LETTERS.index(L))

# -------------------------------
# Batch computation per row (t)
# -------------------------------

def compute_batch(winners: List[str], oldest_to_newest: bool) -> pd.DataFrame:
    """
    winners: list of 5-char strings
    oldest_to_newest: display order later, but the computation uses the
                      true chronological order oldest->newest internally.
    Returns a DataFrame with all derived columns.
    """
    # Internally we want oldest->newest to compute rolling windows cleanly
    chron = winners[:]  # already oldest->newest? If not, caller will ensure later.
    # If the incoming list is most-recent->oldest, reverse it first
    # (we'll handle final display order separately).
    # The app will pass winners already adjusted.

    rows = []
    W = 10  # rolling window length for heat map
    DUE_W = 2  # due window W=2 (digits not seen in last 2 draws)

    for t in range(len(chron)):
        mr_winner = chron[t]

        # Build windows
        prev_win = chron[max(0, t - W - 1):max(0, t - 1)]   # the window *before* the seed (t-1)
        curr_win = chron[max(0, t - W):t]                   # the window before current (t)

        # Heat maps
        prev_map = heat_map_from_window(prev_win) if len(prev_win) >= 1 else ""
        curr_map = heat_map_from_window(curr_win) if len(curr_win) >= 1 else ""

        # Core letters from MR winner (prev map):
        # - Map the SEED (t-1) digits onto the *prev_map* ranks to get letters
        if t - 1 >= 0 and prev_map:
            seed_digits = [int(c) for c in chron[t - 1]]
            d2L_prev = rank_letters_from_map(prev_map)
            core_letters = sorted({d2L_prev[d] for d in seed_digits}, key=lambda L: LETTERS.index(L))
        else:
            core_letters = []

        # U letters = union of ±1 neighborhoods of core_letters
        U = sorted(
            set(L for c in core_letters for L in neighbors_pm1(c)),
            key=lambda L: LETTERS.index(L)
        )

        # Due W=2 (digits not seen in the last 2 draws *before* t)
        tail = chron[max(0, t - DUE_W):t]
        seen = set(ch for w in tail for ch in w)
        due = [str(d) for d in range(10) if str(d) not in seen]

        # "loser list" (least->most likely):
        # pragmatic ordering:
        # 1) digits whose letters are NOT in U (less likely next) come first,
        # 2) then digits whose letters are in U,
        # 3) within each bucket, sort by curr_map rank (colder first), then digit asc.
        if curr_map:
            d2L_curr = rank_letters_from_map(curr_map)
            not_in_U = []
            in_U = []
            for d in range(10):
                L = d2L_curr[d]
                item = (LETTERS.index(L), d)  # rank index, digit
                if L in U:
                    in_U.append(item)
                else:
                    not_in_U.append(item)
            # colder first => higher index first; so sort by (-rank_index, digit)
            not_in_U_sorted = [d for _, d in sorted(not_in_U, key=lambda x: (-x[0], x[1]))]
            in_U_sorted = [d for _, d in sorted(in_U, key=lambda x: (-x[0], x[1]))]
            loser_order = not_in_U_sorted + in_U_sorted
            loser_str = "".join(str(d) for d in loser_order)
        else:
            loser_str = ""

        rows.append({
            "mr_winner": mr_winner,
            "prev_map_hot_to_cold": prev_map,
            "curr_map_hot_to_cold": curr_map,
            "core_letters": ", ".join(core_letters) if core_letters else "",
            "U_letters": ", ".join(U) if U else "",
            "due_W2": ", ".join(due) if due else "",
            "loser_list_0_9": loser_str,
        })

    df = pd.DataFrame(rows)
    # Annotate a stable index for display; we'll reorder for the UI as needed
    df.insert(0, "index", range(len(df)))
    return df

# ---------------
# Streamlit UI
# ---------------

st.set_page_config(page_title="Loser List Batch Explorer — ±1 Neighborhood Method", layout="wide")

st.title("Loser List Batch Explorer — ±1 Neighborhood Method")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload history (TXT/CSV)", type=["csv", "txt"])
    order = st.radio(
        "History order in file",
        options=["Most-recent → Oldest", "Oldest → Most-recent"],
        index=0,
        help="Tell the app how your file is ordered so it can compute rolling windows correctly.",
    )
    pad4 = st.checkbox("Pad 4-digit items to 5 digits", value=True,
                       help="If checked, 4-digit items like 1234 become 01234.")
    recompute = st.checkbox("Recompute anyway", value=False,
                            help="Force recomputation even if inputs look unchanged.")
    compute = st.button("Compute", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload a file and click **Compute** to see results.")
    st.stop()

# Parse
raw_winners = read_winners_from_file(uploaded)
if not raw_winners:
    st.error("No 4- or 5-digit tokens found in the uploaded file.")
    st.stop()

winners = normalize_length(raw_winners, pad4)
if not winners:
    st.error("Nothing to process after normalization. (Maybe all items were 4-digit and padding is off?)")
    st.stop()

# Persist user inputs to avoid auto-refresh recalcs
key_sig = (uploaded.name, len(uploaded.getvalue()), order, pad4)

if "cache_key" not in st.session_state:
    st.session_state["cache_key"] = None
if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None
if compute or recompute or st.session_state["cache_key"] != key_sig:
    # Make sure computation uses true chronological order oldest->newest
    if order == "Most-recent → Oldest":
        winners_chron = list(reversed(winners))
    else:
        winners_chron = winners[:]

    batch_df = compute_batch(winners_chron, oldest_to_newest=True)
    st.session_state["batch_df"] = batch_df
    st.session_state["cache_key"] = key_sig

batch_df = st.session_state["batch_df"]
if batch_df is None or batch_df.empty:
    st.warning("Nothing to show.")
    st.stop()

# -------------------------
# View options & rendering
# -------------------------

st.divider()
st.subheader("View options")

chronological_view = st.checkbox("Show oldest → newest (seed above next)", value=True)
pair_view = st.checkbox("Show pairs: seed (t−1) above next winner (t)", value=True)

# Select display order ONLY (does not affect computations)
display_df = batch_df.copy()
if chronological_view:
    display_df = display_df.iloc[::-1].reset_index(drop=True)

st.subheader("Batch table")
st.dataframe(display_df, use_container_width=True, height=420)

# Pair step-through
if pair_view and len(display_df) >= 2:
    st.subheader("Seed → Next winner (step-through)")
    i = st.slider("Pair index (t; 2..N)", min_value=2, max_value=len(display_df), value=2)
    seed_row = display_df.iloc[i - 2]  # t-1
    next_row = display_df.iloc[i - 1]  # t

    def show_row(title, row):
        cols = [
            "mr_winner",
            "prev_map_hot_to_cold",
            "curr_map_hot_to_cold",
            "core_letters",
            "U_letters",
            "due_W2",
            "loser_list_0_9",
        ]
        pretty = row[cols].to_frame().rename(columns={row.name: ""})
        st.markdown(f"**{title}**")
        st.table(pretty)

    show_row("Seed (t−1)", seed_row)
    show_row("Next winner (t)", next_row)

    # small Δ summary you can extend
    st.caption("Δ quick look")
    seed_core = set([s.strip() for s in str(seed_row.core_letters).split(",") if s.strip()])
    next_core = set([s.strip() for s in str(next_row.core_letters).split(",") if s.strip()])
    overlap = len(seed_core & next_core)
    seed_due = set([s.strip() for s in str(seed_row.due_W2).split(",") if s.strip()])
    next_digits = set(list(str(next_row.mr_winner)))
    due_used = len(seed_due & next_digits) > 0
    st.write({
        "core_letters_overlap": overlap,
        "seed_due_used_in_next": due_used,
    })

# Export as-shown
st.download_button(
    "Download table (as shown) CSV",
    data=display_df.to_csv(index=False).encode(),
    file_name="loser_list_batch_view.csv",
    mime="text/csv",
    use_container_width=True,
)
