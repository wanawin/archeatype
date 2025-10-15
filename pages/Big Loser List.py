# big_loser_list_app.py
# Loser List Batch Explorer — ±1 Neighborhood Method
import streamlit as st
import pandas as pd
import re
import hashlib
from collections import Counter
from typing import List, Tuple, Dict

st.set_page_config(page_title="Loser List Batch Explorer", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Emergency reset (useful if you suspect stale cache/state)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Clear cache & state"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.session_state.clear()
        st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# UI: Uploader + options
# ──────────────────────────────────────────────────────────────────────────────
st.title("Loser List Batch Explorer — ±1 Neighborhood Method")

with st.sidebar:
    uploaded = st.file_uploader("Upload history (TXT/CSV)", type=["txt", "csv"])

    order_choice = st.radio(
        "History order in file",
        ["Most-recent → Oldest", "Oldest → Most-recent"],
        index=0,
        help="This only flips the parsed list; parsing preserves file appearance order."
    )
    pad4 = st.checkbox("Pad 4-digit items to 5 digits", value=True,
                       help="If a token has 4 digits, left-pad to 5 (e.g., 1234 → 01234).")
    force = st.checkbox("Recompute anyway", value=False,
                        help="Bypass cache for this run.")
    compute = st.button("Compute", type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

A2J = "ABCDEFGHIJ"  # rank letters (A = hottest, J = coldest)

def parse_winners(content_bytes: bytes, pad4_flag: bool) -> List[str]:
    """
    Extract 4–5 digit tokens, keep the exact order they appear,
    normalize to 5 digits if requested.
    """
    text = content_bytes.decode("utf-8", errors="ignore")
    tokens = re.findall(r'(?<!\d)(\d{4,5})(?!\d)', text)
    out = []
    for t in tokens:
        if len(t) == 5:
            out.append(t)
        elif len(t) == 4 and pad4_flag:
            out.append(t.zfill(5))
    return out

def window_freqs(seq: List[str]) -> Dict[int, int]:
    """Count digit frequency 0–9 across a list of 5-digit winners."""
    c = Counter()
    for w in seq:
        for ch in w:
            c[int(ch)] += 1
    # ensure keys 0..9 exist
    for d in range(10):
        c[d] += 0
    return dict(c)

def hot_to_cold_string(freqs: Dict[int, int]) -> str:
    """
    Return a 10-digit string ordered from hottest to coldest (A→J).
    Break ties by digit ascending (stable).
    """
    ordered = sorted(range(10), key=lambda d: (-freqs[d], d))
    return "".join(str(d) for d in ordered)

def cold_to_hot_string(freqs: Dict[int, int]) -> str:
    """Return digits ordered from coldest to hottest (least→most likely)."""
    ordered = sorted(range(10), key=lambda d: (freqs[d], d))
    return "".join(str(d) for d in ordered)

def letter_map_from_htc(htc: str) -> Dict[int, str]:
    """
    Given a hot→cold string (e.g., '304...'), build map digit→letter,
    where A=position 0 (hottest), ..., J=position 9 (coldest).
    """
    mapping = {}
    for i, ch in enumerate(htc):
        mapping[int(ch)] = A2J[i]
    return mapping

def letters_for_winner(winner: str, digit2letter: Dict[int, str]) -> List[str]:
    """Map each digit in the winner to its letter on the given hot→cold map."""
    out = []
    for ch in winner:
        out.append(digit2letter[int(ch)])
    # Keep order but remove duplicates while preserving first appearance
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def neighborhood_union(letters: List[str]) -> List[str]:
    """
    ±1 neighborhood over A..J for the provided letters.
    Example: C → {B,C,D}; J → {I,J}, A → {A,B}
    Return unique letters sorted by A..J order.
    """
    idx = {ch: i for i, ch in enumerate(A2J)}
    bag = set()
    for L in letters:
        i = idx[L]
        bag.add(A2J[i])
        if i - 1 >= 0:
            bag.add(A2J[i-1])
        if i + 1 < 10:
            bag.add(A2J[i+1])
    return [ch for ch in A2J if ch in bag]

def due_digits_last_W(history: List[str], W: int) -> List[int]:
    """Digits 0–9 that do not appear in the last W winners (history is a list of 5-digit strings)."""
    recent = history[-W:] if W > 0 else []
    seen = set(int(ch) for w in recent for ch in w)
    return [d for d in range(10) if d not in seen]

# ──────────────────────────────────────────────────────────────────────────────
# Parse + Preview
# ──────────────────────────────────────────────────────────────────────────────

winners: List[str] = []
content_hash = None

if uploaded is not None:
    raw = uploaded.getvalue()  # ALWAYS current content
    content_hash = hashlib.md5(raw).hexdigest()

    @st.cache_data(show_spinner=False)
    def _cached_parse(b: bytes, pad: bool, _sig: str) -> List[str]:
        return parse_winners(b, pad)

    winners = _cached_parse(raw, pad4, content_hash) if not force else parse_winners(raw, pad4)

    # Apply the order toggle AFTER parsing (we keep the file's appearance order in parse_winners)
    # We want the list in "Oldest → Most-recent" internally, so reverse if needed.
    # If user says "Most-recent → Oldest", that means the first line is most-recent,
    # we flip to have oldest-first processing for the math below.
    if order_choice == "Most-recent → Oldest":
        winners = list(reversed(winners))

    st.caption("Parsed winners preview (Oldest → Most-recent after toggle):")
    colA, colB = st.columns(2)
    with colA:
        st.write("Head", winners[:10])
    with colB:
        st.write("Tail", winners[-10:])

else:
    st.info("Upload a history file (TXT or CSV) to begin.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Compute on demand
# ──────────────────────────────────────────────────────────────────────────────
if not compute:
    st.info("Adjust options if needed, then click **Compute** to build the batch table.")
    st.stop()

# Minimum winners to make sense
if len(winners) < 12:
    st.error("Please provide at least 12 winners (more preferred).")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Batch computation
# We treat the list as Oldest → Most-recent (after toggle).
# For each index j (current winner), we build the row based on windows ending at j-1 and j.
# Window length for maps: 10 previous draws.
# ──────────────────────────────────────────────────────────────────────────────

rows = []
WIN = 10
for j in range(len(winners)):
    mr = winners[j]

    # windows:
    prev_window_start = max(0, j - WIN)
    prev_window = winners[prev_window_start:j]  # up to but excluding current (j)
    curr_window_start = max(0, j - WIN + 1)
    curr_window = winners[curr_window_start:j+1]  # including current

    # Need at least 1 in prev_window to compute letters meaningfully
    if len(prev_window) == 0:
        # Leave row with N/A-like placeholders
        rows.append({
            "index": j,
            "mr_winner": mr,
            "prev_map_hot_to_cold": "",
            "curr_map_hot_to_cold": "",
            "core_letters": "",
            "U_letters": "",
            "due_W2": "",
            "loser_list_0_9": "",
        })
        continue

    # Maps
    prev_freqs = window_freqs(prev_window)
    prev_htc = hot_to_cold_string(prev_freqs)

    curr_freqs = window_freqs(curr_window)
    curr_htc = hot_to_cold_string(curr_freqs)

    # Core letters: map each digit of current winner on the PREV map
    d2L_prev = letter_map_from_htc(prev_htc)
    core = letters_for_winner(mr, d2L_prev)  # unique in order of first appearance

    # U letters: ±1 neighborhood union (A..J)
    U = neighborhood_union(core)

    # Due_W2: based on last 2 draws BEFORE current (i.e., winners[j-2:j])
    due_W2 = due_digits_last_W(winners[max(0, j-2):j], W=2)

    # Loser list: digits coldest→hottest on the CURRENT map
    loser_0_9 = cold_to_hot_string(curr_freqs)

    rows.append({
        "index": j,
        "mr_winner": mr,
        "prev_map_hot_to_cold": prev_htc,
        "curr_map_hot_to_cold": curr_htc,
        "core_letters": ", ".join(core),
        "U_letters": ", ".join(U),
        "due_W2": ", ".join(str(d) for d in due_W2),
        "loser_list_0_9": loser_0_9,
    })

df = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# View options + table
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Batch table")

with st.expander("View options", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        show_oldest_to_newest = st.checkbox("Show oldest → newest", value=True,
                                            help="When checked, table shows the same internal order (oldest first).")
    with col2:
        pair_view = st.checkbox("Show pairs: seed (row j-1) directly above winner (row j)", value=False,
                                help="Helpful if you want to see the dependency row-by-row. (The data itself is already computed that way.)")

# Order dataframe for display
df_disp = df.copy()
if show_oldest_to_newest:
    df_disp = df_disp.sort_values("index", ascending=True)
else:
    df_disp = df_disp.sort_values("index", ascending=False)

st.dataframe(df_disp, use_container_width=True, height=500)

# Row detail
st.subheader("Row detail")
row_idx = st.number_input("Select row index (0 = oldest)", min_value=0, max_value=int(df["index"].max()), value=0, step=1)
detail = df[df["index"] == row_idx]
st.write(detail if not detail.empty else "No data for that index.")

# Download
st.download_button(
    "Download batch table (CSV)",
    data=df.to_csv(index=False),
    file_name="loser_list_batch_table.csv",
    mime="text/csv"
)

st.caption("""
**Notes**
- We treat uploaded winners in the exact order you provided; the toggle flips it to *Oldest → Most-recent* internally for processing.  
- `prev_map_hot_to_cold` uses the previous 10 winners (ending at row j−1).  
- `curr_map_hot_to_cold` uses a 10-winner window **including** the current row j (this is what you'd use to predict j+1).  
- `core_letters` are A–J letters for the current winner mapped onto the *previous* map; `U_letters` is the ±1 neighborhood union.  
- `due_W2` lists digits that didn’t appear in the prior 2 draws (before the current).  
- `loser_list_0_9` orders digits from **coldest→hottest** on the current map.  
""")
