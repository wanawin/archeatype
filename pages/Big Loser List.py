# pages/Big_Loser_List.py
import io
import csv
from collections import Counter
from typing import List, Tuple, Iterable

import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

LETTERS = list("ABCDEFGHIJ")  # A=hot … J=cold


def _only_digits(s: str) -> str:
    """Return only the digits from s."""
    return "".join(ch for ch in s if ch.isdigit())


def _parse_history(file_bytes: bytes, filename: str, pad4: bool) -> List[str]:
    """
    Accepts TXT or CSV. Returns a list of 5-char strings (winners) in file order.
    - One-item-per-line TXT is fine.
    - CSV: will look for any column that contains digit-like tokens. Headers ignored.
    - Non-digit tokens are skipped.
    """
    text = file_bytes.decode("utf-8", errors="ignore")

    winners: List[str] = []

    # Try CSV first; if sniff fails we'll fall back to line-by-line
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text.splitlines()[0])
        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)
        # If there's a header row, skip it; otherwise just use all rows
        # We'll collect any digit tokens from each row.
        for row in rows:
            for cell in row:
                token = _only_digits(str(cell))
                if token:
                    winners.append(token)
        if not winners:
            raise ValueError("No digit tokens in CSV")
    except Exception:
        # Not a CSV or no usable tokens → parse by lines
        for line in text.splitlines():
            token = _only_digits(line)
            if token:
                winners.append(token)

    # Normalize to 4/5-digit; pad 4→5 if requested
    norm: List[str] = []
    for tok in winners:
        if len(tok) == 5:
            norm.append(tok)
        elif len(tok) == 4 and pad4:
            norm.append("0" + tok)
        elif len(tok) == 4 and not pad4:
            # keep as-is (still valid for map math; we’ll zero-fill on display)
            norm.append(tok)
        else:
            # Skip outliers (3, 6+ etc.)
            continue

    # Finally enforce 5-char strings for our output columns
    norm = [w if len(w) == 5 else w.zfill(5) for w in norm]
    return norm


def _hot_to_cold_map(last_10: Iterable[str]) -> List[int]:
    """
    Given the last 10 winners (each 5 chars), return digits 0..9 sorted by
    frequency (hot→cold). Ties break by digit ascending.
    """
    cnt = Counter(int(ch) for w in last_10 for ch in w)
    # Ensure all digits appear
    for d in range(10):
        cnt.setdefault(d, 0)
    # Sort hot→cold, tie → lower digit first
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in order]


def _letters_for_map(digits_order_hot_to_cold: List[int]) -> dict:
    """
    Map each digit to its rank letter A..J (A=hot … J=cold)
    """
    return {digit: LETTERS[idx] for idx, digit in enumerate(digits_order_hot_to_cold)}


def _core_letters(winner: str, prev_map_letters: dict) -> List[str]:
    """Letters used by MR winner, on the previous map."""
    used = {prev_map_letters[int(ch)] for ch in winner}
    return sorted(used, key=lambda L: LETTERS.index(L))


def _plusminus1_union(letters: Iterable[str]) -> List[str]:
    """Union of ±1 neighborhoods for the given letters (in A..J index space)."""
    idxs = {LETTERS.index(L) for L in letters}
    u = set()
    for i in idxs:
        for j in (i - 1, i, i + 1):
            if 0 <= j < 10:
                u.add(LETTERS[j])
    return sorted(u, key=lambda L: LETTERS.index(L))


def _due_w2(prev1: str, prev2: str) -> List[int]:
    """Digits NOT present in the previous 2 winners (ascending)."""
    present = set(int(ch) for s in (prev1, prev2) if s for ch in s)
    return [d for d in range(10) if d not in present]


def _loser_list_0_9_from_curr(curr_hot_to_cold: List[int]) -> str:
    """Return 10 digits coldest→hottest as a concatenated string."""
    cold_to_hot = list(reversed(curr_hot_to_cold))
    return "".join(str(d) for d in cold_to_hot)


def _build_batch(winners: List[str]) -> pd.DataFrame:
    """
    Build the batch table from an ordered list of winners (oldest→most-recent).
    Row i uses:
      - prev map: winners[i-10 : i]  (10 draws before MR winner)
      - curr map: winners[i-9  : i+1] (MR winner + 9 before → next-draw map)
      - due_W2:   digits absent from winners[i-2], winners[i-1]
    Only rows with i >= 10 are produced.
    """
    rows = []
    for i in range(10, len(winners)):
        mr = winners[i]              # MR winner at index i
        prev10 = winners[i-10:i]     # 10 draws BEFORE mr
        curr10 = winners[i-9:i+1]    # map used for NEXT draw (mr + 9 before)

        prev_map = _hot_to_cold_map(prev10)
        curr_map = _hot_to_cold_map(curr10)

        prev_letters = _letters_for_map(prev_map)
        core = _core_letters(mr, prev_letters)
        u_letters = _plusminus1_union(core)

        prev1 = winners[i-1] if i-1 >= 0 else ""
        prev2 = winners[i-2] if i-2 >= 0 else ""
        due = _due_w2(prev1, prev2)

        row = {
            "mr_winner": int(mr),
            "prev_map_hot_to_cold": "".join(str(d) for d in prev_map),
            "curr_map_hot_to_cold": "".join(str(d) for d in curr_map),
            "core_letters": ", ".join(core) if core else "",
            "U_letters": ", ".join(u_letters) if u_letters else "",
            "due_W2": ", ".join(str(d) for d in due) if due else "",
            "loser_list_0_9": _loser_list_0_9_from_curr(curr_map),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = "index"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Loser List Batch Explorer — ±1 Neighborhood Method", layout="wide")

st.title("Loser List Batch Explorer — ±1 Neighborhood Method")

# Clear cache & state (safe across Streamlit versions)
with st.sidebar:
    if st.button("Clear cache & state"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.session_state.clear()
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# Left controls
with st.sidebar:
    st.subheader("Upload history (TXT/CSV)")
    up = st.file_uploader("Drag and drop file here", type=["txt", "csv"])

    st.markdown("### History order in file")
    order = st.radio(
        " ",
        ["Most-recent → Oldest", "Oldest → Most-recent"],
        index=1,
        label_visibility="collapsed",
    )

    pad4 = st.checkbox("Pad 4-digit items to 5 digits", value=True)
    force = st.checkbox("Recompute anyway", value=False)
    compute = st.button("Compute")

# Maintain state so the page doesn’t auto-reset
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None
if "winners_ordered" not in st.session_state:
    st.session_state.winners_ordered = []

# Compute on demand
if compute and up is not None:
    try:
        winners = _parse_history(up.read(), up.name, pad4=pad4)

        # If the file is newest→oldest and user chose "Oldest→Most-recent",
        # reverse. Otherwise keep file order.
        if order.startswith("Most-recent"):
            # File lists MR first; convert to oldest→most-recent baseline
            winners = list(reversed(winners))
        else:
            # File already oldest→most-recent
            pass

        st.session_state.winners_ordered = winners
        st.session_state.batch_df = _build_batch(winners)

    except Exception as e:
        st.error(f"Failed to compute: {e}")

elif compute and up is None:
    st.warning("Please upload a TXT/CSV of winners first.")

elif force and st.session_state.winners_ordered:
    st.session_state.batch_df = _build_batch(st.session_state.winners_ordered)

# Helper: small method refresher
with st.expander("Method refresher"):
    st.markdown(
        """
**Per row (winner `mr_winner`):**

- **prev_map_hot_to_cold** = frequency map from the **10 draws before** `mr_winner` (A=hot … J=cold).
- **curr_map_hot_to_cold** = frequency map from `mr_winner` + the **9 draws before** it (i.e., the map you’d use to guess the *next* draw).
- **core_letters** = letters used by `mr_winner` when tagged on the **previous** map.
- **U_letters** = union of ±1 neighborhoods of those letters (B → {A,B,C}, F → {E,F,G}, J → {I,J}, etc).
- **due_W2** = digits not present in the **previous two** winners (`i-2`, `i-1`).
- **loser_list_0_9** = digits **coldest→hottest** from the current map (a 10-digit string).
"""
    )

# View options
st.markdown("### View options")
col_v1, col_v2 = st.columns([1, 1])
with col_v1:
    show_oldest_first = st.checkbox("Show oldest → newest in table", value=True)
with col_v2:
    show_pairs = st.checkbox("Show pairs: seed (prev) → next", value=False)

# Batch table
st.markdown("### Batch table")
df = st.session_state.batch_df
if df is None or df.empty:
    st.info("Upload a file and click **Compute** to see results.")
else:
    view_df = df.copy()
    # Table order: by default rows are chronological (0=first row produced, which
    # corresponds to the earliest index where we have 10 winners of history).
    if not show_oldest_first:
        view_df = view_df.iloc[::-1].copy()

    st.dataframe(view_df, use_container_width=True, height=520)

    # Row detail
    st.markdown("### Row detail")
    idx = st.number_input(
        "Select row index (0 is the earliest row that has full history)",
        min_value=int(view_df.index.min()),
        max_value=int(view_df.index.max()),
        value=int(view_df.index.min()),
        step=1,
    )

    try:
        # Convert back to original df index if user is viewing reversed
        if show_oldest_first:
            row = df.loc[idx]
            base_idx = idx
        else:
            # map the visible idx back to original position
            # visible order: reversed df; original index at that position is:
            base_idx = int(view_df.index[view_df.index == idx][0])
            row = df.loc[base_idx]

        st.write(
            pd.DataFrame(
                {
                    "mr_winner": [row["mr_winner"]],
                    "prev_map_hot_to_cold": [row["prev_map_hot_to_cold"]],
                    "curr_map_hot_to_cold": [row["curr_map_hot_to_cold"]],
                    "core_letters": [row["core_letters"]],
                    "U_letters": [row["U_letters"]],
                    "due_W2": [row["due_W2"]],
                    "loser_list_0_9": [row["loser_list_0_9"]],
                }
            )
        )

        if show_pairs:
            winners = st.session_state.winners_ordered
            # MR index in the ordered winners list is base_idx (shifted by 10 because our table starts at i=10)
            mr_global_i = base_idx + 10
            seed = winners[mr_global_i - 1] if mr_global_i - 1 >= 0 else ""
            nxt = winners[mr_global_i + 1] if mr_global_i + 1 < len(winners) else ""
            st.markdown(
                f"**Pair around MR** — seed (prev) → **{seed or '—'}**, next → **{nxt or '—'}**"
            )
    except Exception as e:
        st.warning(f"Could not show row detail: {e}")
