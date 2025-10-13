# pages/wnranalyzr.py
from __future__ import annotations
import io, re, csv
from collections import Counter, deque
from typing import List, Tuple
import streamlit as st
import pandas as pd

st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")

# -----------------------------
# Session & UI helpers
# -----------------------------
def ss_get(k, default):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]

def order_winners(winners: List[str], mode: str) -> List[str]:
    """Return winners in requested chronological order."""
    if mode == "Newest first":
        # Assume file list is newest→oldest already; keep as-is
        return winners
    if mode == "Oldest first":
        return list(reversed(winners))
    # Auto: detect with a simple heuristic — if first date-like token appears older, flip.
    return winners

# -----------------------------
# Robust file loaders
# -----------------------------
FIVE_DIGIT_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")

def normalize_line_to_winner(line: str) -> str | None:
    """
    Try to derive a 5-digit winner from a single line with many possible formats.
    Accepts:
      - contiguous '00458'
      - separated digits '0 0 4 5 8', '0,0,4,5,8', '0-0-4-5-8'
      - grouped '00 458', '0-045-8', etc.
    Returns None if the line cannot be normalized to exactly 5 digits.
    """
    # 1) Direct hit (contiguous 5-digit)
    m = FIVE_DIGIT_RE.search(line)
    if m:
        return m.group(1)

    # 2) Tokenize digits and glue if total digits == 5
    digits = re.findall(r"\d", line)
    if len(digits) == 5:
        return "".join(digits)

    # 3) If there are more than 5 digits, try contiguous windows
    if len(digits) > 5:
        # Favor earliest contiguous 5 digits that also appear near each other in text
        return "".join(digits[:5])

    return None

def load_txt(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    winners: List[str] = []

    # Quick sweep: collect all contiguous 5-digit hits anywhere in the text
    hits = FIVE_DIGIT_RE.findall(text)
    if hits:
        winners.extend(hits)

    # Line-by-line fallback for split formats
    for line in text.splitlines():
        w = normalize_line_to_winner(line)
        if w and w not in winners:
            winners.append(w)

    return winners

def load_csv(file_bytes: bytes) -> List[str]:
    winners: List[str] = []
    buf = io.StringIO(file_bytes.decode("utf-8-sig", errors="ignore"))
    # Try pandas first
    try:
        df = pd.read_csv(buf, dtype=str)
        # Heuristic: pick the first column that looks like 5-digit numbers
        for col in df.columns:
            series = df[col].astype(str).str.extract(FIVE_DIGIT_RE, expand=False).dropna()
            if not series.empty:
                winners.extend(series.tolist())
                break
        if winners:
            return winners
    except Exception:
        buf.seek(0)

    # Fallback: csv.reader
    buf.seek(0)
    rdr = csv.reader(buf)
    for row in rdr:
        for cell in row:
            m = FIVE_DIGIT_RE.search(str(cell))
            if m:
                winners.append(m.group(1))
                break
    return winners

def load_uploaded(file) -> List[str]:
    if not file:
        return []
    name = file.name.lower()
    data = file.read()  # bytes
    if name.endswith(".txt"):
        return load_txt(data)
    if name.endswith(".csv"):
        return load_csv(data)
    # Try both
    winners = load_txt(data)
    if not winners:
        winners = load_csv(data)
    return winners

# -----------------------------
# Hot / Cold / Due calculations
# -----------------------------
def hottest_to_coldest(winners: List[str], window: int) -> List[str]:
    pool = winners if window == 0 else winners[:window]
    cnt = Counter("".join(pool))  # count each digit over the window
    # Ensure all digits present
    for d in "0123456789":
        cnt.setdefault(d, 0)
    # Sort by high→low, tie by digit asc
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in order]

def label_hot_cold(digits_rank: List[str], hot_pct: int, cold_pct: int) -> Tuple[set, set]:
    n = len(digits_rank)
    n_hot = max(0, round(n * hot_pct / 100))
    n_cold = max(0, round(n * cold_pct / 100))
    hot = set(digits_rank[:n_hot]) if n_hot else set()
    cold = set(digits_rank[-n_cold:]) if n_cold else set()
    return hot, cold

def due_set(winners: List[str], due_lookback: int) -> set:
    if due_lookback <= 0:
        return set()
    recent = winners[:due_lookback]
    seen = set("".join(recent))
    return set("0123456789") - seen

def per_winner_counts(winners: List[str], hot: set, cold: set, due: set) -> pd.DataFrame:
    rows = []
    for w in winners:
        h = sum(ch in hot for ch in w)
        c = sum(ch in cold for ch in w)
        d = sum(ch in due for ch in w)
        rows.append({"winner": w, "hot_hits": h, "cold_hits": c, "due_hits": d, "neutral_hits": 5 - (h + c + d)})
    return pd.DataFrame(rows)

def export_txt_summary(df: pd.DataFrame) -> str:
    lines = ["winner, hot_hits, cold_hits, due_hits, neutral_hits"]
    for _, r in df.iterrows():
        lines.append(f"{r['winner']}, {r['hot_hits']}, {r['cold_hits']}, {r['due_hits']}, {r['neutral_hits']}")
    return "\n".join(lines)

# -----------------------------
# UI
# -----------------------------
st.title("DC5 Hot/Cold/Due Analyzer — v0.4 (Run button + robust loaders)")

# Sidebar – input & ordering
st.sidebar.header("1) Input")
uploaded = st.sidebar.file_uploader("Upload winners (.csv or .txt)", type=["csv", "txt"])

st.sidebar.header("2) File order")
order_mode = st.sidebar.radio("Row order in file", ["Auto (guess)", "Newest first", "Oldest first"], index=0)

st.sidebar.header("4) Hot/Cold/Due params")
window = st.sidebar.number_input("Window (trailing draws) for hot/cold ranking (0 = all)", 0, 5000, 10, step=1)
hot_pct = st.sidebar.slider("% of digits labeled Hot (by frequency)", 0, 100, 30)
cold_pct = st.sidebar.slider("% of digits labeled Cold (by frequency)", 0, 100, 30)
due_lookback = st.sidebar.number_input("Due threshold: 'not seen in last N draws'", 0, 500, 2, step=1)

st.sidebar.header("5) Export control")
prefix = st.sidebar.text_input("Export file prefix", "dc5_analysis")

# Run button
st.subheader("Run")
if st.button("▶️ Run analysis (top)", use_container_width=False):
    st.session_state["__RUN__"] = True

# Load & parse winners (do this *only* when user clicks Run)
if ss_get("__RUN__", False):
    winners = load_uploaded(uploaded)
    if not winners:
        st.warning("Loaded 0 rows. Parsed TXT: found 0 winners.\n\nCheck the file contains 5-digit winners anywhere in the text.\nLeading zeros are preserved.")
    else:
        winners = order_winners(winners, order_mode)

    # If no data, stop here but keep UI active
    if not winners:
        st.stop()

    # Compute hot/cold/due
    rank = hottest_to_coldest(winners, window)
    hot, cold = label_hot_cold(rank, hot_pct, cold_pct)
    due = due_set(winners, due_lookback)

    st.markdown("### Hottest → coldest (quick view)")
    st.write(", ".join(rank))

    # Per-winner table
    df = per_winner_counts(winners, hot, cold, due)

    st.markdown("### Pattern summaries")
    st.dataframe(df[["winner", "hot_hits", "cold_hits", "due_hits", "neutral_hits"]], use_container_width=True, height=380)

    # Downloads (no auto-refresh)
    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download per-winner CSV",
            data=csv_bytes,
            file_name=f"{prefix}_per_winner.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        txt_bytes = export_txt_summary(df).encode("utf-8")
        st.download_button(
            "Download TXT summary",
            data=txt_bytes,
            file_name=f"{prefix}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Debug info (collapsible)
    with st.expander("Debug / guards used"):
        st.write({"uploaded_rows": len(winners), "window": window, "hot_pct": hot_pct, "cold_pct": cold_pct, "due_lookback": due_lookback})
        st.write({"hot": sorted(hot), "cold": sorted(cold), "due": sorted(due)})
else:
    st.info("Upload a file, set params, then click **Run analysis (top)**.")
