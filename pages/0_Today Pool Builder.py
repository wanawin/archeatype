# 0_Today Pool Builder.py
# Build a clean today_pool.csv from a pasted list (supports "one per line" double-spaced input)
from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Today Pool Builder", layout="wide")
st.title("Today Pool Builder")

st.caption(
    "Paste 5-digit combos as a vertical list (double-spaced is OK) or any messy text. "
    "This cleans, pads to 5 digits, de-duplicates (optional), and lets you download a ready-to-use CSV."
)

# ---------- Controls ----------
with st.sidebar:
    st.header("Options")

    input_mode = st.radio(
        "Input format",
        ["One per line (recommended)", "Freeform / mixed text"],
        help=(
            "• One per line: paste a vertical list, blank lines OK; we'll read each non-empty line.\n"
            "• Freeform: we extract all digit runs from the entire text."
        )
    )

    colname = st.text_input("CSV column name", value="combo")

    drop_dupes = st.checkbox("Drop duplicates (keep first occurrence)", value=True)

    # Short handling/padding
    if input_mode == "One per line (recommended)":
        accept_short = True
        pad_short = True
    else:
        accept_short = st.checkbox("Accept 1–4 digit tokens", value=False)
        pad_short = st.checkbox("Pad 1–4 digit tokens with zeros to 5 digits", value=True)

    save_local_copy = st.checkbox("Also write files to app folder (if permitted)", value=False)
    dated_archive = st.checkbox("Make a dated archive file in /pools/", value=True)
    archive_name = st.text_input(
        "Archive filename (uses today’s date)",
        value=f"pool_{date.today()}.csv",
        help="File will be created under a /pools/ directory"
    )

st.subheader("Paste your list below")
raw = st.text_area(
    "For 'One per line', paste one combo per line (blank lines are fine). "
    "No commas needed.",
    height=260,
    placeholder="e.g.\n01234\n\n56789\n\n00123\n..."
)

# ---------- Parse / Clean ----------
def parse_line_mode(text: str):
    """Treat each non-empty line as one item; strip to digits; handle short/long."""
    valid, padded, short_rejected, too_long = [], [], [], []
    lines = text.splitlines()
    for line in lines:
        s = re.sub(r"\D", "", line)  # keep digits only
        if not s:
            continue  # blank or no digits on this line
        if len(s) > 5:
            too_long.append(s)
            continue
        if len(s) < 5:
            if accept_short:
                valid.append(s.zfill(5) if pad_short else s.zfill(5))
                padded.append(s)
            else:
                short_rejected.append(s)
            continue
        # exactly 5
        valid.append(s)
    # total_in = count of non-empty lines containing digits
    total_in = sum(1 for line in lines if re.search(r"\d", line))
    return valid, padded, short_rejected, too_long, total_in

def parse_freeform(text: str):
    """Extract digit runs from the whole text."""
    if not text.strip():
        return [], [], [], [], 0
    runs = re.findall(r"\d+", text)
    valid, padded, short_rejected, too_long = [], [], [], []
    for r in runs:
        if len(r) == 5:
            valid.append(r)
        elif len(r) < 5:
            if accept_short:
                valid.append(r.zfill(5) if pad_short else r.zfill(5))
                padded.append(r)
            else:
                short_rejected.append(r)
        else:
            too_long.append(r)
    return valid, padded, short_rejected, too_long, len(runs)

if input_mode.startswith("One per line"):
    valid, padded, short_rejected, too_long, total_in = parse_line_mode(raw)
else:
    valid, padded, short_rejected, too_long, total_in = parse_freeform(raw)

# Deduplicate while preserving order (optional)
def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

if drop_dupes:
    valid = dedupe_preserve_order(valid)

# Final safety: enforce 5 digits and filter anything weird
valid = [re.sub(r"\D", "", x).zfill(5) for x in valid]
valid = [x for x in valid if re.fullmatch(r"\d{5}", x)]

# ---------- Build DF + Stats ----------
df = pd.DataFrame({colname: valid})

def parity(s: str) -> str:
    return "Even" if (sum(int(c) for c in s) % 2 == 0) else "Odd"

if not df.empty:
    df["parity"] = df[colname].map(parity)

total_valid = len(df)
total_padded = len(padded)
total_short_rej = len(short_rejected)
total_long = len(too_long)

# Approx dupes removed (best-effort; in line-mode total_in ≈ count of lines containing digits)
dupes_removed = 0
if drop_dupes and total_in:
    # rough: items seen - uniques - rejects
    dupes_removed = max(0, total_in - total_padded - total_short_rej - total_long - total_valid)

st.subheader("Summary")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Items seen", total_in)
with c2: st.metric("Valid 5-digit combos", total_valid)
with c3: st.metric("Padded 1–4 digit", total_padded)
with c4: st.metric(">5 digit rejected", total_long)
with c5: st.metric("Duplicates removed", dupes_removed if drop_dupes else 0)

if not df.empty:
    even_n = (df["parity"] == "Even").sum()
    odd_n  = total_valid - even_n
    st.caption(f"Parity in cleaned pool — Even: {even_n} | Odd: {odd_n}")

    st.subheader("Preview")
    st.dataframe(df.head(400), use_container_width=True, hide_index=True, height=420)

# Show what got rejected (optional)
with st.expander("Show rejected items", expanded=False):
    if total_padded: st.write(f"**Padded (accepted):** {padded}")
    if total_short_rej: st.write(f"**Short rejected (not accepted):** {short_rejected}")
    if total_long: st.write(f"**Too long (>5 digits):** {too_long}")
    if not any([total_padded, total_short_rej, total_long]):
        st.caption("— none —")

# ---------- Downloads ----------
st.subheader("Download")
if df.empty:
    st.info("Nothing to download yet. Paste a list above.")
else:
    pool_csv_name = "today_pool.csv"
    buf = io.StringIO()
    df[[colname]].to_csv(buf, index=False)
    st.download_button(
        f"Download {pool_csv_name}",
        data=buf.getvalue(),
        file_name=pool_csv_name,
        mime="text/csv",
        type="primary"
    )

    if save_local_copy:
        try:
            Path(pool_csv_name).write_text(buf.getvalue(), encoding="utf-8")
            st.success(f"Wrote {pool_csv_name} to app folder.")
        except Exception as e:
            st.warning(f"Could not write {pool_csv_name} locally: {e}")

    if dated_archive:
        pools_dir = Path("pools")
        pools_dir.mkdir(exist_ok=True)
        arch_path = pools_dir / archive_name
        arch_buf = io.StringIO()
        df[[colname]].to_csv(arch_buf, index=False)
        st.download_button(
            f"Download archive: pools/{archive_name}",
            data=arch_buf.getvalue(),
            file_name=str(arch_path).replace("\\", "/"),
            mime="text/csv",
        )
        if save_local_copy:
            try:
                arch_path.write_text(arch_buf.getvalue(), encoding="utf-8")
                st.success(f"Wrote archive to {arch_path}.")
            except Exception as e:
                st.warning(f"Could not write archive locally: {e}")

# ---------- Hints ----------
with st.expander("Tips & notes", expanded=False):
    st.markdown("""
- **One per line mode** is built for your format: a vertical list, blank lines allowed, no commas required.  
- We strip non-digits on each line; if the result is 1–4 digits we **pad to 5** (e.g., `123` → `00123`).  
- Tokens with **>5 digits** on a line are rejected to avoid guessing.  
- Turn on **dated archive** to store a pool snapshot per day (e.g. `pools/pool_YYYY-MM-DD.csv`).  
""")
