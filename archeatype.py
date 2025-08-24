from __future__ import annotations
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Archetype Safe Filter Explorer", layout="wide")
st.title("Archetype Safe Filter Explorer")

# Where to look for exports
SEARCH_DIRS = [Path("."), Path("outputs"), Path("out"), Path("data")]

# Filename filter (substring match). Clear to show all CSVs.
needle = st.text_input(
    "Filter file names (substring, optional)",
    value="archetype",
    help="Examples: archetype, seed, export, perf. Leave blank to show all CSV files."
).strip().lower()

# Collect CSVs and de-duplicate by absolute path
found: list[Path] = []
seen: set[str] = set()
for d in SEARCH_DIRS:
    if d.exists():
        for p in d.glob("*.csv"):
            if needle and needle not in p.name.lower():
                continue
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                found.append(p)

found = sorted(found, key=lambda p: (str(p.parent), p.name))

if not found:
    st.info(
        "No matching CSVs found. Try clearing the filename filter above or "
        "rename your export to include a helpful word (e.g., 'archetype'). "
        "Searched in: " + ", ".join(str(d) for d in SEARCH_DIRS)
    )
else:
    st.caption(f"Found {len(found)} CSV file(s). Click to preview & download.")
    for i, f in enumerate(found):
        with st.expander(f"{f.parent}/**{f.name}**", expanded=False):
            # Try reading as text CSV; keep values as strings for safety.
            try:
                df = pd.read_csv(f, dtype=str, engine="python", on_bad_lines="skip")
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue

            # Basic file stats
            st.write(f"Rows: **{len(df):,}** &nbsp;&nbsp; Columns: **{len(df.columns):,}**")

            # Unique widget keys per file to avoid DuplicateElementKey
            show_all = st.toggle("Show all rows", value=False, key=f"show_{i}")
            max_rows = None if show_all else 500

            st.dataframe(
                df if max_rows is None else df.head(max_rows),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="Download this CSV",
                data=f.read_bytes(),
                file_name=f.name,
                mime="text/csv",
                type="primary",
                key=f"dl_{i}",
            )
