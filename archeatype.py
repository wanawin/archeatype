# archeatype/archeatype.py — minimal, safe CSV explorer for archetype outputs
from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Archetype Safe Filter Explorer", layout="wide")

st.title("Archetype Safe Filter Explorer")

root = Path(".")

# Any archetype exports we might produce
PATTERNS = [
    "archetype_export_*.csv",
    "arch_seed_archetype_perf.csv",
    "archetype_filter_perf.csv",
    "archetype_*_perf*.csv",
]

# Gather files once
files: list[Path] = []
for pat in PATTERNS:
    files.extend(sorted(root.glob(pat)))

if not files:
    st.info(
        "No archetype CSVs found in the repo root. When files like "
        "'arch_seed_archetype_perf.csv' or 'archetype_filter_perf.csv' exist, "
        "they’ll show up here."
    )
else:
    st.write(f"Found **{len(files)}** file(s). Click to preview & download.")
    for i, f in enumerate(files):
        with st.expander(f.name, expanded=(i == 0)):
            # Read defensively
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue

            # Unique control keys per file (avoid DuplicateElementKey errors)
            max_rows = int(max(1, len(df)))
            default_rows = min(100, max_rows)
            n_show = st.number_input(
                "Rows to show",
                min_value=1,
                max_value=max_rows,
                value=default_rows,
                step=min(50, max_rows),
                key=f"rows_{i}",
            )

            st.dataframe(df.head(int(n_show)), use_container_width=True)

            # Download buttons
            st.download_button(
                "Download CSV",
                data=f.read_bytes(),
                file_name=f.name,
                mime="text/csv",
                key=f"dl_csv_{i}",
            )

            # Optional: quick Excel export
            if st.checkbox("Also offer Excel (.xlsx) export", key=f"xlsx_{i}"):
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                    df.to_excel(xw, index=False, sheet_name="data")
                st.download_button(
                    "Download Excel",
                    data=bio.getvalue(),
                    file_name=f.with_suffix(".xlsx").name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{i}",
                )
