from __future__ import annotations  # <-- delete this line if you prefer Option A
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Archetype Safe Filter Explorer", layout="wide")
st.title("Archetype Safe Filter Explorer")

OUT_DIR = Path(".")

# Look for archetype CSVs written by the recommender
candidates = (
    list(OUT_DIR.glob("arch_*.csv")) +
    list(OUT_DIR.glob("archetype_*.csv")) +
    list(OUT_DIR.glob("*archetype*.csv")) +
    list(OUT_DIR.glob("*archetypes*.csv"))
)

if not candidates:
    st.info(
        "No archetype CSVs found in the repo root. "
        "When files like 'arch_seed_archetype_perf.csv' or 'archetype_filter_perf.csv' "
        "exist, they’ll show up here."
    )
else:
    st.caption(f"Found {len(candidates)} file(s). Click to preview & download.")
    for f in sorted(candidates):
        with st.expander(f.name, expanded=False):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue

            show_all = st.toggle(f"Show all rows for {f.name}", value=False, key=f"all_{f.name}")
            st.dataframe(df if show_all else df.head(500), use_container_width=True, hide_index=True)

            st.download_button(
                label=f"Download {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                mime="text/csv",
                type="primary",
                key=f"dl_{f.name}",
            )
