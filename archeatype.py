# /archeatype/arechatype.py
# Minimal, safe Streamlit page to view archetype CSVs (if present)

from __future__ import annotations
import io
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Archetype Safe Filter Explorer", layout="wide")
st.title("Archetype Safe Filter Explorer")

# Where to look for outputs the recommender would have written
OUT_DIR = Path(".")

# Find any reasonable filenames you might have meant
candidates = (
    list(OUT_DIR.glob("arch_*.csv")) +
    list(OUT_DIR.glob("archetype_*.csv")) +
    list(OUT_DIR.glob("*archetype*.csv")) +
    list(OUT_DIR.glob("*archetypes*.csv"))
)

if not candidates:
    st.info(
        "No archetype CSVs found in the repo root. "
        "Once your recommender writes files like 'arch_seed_archetype_perf.csv' "
        "or 'archetype_filter_perf.csv', they’ll appear here automatically."
    )
else:
    st.caption(f"Found {len(candidates)} file(s). Click a header to preview & download.")
    for f in sorted(candidates):
        with st.expander(f.name, expanded=False):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue

            # Light preview (don’t freeze the app on huge files)
            show_all = st.toggle(f"Show all rows for {f.name}", value=False, key=f"all_{f.name}")
            preview = df if show_all else df.head(500)
            st.dataframe(preview, use_container_width=True, hide_index=True)

            # Download
            st.download_button(
                label=f"Download {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                mime="text/csv",
                type="primary",
                key=f"dl_{f.name}",
            )

st.markdown("---")
st.caption(
    "Tip: If you meant to put this page under 'archetype', you can rename the folder/file. "
    "Just keep the import line `import streamlit as st` at the very top."
)
