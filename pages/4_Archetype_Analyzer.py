
# archetype_app.py — run & view archetype → filter safety
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import streamlit as st

# ---- config (edit if your filenames differ)
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
OUT_DIR     = Path(".")

# ---- try to import the analyzer (must exist as archetype_safety.py in repo)
try:
    from archetype_safety import analyze_archetype_safety
    ANALYZER_OK = True
except Exception as e:
    ANALYZER_OK = False
    ANALYZER_ERR = str(e)

def _download_btn(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        return
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(label, buff.getvalue(), file_name=filename, mime="text/csv")

def _load_if_exists(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
    return None

st.set_page_config(page_title="Archetype → Filter safety", layout="wide")
st.title("Archetype → Filter safety (history)")

st.caption(
    "This tool replays history to see, for each **filter**, which **seed archetypes** "
    "it tended to **pass** (safe) or **fail** (unsafe). Use it to decide when your large filters are safest."
)

# Controls
with st.sidebar:
    st.header("Run analysis")
    winners_path = st.text_input("Winners CSV", WINNERS_CSV)
    filters_path = st.text_input("Filters CSV", FILTERS_CSV)
    min_support  = st.number_input("Min applicable days for a signal", 1, 9999, 12)
    min_lift     = st.number_input("Min lift vs baseline for TOP signals", 1.00, 10.0, 1.10, step=0.01)
    max_lag      = st.number_input("Max lift vs baseline for DANGER signals", 0.10, 1.00, 0.90, step=0.01)
    
