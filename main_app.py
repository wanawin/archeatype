# app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="DC5 Toolkit — Main", layout="wide")
st.title("DC5 Toolkit — Main")

# Shared defaults other pages can read via st.session_state
st.session_state.setdefault("WINNERS_CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
st.session_state.setdefault("FILTERS_CSV", "lottery_filters_batch_10.csv")
st.session_state.setdefault("POOL_CSV", "today_pool.csv")

st.markdown("""
### What’s here
- **Archetype — Large Filters Planner**  
  Paste applicable filter IDs + a pool; see *large* filters, trigger filters, and a **winner-preserving plan** to reach ≤45.
- **Recommender Runner**  
  Classic end-to-end run that applies all relevant filters and writes outputs (sequence, avoid_pairs, one-pager, etc.).
- **Profiler (Build/Refresh)**  
  Utilities to build/refresh your case history tables used by other tools.
- **Archetype Analyzer (history CSVs)**  
  Replays history to produce the 4 archetype CSVs (dimension and composite stats, plus top/danger signals).
""")

with st.expander("Global defaults (used by all pages)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        winners = st.text_input("Winners CSV", st.session_state["WINNERS_CSV"])
    with c2:
        filters_ = st.text_input("Filters CSV", st.session_state["FILTERS_CSV"])
    with c3:
        pool = st.text_input("Pool CSV", st.session_state["POOL_CSV"])

    if st.button("Save defaults", type="primary"):
        st.session_state["WINNERS_CSV"] = winners
        st.session_state["FILTERS_CSV"] = filters_
        st.session_state["POOL_CSV"] = pool
        st.success("Saved. All pages will read these defaults via session_state.")
