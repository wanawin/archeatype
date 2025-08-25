import importlib, streamlit as st
try:
    app = importlib.import_module("archetype_large_filter_scan")
    importlib.reload(app)
except Exception as e:
    st.error(f"Could not load Large Filters Planner: {e}")
