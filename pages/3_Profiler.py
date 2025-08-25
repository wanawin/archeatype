import importlib, streamlit as st
try:
    app = importlib.import_module("archetype_lab_app")
    importlib.reload(app)
except Exception as e:
    st.error(f"Could not load Profiler: {e}")
