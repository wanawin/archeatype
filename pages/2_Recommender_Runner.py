import importlib, streamlit as st
try:
    app = importlib.import_module("streamlined_streamlit_app")
    importlib.reload(app)
except Exception as e:
    st.error(f"Could not load Recommender Runner: {e}")
