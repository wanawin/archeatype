import streamlit as st
import importlib
import streamlined_streamlit_app as app  # your recommender UI at repo root
importlib.reload(app)
# shim page – renders the recommender UI you already have
from streamlined_streamlit_app import *
