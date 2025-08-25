import streamlit as st
import importlib
import archetype_large_filter_scan as app  # lives at repo root
importlib.reload(app)  # run the page code each rerun
# shim page – renders the large-filters planner
from archetype_large_filter_scan import *
