import streamlit as st
import importlib
import archetype_lab_app as app  # build/refresh utilities at repo root
importlib.reload(app)
# shim page – renders the analyzer that writes the 4 archetype CSVs
from archetype_app import *
