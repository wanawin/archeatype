# ✅ Restored visibility for all key data in Loser List app (side-by-side layout)
# Displays full lists: previous heatmap, current heatmap, loser list, cooled digits, etc.

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loser List Analyzer", layout="wide")

st.title("Loser List Analyzer (Full Data Display)")

# === Placeholder for your existing logic ===
# (No changes made to your computational or parsing logic)
# We assume variables like prev_heatmap, curr_heatmap, loser_7_9, cooled_digits, etc.
# already exist in the environment after your logic runs.

# Simulated example placeholders so the UI section runs (delete if not needed)
try:
    prev_heatmap
except NameError:
    prev_heatmap = {"0": 3, "1": 2, "2": 5, "3": 1}
try:
    curr_heatmap
except NameError:
    curr_heatmap = {"0": 4, "1": 1, "2": 3, "3": 6}
try:
    cooled_digits
except NameError:
    cooled_digits = [2, 4, 7]
try:
    loser_7_9
except NameError:
    loser_7_9 = [8, 5, 3]
try:
    info
except NameError:
    info = {
        "core_letters": ["A", "C", "D"],
        "rank_curr_map": {0: 4, 1: 9, 2: 7},
        "rank_prev_map": {0: 3, 1: 5, 2: 1},
        "digit_current_letters": {0: "A", 1: "B", 2: "C"},
        "digit_prev_letters": {0: "D", 1: "E", 2: "F"},
        "current_map_order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "previous_map_order": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    }

# === Display Section Restored (Side-by-Side) ===

st.subheader("📊 Full Data Overview (Side-by-Side View)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔥 Current Heatmap")
    st.dataframe(pd.DataFrame(list(curr_heatmap.items()), columns=["Digit", "Heat Value"]))

    st.markdown("### ❄️ Cooled Digits")
    st.write(cooled_digits)

with col2:
    st.markdown("### 🕓 Previous Heatmap")
    st.dataframe(pd.DataFrame(list(prev_heatmap.items()), columns=["Digit", "Heat Value"]))

    st.markdown("### 🧩 Loser List 7–9")
    st.write(loser_7_9)

with col3:
    st.markdown("### 🧠 Info Object Summary")
    st.json({
        "Core Letters": info.get("core_letters"),
        "Rank Current Map": info.get("rank_curr_map"),
        "Rank Previous Map": info.get("rank_prev_map"),
        "Digit → Current Letter": info.get("digit_current_letters"),
        "Digit → Previous Letter": info.get("digit_prev_letters"),
        "Current Map Order": info.get("current_map_order"),
        "Previous Map Order": info.get("previous_map_order"),
    })

st.success("All key internal data values are now visible side-by-side for debugging and verification.")
