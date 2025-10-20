# loserlist w filters.py — Corrected Version
# Self-contained app that defines its own variables, computes numeric substitutions,
# generates LL and CF filters, and outputs a ready-to-copy CSV block.
# No external CSVs (like lottery_filters_batch10.csv) are ever read or called.

import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Loser List Filter Generator", layout="wide")
st.title("Loser List Filter Generator (Self-Contained)")

# =====================================
#  Define numeric variable groups inline
# =====================================
core_digits = [0, 9, 1, 2, 4]
cooled_digits = [2, 4, 7, 9]
loser_7_9 = [7, 8, 9]
new_core_digits = [3, 5, 6]
mirror_map = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}

# =====================================
#  Filter generation logic (LL + CF)
# =====================================
def generate_filters():
    filters = []

    # --- LL filters ---
    filters.append({
        "id": "LL001",
        "name": "If combo contains ≥3 digits from core digits, eliminate combo",
        "enabled": True,
        "applicable_if": "True",
        "expression": f"sum(d in {core_digits} for d in combo_digits) >= 3",
    })

    filters.append({
        "id": "LL002",
        "name": "If combo has any cooled digit repeating twice, eliminate combo",
        "enabled": True,
        "applicable_if": "True",
        "expression": f"any(combo_digits.count(d) > 1 for d in {cooled_digits})",
    })

    filters.append({
        "id": "LL003",
        "name": "If combo includes any digit from loser 7-9 more than twice, eliminate combo",
        "enabled": True,
        "applicable_if": "True",
        "expression": f"any(combo_digits.count(d) > 2 for d in {loser_7_9})",
    })

    # --- CF filters ---
    filters.append({
        "id": "CF001",
        "name": "If seed contains J, eliminate combos not containing E, F, or H",
        "enabled": True,
        "applicable_if": "'J' in seed_letters",
        "expression": "not any(x in combo_letters for x in ['E','F','H'])",
    })

    filters.append({
        "id": "CF002R",
        "name": "Eliminate if core size in {2,3,5}",
        "enabled": True,
        "applicable_if": "True",
        "expression": "core_size in [2,3,5]",
    })

    filters.append({
        "id": "CF003",
        "name": "Eliminate if core contains J",
        "enabled": True,
        "applicable_if": "True",
        "expression": "'J' in core_letters",
    })

    filters.append({
        "id": "CF004",
        "name": "Keep bias if core contains B or E",
        "enabled": True,
        "applicable_if": "True",
        "expression": "any(x in core_letters for x in ['B','E'])",
    })

    return filters

# Generate filters
filters = generate_filters()

# ============================
#  Display all generated data
# ============================
st.subheader("Defined Variables (for Verification)")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**core_digits:** {core_digits}")
    st.markdown(f"**cooled_digits:** {cooled_digits}")
    st.markdown(f"**loser_7_9:** {loser_7_9}")
with col2:
    st.markdown(f"**new_core_digits:** {new_core_digits}")
    st.markdown(f"**mirror_map:** {mirror_map}")

# Show filters on screen in CSV format
csv_preview = io.StringIO()
pd.DataFrame(filters).to_csv(csv_preview, index=False)
st.subheader("Generated Filters (Copy/Paste into Main CSV)")
st.code(csv_preview.getvalue())

# Download link
st.download_button(
    label="Download LL+CF Filters CSV",
    data=csv_preview.getvalue().encode('utf-8'),
    file_name="loserlist_filters_LL_CF.csv",
    mime="text/csv"
)

st.caption("This file contains only LL and CF filters — self-contained, numeric substitutions complete.")
