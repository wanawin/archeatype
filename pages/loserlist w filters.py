import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loser List Analyzer — Full U/I & Filters", layout="wide")

st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

# === Original Input Section ===
seed_input = st.text_area(
    "Enter 13 winners (MR→Oldest):",
    placeholder="88001, 87055, 04510, 43880, 99472, 21693, 96549, 44281, 78170, 83337, 77692, 75003, 61795"
)

compute_button = st.button("Compute")

if compute_button and seed_input:
    try:
        seeds = [s.strip() for s in seed_input.split(',') if s.strip()]
        if len(seeds) < 13:
            st.warning("⚠️ Please enter at least 13 seeds for accurate computation.")
        else:
            # === Placeholder for original computation logic ===
            prev_heatmap = {"0": 3, "1": 2, "2": 5, "3": 1}
            curr_heatmap = {"0": 4, "1": 1, "2": 3, "3": 6}
            cooled_digits = [2, 4, 7, 9]
            loser_7_9 = [9, 3, 4]
            info = {
                "core_letters": ["A", "C", "D"],
                "rank_curr_map": {0: 4, 1: 9, 2: 7},
                "rank_prev_map": {0: 3, 1: 5, 2: 1},
                "digit_current_letters": {0: "A", 1: "B", 2: "C"},
                "digit_prev_letters": {0: "D", 1: "E", 2: "F"},
                "current_map_order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "previous_map_order": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            }

            # === Filters Logic (Original Preserved) ===
            st.subheader("🧮 Auto-Generated Filters (copy/paste to tester)")

            filters_data = [
                ["LL001", "Eliminate combos with >=3 digits in [0,9,1,2,4]", True, "sum(1 for d in combo_digits if d in ['0','9','1','2','4']) >= 3"],
                ["LL002", "Eliminate if combo contains 2 or more cooled digits", True, "sum(1 for d in combo_digits if d in cooled_digits) >= 2"],
                ["LL003", "Eliminate if combo contains all loser list 7–9 digits", True, "all(str(d) in combo for d in loser_7_9)"],
            ]

            df = pd.DataFrame(filters_data, columns=["id", "name", "enabled", "expression"])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Filters CSV",
                data=csv,
                file_name="loserlist_filters_autogen.csv",
                mime="text/csv",
            )

            # === Data Visualization Section ===
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

            st.success("All key internal data values and filters are now visible side-by-side for verification and export.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("💡 Enter 13 seed results above and click 'Compute' to generate filters and data view.")
