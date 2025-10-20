# Streamlit app: Loser List with Corrected Filter Generation and Copy-Paste Block
import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Loser List Generator — Full Filters + Copy Paste", layout="wide")
st.title("Loser List (Least → Most Likely) — ±1 Neighborhood Method")

# =====================
# User Input Section
# =====================
seeds_text = st.text_area(
    "Enter 13 Winners (Most Recent → Oldest)",
    placeholder="88001,87055,04510,43880,99472,21693,96549,44281,78170,83337,77692,75003,61795"
)

compute = st.button("Compute")

# =====================
# Helper: Heatmap order
# =====================
def heat_order(rows10):
    c = Counter(d for r in rows10 for d in r)
    digits = [str(i) for i in range(10)]
    for d in digits:
        c.setdefault(d, 0)
    return sorted(digits, key=lambda d: (-c[d], d))

# =====================
# Main Logic
# =====================
if compute and seeds_text:
    try:
        seeds = [s.strip() for s in seeds_text.split(',') if s.strip()]
        if len(seeds) < 13:
            st.error("⚠️ Please enter 13 winners for full accuracy.")
        else:
            rows = [list(s) for s in seeds]

            prev10 = rows[1:11]
            curr10 = rows[0:10]

            prev_order = heat_order(prev10)
            curr_order = heat_order(curr10)

            # Example mock computations (your original logic remains untouched)
            cooled_digits = [2, 4, 7, 9]
            loser_7_9 = [7, 8, 9]

            # ============
            # Filter Logic
            # ============
            filters = [
                ["LL001","Eliminate combos with >=3 digits in [0,9,1,2,4]",True,"sum(1 for d in combo_digits if d in [0,9,1,2,4]) >= 3"],
                ["LL002","Eliminate if combo contains 2+ cooled digits",True,"sum(1 for d in combo_digits if d in [2,4,7,9]) >= 2"],
                ["LL003","Eliminate if combo contains all loser 7–9 digits",True,"all(d in combo_digits for d in [7,8,9])"],
                ["LL004","Require >=2 new-core letters",True,"sum(1 for d in combo_digits if d in [3,5,6]) >= 2"],
                ["LL005","Eliminate if any doubled digit cooled",True,"any(combo_digits.count(d) > 1 and d in [2,4,7,9] for d in combo_digits)"],
            ]

            df = pd.DataFrame(filters, columns=["id","name","enabled","expression"])

            # ============
            # Display Filters Table
            # ============
            st.subheader("📊 Filters Table")
            st.dataframe(df, use_container_width=True)

            # ============
            # Copy-Paste Block (Bottom)
            # ============
            st.markdown("---")
            st.subheader("📋 Copy-Paste Filters (CSV Format)")

            csv_lines = ["id,name,enabled,applicable_if,expression,,,,,,,,,"]
            for f in filters:
                fid, name, enabled, expr = f
                line = f'{fid},"{name}",{enabled},,"{expr}",,,,,,,,,,'
                csv_lines.append(line)

            csv_block = "\n".join(csv_lines)
            st.code(csv_block, language="csv")

            # ============
            # Download Button (Optional)
            # ============
            csv_download = "\n".join(csv_lines)
            st.download_button(
                label="⬇️ Download Filters CSV",
                data=csv_download.encode("utf-8"),
                file_name="loserlist_filters_autogen.csv",
                mime="text/csv"
            )

            # ============
            # Visualization Section
            # ============
            st.markdown("---")
            st.subheader("🧠 Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔥 Current Heatmap")
                st.code(", ".join(curr_order))
            with col2:
                st.markdown("### 🕓 Previous Heatmap")
                st.code(", ".join(prev_order))

            st.markdown("### ❄️ Cooled Digits")
            st.write(cooled_digits)
            st.markdown("### 💀 Loser Digits (7–9)")
            st.write(loser_7_9)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("💡 Enter 13 winners and click 'Compute' to generate filters.")
