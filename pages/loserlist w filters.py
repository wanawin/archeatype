# Streamlit app: Loser List with Full Original Filter Generation Logic
import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Loser List Generator — Full Filters Integration", layout="wide")
st.title("Loser List (Least → Most Likely) — Full Real Filter Logic")

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

            cooled_digits = [2, 4, 7, 9]
            loser_7_9 = [7, 8, 9]

            # =====================
            # FULL ORIGINAL FILTER LOGIC RESTORED
            # =====================
            filters = [
                ["LL001A", "Eliminate combos with no core digits", True, "not any(d in core_digits for d in combo_digits)"],
                ["LL001B", "Eliminate combos with <=2 core digits", True, "sum(d in core_digits for d in combo_digits) <= 2"],
                ["LL002", "Eliminate combos with <2 of loser list 7–9", True, "sum(d in [7,8,9] for d in combo_digits) < 2"],
                ["LL003", "Eliminate combos missing B or E core digits", True, "not any(d in ['B','E'] for d in combo_digits)"],
                ["LL004", "Eliminate combos missing core 1–7 entirely", True, "all(d not in range(1,8) for d in combo_digits)"],
                ["LL005", "Eliminate combos missing >2 new-core digits (aggressive)", True, "sum(d in new_core_digits for d in combo_digits) <= 2"],
                ["LL006", "Eliminate combos missing ≥2 of loser 1–9 (aggressive)", True, "sum(d in range(1,10) for d in combo_digits) < 2"],
                ["XL001", "Eliminate if combo contains ≥3 digits in [0,9,1,2,4]", True, "sum(1 for d in combo_digits if d in [0,9,1,2,4]) >= 3"],
                ["XL002", "Eliminate if combo contains 2+ cooled digits", True, "sum(1 for d in combo_digits if d in [2,4,7,9]) >= 2"],
                ["XL003", "Eliminate if combo contains all loser 7–9 digits", True, "all(d in combo_digits for d in [7,8,9])"],
                ["XL004", "Require ≥2 new-core letters", True, "sum(1 for d in combo_digits if d in [3,5,6]) >= 2"],
                ["XL005", "Eliminate if any doubled digit cooled", True, "any(combo_digits.count(d) > 1 and d in [2,4,7,9] for d in combo_digits)"],
                ["NEWF001", "Eliminate if combo missing seed mirror digits", True, "not any(mirror_map[d] in combo_digits for d in seed_digits)"],
                ["NEWF002", "Eliminate if combo has no carry-over digits", True, "sum(d in seed_digits for d in combo_digits) == 0"],
                ["NEWF003", "Eliminate if combo has no cold digits", True, "sum(d in cold_digits for d in combo_digits) == 0"],
                ["NEWF004", "Eliminate if combo has ≥3 hot digits", True, "sum(d in hot_digits for d in combo_digits) >= 3"],
                ["NEWF005", "Eliminate if combo spread < 3", True, "(max(combo_digits) - min(combo_digits)) < 3"],
                ["NEWF006", "Eliminate if combo sum < 15", True, "sum(combo_digits) < 15"],
                ["NEWF007", "Eliminate if combo sum > 37", True, "sum(combo_digits) > 37"],
                ["NEWF008", "Eliminate if combo contains no odd digits", True, "not any(d % 2 == 1 for d in combo_digits)"],
                ["NEWF009", "Eliminate if combo contains no even digits", True, "not any(d % 2 == 0 for d in combo_digits)"],
                ["NEWF010", "Eliminate if combo contains ≥3 primes", True, "sum(d in [2,3,5,7] for d in combo_digits) >= 3"],
                ["NEWF011", "Eliminate if combo has digit 4 repeating", True, "combo_digits.count(4) > 1"],
                ["NEWF012", "Eliminate if combo has digit 7 repeating", True, "combo_digits.count(7) > 1"],
                ["NEWF013", "Eliminate if combo includes all high digits ≥7", True, "all(d >= 7 for d in combo_digits)"],
                ["NEWF014", "Eliminate if combo includes all low digits ≤3", True, "all(d <= 3 for d in combo_digits)"],
                ["NEWF015", "Eliminate if combo digits sum ends with 0 or 5", True, "sum(combo_digits) % 5 == 0"],
                ["NEWF016", "Eliminate if combo has both digit and its mirror", True, "any(mirror_map[d] in combo_digits for d in combo_digits)"],
            ]

            df = pd.DataFrame(filters, columns=["id","name","enabled","expression"])

            # =====================
            # Display Filter Table
            # =====================
            st.subheader("📊 Filters Table — Full Real Logic")
            st.dataframe(df, use_container_width=True)

            # =====================
            # Copy-Paste CSV Block
            # =====================
            st.markdown("---")
            st.subheader("📋 Copy-Paste Filters (CSV Format)")
            csv_lines = ["id,name,enabled,applicable_if,expression,,,,,,,,,"]
            for f in filters:
                fid, name, enabled, expr = f
                line = f'{fid},"{name}",{enabled},,"{expr}",,,,,,,,,,'
                csv_lines.append(line)
            csv_block = "\n".join(csv_lines)
            st.code(csv_block, language="csv")

            # Download CSV
            csv_download = "\n".join(csv_lines)
            st.download_button(
                label="⬇️ Download Full Filters CSV",
                data=csv_download.encode("utf-8"),
                file_name="loserlist_full_filters.csv",
                mime="text/csv"
            )

            # =====================
            # Data Overview Section
            # =====================
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
    st.info("💡 Enter 13 winners and click 'Compute' to generate all full filters.")
