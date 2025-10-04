import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Large Filters Planner", layout="wide")

st.title("Large Filters Planner")

# ----------------------------
# SEED CONTEXT
# ----------------------------
st.header("Seed Context")
col1, col2, col3 = st.columns(3)
seed = col1.text_input("Seed (prev draw)")
prev_seed_2 = col2.text_input("Prev Seed (2-back)")
prev_seed_3 = col3.text_input("Prev Prev Seed (3-back)")

# ----------------------------
# HOT / COLD / DUE DIGITS
# ----------------------------
st.header("Hot / Cold / Due digits (optional)")
col1, col2, col3 = st.columns(3)
hot_digits = col1.text_input("Hot digits")
cold_digits = col2.text_input("Cold digits")
due_digits = col3.text_input("Due digits")

# ----------------------------
# COMBO POOL
# ----------------------------
st.header("Combo Pool")
combo_text = st.text_area("Paste combos (comma/space/newline separated):")
combo_file = st.file_uploader("Or upload combo pool CSV", type=["csv"])

pool_df = pd.DataFrame()
if combo_text:
    combos = []
    for line in combo_text.replace(",", "\n").splitlines():
        val = line.strip()
        if val:
            combos.append(val)
    pool_df = pd.DataFrame({"Result": combos})
elif combo_file:
    pool_df = pd.read_csv(combo_file)
    if "Result" not in pool_df.columns:
        st.warning("Uploaded CSV must have a column named 'Result'")

if not pool_df.empty:
    st.success(f"Pool loaded: {len(pool_df)} combos")

# ----------------------------
# FILTER INPUT
# ----------------------------
st.header("Filters")
filter_ids_text = st.text_area("Paste applicable Filter IDs (optional)")
filter_file = st.file_uploader("Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)", type=["csv"])

filters_df = pd.DataFrame()
if filter_ids_text:
    ids = []
    for part in filter_ids_text.replace(",", "\n").splitlines():
        v = part.strip()
        if v:
            ids.append(v)
    filters_df = pd.DataFrame({"id": ids})
elif filter_file:
    filters_df = pd.read_csv(filter_file)

if not filters_df.empty:
    st.success(f"Filters loaded: {len(filters_df)}")

# ----------------------------
# PLANNER MODE / SETTINGS
# ----------------------------
st.sidebar.subheader("Mode")
mode = st.sidebar.radio("Select mode", ["Playlist Reducer", "Safe Filter Explorer"])
min_elim = st.sidebar.number_input("Min eliminations to call it 'Large'", value=60)
beam_width = st.sidebar.number_input("Beam width (search breadth)", value=6)
max_steps = st.sidebar.number_input("Max steps (search depth)", value=18)
exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=False)

# ----------------------------
# RUN BUTTON
# ----------------------------
if st.button("Run Planner"):
    if pool_df.empty:
        st.error("Please provide a combo pool first")
    elif filters_df.empty:
        st.warning("No filter IDs loaded — running with no filters.")
    else:
        st.info(f"Running {mode} with {len(filters_df)} filters and {len(pool_df)} combos...")
        # Placeholder logic — replace with your filter logic
        results = pool_df.copy()
        st.dataframe(results.head(50))
        st.success("Planner finished (demo output).")

