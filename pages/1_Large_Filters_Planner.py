import streamlit as st
import pandas as pd

st.title("Large Filters Planner")

# ---- Mode selection ----
mode = st.sidebar.radio(
    "Mode",
    ["Playlist Reducer", "Safe Filter Explorer"],
    help="Playlist Reducer = original winner-preserving logic, faster bigger cuts.\n"
         "Safe Filter Explorer = lower elimination threshold, deeper search to surface more high-safety filters."
)

if mode == "Playlist Reducer":
    default_min_elims = 120
    default_beam = 5
    default_steps = 15
else:  # Safe Filter Explorer
    default_min_elims = 60
    default_beam = 6
    default_steps = 18

min_elims = st.sidebar.number_input(
    "Min eliminations to call it ‘Large’",
    min_value=1, max_value=99999,
    value=default_min_elims
)
beam_width = st.sidebar.number_input(
    "Beam width (search breadth)",
    min_value=1, max_value=50,
    value=default_beam
)
max_steps = st.sidebar.number_input(
    "Max steps (search depth)",
    min_value=1, max_value=50,
    value=default_steps
)

exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=True)

# ---- Combo Pool Input ----
st.subheader("Combo Pool")
pool_text = st.text_area("Paste combos here (one per line):", height=150)
pool_file = st.file_uploader("Or upload combo pool CSV (must have a 'Result' column)", type=["csv"])

if pool_text.strip():
    pool = [x.strip() for x in pool_text.splitlines() if x.strip()]
elif pool_file:
    df_pool = pd.read_csv(pool_file)
    if 'Result' in df_pool.columns:
        pool = df_pool['Result'].astype(str).tolist()
    else:
        st.error("CSV must have a 'Result' column.")
        st.stop()
else:
    st.info("Upload or paste a combo pool to continue.")
    st.stop()

pool_size = len(pool)

# ---- Filters Input ----
st.subheader("Filters")
filters_text = st.text_area("Paste filter CSV content here (optional):", height=150)
filters_file = st.file_uploader("Or upload filters CSV", type=["csv"])

if filters_text.strip():
    from io import StringIO
    filters_df = pd.read_csv(StringIO(filters_text))
elif filters_file:
    filters_df = pd.read_csv(filters_file)
else:
    st.info("Upload or paste filter CSV to continue.")
    st.stop()

# ---- Build candidate filter set ----
large_df = filters_df.copy()
if "elim_count_on_pool" in large_df.columns:
    large_df = large_df[large_df["elim_count_on_pool"] >= int(min_elims)]
else:
    st.error("Filters CSV must have an 'elim_count_on_pool' column.")
    st.stop()

if exclude_parity and "parity_wiper" in large_df.columns:
    large_df = large_df[~large_df["parity_wiper"]]

st.subheader("Candidate Large Filters")
st.write(f"{len(large_df)} filters qualify as 'Large' with current settings.")
st.dataframe(large_df)

# --- Placeholder for your original planning algorithm ---
# It will continue to use `large_df`, `beam_width`, `max_steps`, and `pool`.
# Nothing else in the planner logic is changed.
