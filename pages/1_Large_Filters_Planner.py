import streamlit as st
import pandas as pd

# ----------------------------
# Loaders
# ----------------------------
@st.cache_data
def load_filters_csv(source):
    if isinstance(source, str):
        return pd.read_csv(source, dtype=str)
    else:
        return pd.read_csv(source, dtype=str)

def parse_list(text):
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]

# ----------------------------
# UI Layout
# ----------------------------
st.set_page_config(page_title="Archetype Helper — Large Filters Planner", layout="wide")

st.title("Archetype Helper — Large Filters, Triggers & Plans")

# ---- Mode toggle
mode = st.radio("Select mode", ["Playlist Reducer", "Safe Filter Explorer"], horizontal=True)

# ---- Parameters
min_elims = st.number_input("Min eliminations to call it 'Large'", min_value=0, value=60)
beam_width = st.number_input("Beam width (search breadth)", min_value=1, value=6)
max_steps = st.number_input("Max steps (search depth)", min_value=1, value=18)
exclude_parity = st.checkbox("Exclude parity-wipers", value=False)

# ---- Seeds
st.subheader("Seed (prev draw)")
seed = st.text_input("Seed (1-back, 5 digits) *")
prev_seed = st.text_input("Prev seed (2-back, optional)")
prev_prev_seed = st.text_input("Prev-prev seed (3-back, optional)")

# ---- Hot / Cold / Due digits
st.subheader("Hot / Cold / Due digits")
hot_digits = st.text_input("Hot digits")
cold_digits = st.text_input("Cold digits")
due_digits = st.text_input("Due digits")

# ---- Combo Pool
st.subheader("Combo Pool")
st.caption("Paste combos as CSV text (must have a 'Result' column)")
pool_text = st.text_area("Paste combos")
pool_file = st.file_uploader("Or upload combo pool CSV", type=["csv"])

# ---- Filters section
st.subheader("Filters")
ids_text = st.text_area("Paste applicable Filter IDs (optional)")
filters_file = st.file_uploader("Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)", type=["csv"])
source = filters_file if filters_file else "lottery_filters_batch_10.csv"

# ----------------------------
# Logic
# ----------------------------
filters_df_full = load_filters_csv(source)
ids = set(parse_list(ids_text))
if ids:
    id_str = filters_df_full["id"].astype(str)
    filters_df = filters_df_full[id_str.isin(ids)].copy()
else:
    filters_df = filters_df_full.copy()

if exclude_parity and "parity_wiper" in filters_df.columns:
    filters_df = filters_df[~filters_df["parity_wiper"]]
if "enabled" in filters_df.columns:
    filters_df = filters_df[filters_df["enabled"] == "True"]

st.write(f"Filters loaded: {len(filters_df)}")

# ---- Pool loading
if pool_text:
    from io import StringIO
    pool_df = pd.read_csv(StringIO(pool_text))
elif pool_file:
    pool_df = pd.read_csv(pool_file)
else:
    pool_df = pd.DataFrame()

if not pool_df.empty:
    st.write(f"Pool size: {len(pool_df)}")

# ---- Run button
if st.button("Run Planner"):
    if pool_df.empty:
        st.warning("Paste CSV combos or upload a CSV to continue.")
    else:
        with st.spinner("Evaluating filters on current pool…"):
            st.success(f"{mode} running on {len(filters_df)} filters")
            # Just show first few filters for now
            if not filters_df.empty:
                st.dataframe(filters_df.head(50))
