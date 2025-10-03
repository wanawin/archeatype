import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Large Filters Planner", layout="wide")

st.title("Archetype Helper — Large Filters, Triggers & Plans")

# ---------------- Defaults ----------------
DEFAULT_HISTORY = "DC5_Midday_Full_Cleaned_Expanded.csv"
DEFAULT_FILTERS = "lottery_filters_batch_10.csv"

# ---------------- Inputs ----------------
st.sidebar.header("Planner Mode")
mode = st.sidebar.radio("Select Mode", ["Playlist Reducer", "Safe Filter Explorer"])
min_elims = st.sidebar.number_input("Min eliminations to call it ‘Large’", 0, 500, 60)
beam_width = st.sidebar.number_input("Beam width (search breadth)", 1, 20, 6)
max_steps = st.sidebar.number_input("Max steps (search depth)", 1, 50, 18)
exclude_parity = st.sidebar.checkbox("Exclude parity-wipers", value=True)

seed = st.text_input("Seed (1-back, 5 digits)*", "")
prev_seed = st.text_input("Prev seed (2-back, optional)", "")
prev_prev_seed = st.text_input("Prev-prev seed (3-back, optional)", "")

hot_digits = st.text_input("Hot digits")
cold_digits = st.text_input("Cold digits")
due_digits = st.text_input("Due digits")

st.subheader("Combo Pool")
combo_text = st.text_area("Paste combos (optional, one per line or comma-separated)")
combo_file = st.file_uploader("Or upload pool CSV", type="csv")

st.subheader("Filters")
filter_ids = st.text_area("Paste applicable Filter IDs (optional)")
uploaded_filters = st.file_uploader(
    "Upload Filters CSV (omit to use default: lottery_filters_batch_10.csv)", type="csv"
)

# ---------------- Load Pool ----------------
def load_pool():
    if combo_file is not None:
        df = pd.read_csv(combo_file)
        return df.iloc[:, 0].astype(str).tolist()
    if combo_text.strip():
        raw = [x.strip() for x in combo_text.replace(",", "\n").splitlines() if x.strip()]
        return raw
    try:
        df = pd.read_csv(DEFAULT_HISTORY)
        return df["Result"].astype(str).tolist() if "Result" in df.columns else df.iloc[:,0].astype(str).tolist()
    except:
        return []
pool = load_pool()
st.caption(f"Pool size: {len(pool)}")

# ---------------- Load Filters ----------------
def load_filters():
    if uploaded_filters is not None:
        f = pd.read_csv(uploaded_filters)
    else:
        f = pd.read_csv(DEFAULT_FILTERS)
    # Normalize column names
    f.columns = [c.strip() for c in f.columns]
    # Defensive — if elim_count_on_pool missing, add with NaN
    if "elim_count_on_pool" not in f.columns:
        f["elim_count_on_pool"] = None
    # Filter by IDs if provided
    if filter_ids.strip():
        ids = [x.strip() for x in filter_ids.replace(",", " ").split()]
        f = f[f["id"].astype(str).isin(ids)]
    return f

try:
    filters_df = load_filters()
    st.caption(f"Filters loaded: {len(filters_df)}")
except Exception as e:
    st.error(f"Failed to load filters: {e}")
    filters_df = pd.DataFrame()

# ---------------- Run Planner ----------------
if st.button("▶ Run Planner"):
    if not pool:
        st.warning("No pool loaded.")
    elif filters_df.empty:
        st.warning("No filters evaluated.")
    else:
        # Attempt to apply filters
        elim_col = "elim_count_on_pool"
        if elim_col not in filters_df.columns:
            st.info("Your filters file does not have elim_count_on_pool — running without elimination metric.")
            scored_df = filters_df.copy()
        else:
            scored_df = filters_df.copy()

        # If mode needs different logic later you can branch here
        if mode == "Playlist Reducer":
            st.success(f"Playlist Reducer running on {len(scored_df)} filters")
        else:
            st.success(f"Safe Filter Explorer running on {len(scored_df)} filters")

        # Example filtering
        # You can expand this part with your seed-matching logic etc.
        st.write(scored_df.head())
