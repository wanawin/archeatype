# archetype_app.py — run & view archetype → filter safety
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import streamlit as st

# ---- config (edit if your filenames differ)
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
OUT_DIR     = Path(".")

# ---- try to import the analyzer (must exist as archetype_safety.py in repo)
try:
    from archetype_safety import analyze_archetype_safety
    ANALYZER_OK = True
except Exception as e:
    ANALYZER_OK = False
    ANALYZER_ERR = str(e)

def _download_btn(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        return
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(label, buff.getvalue(), file_name=filename, mime="text/csv")

def _load_if_exists(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
    return None

st.set_page_config(page_title="Archetype → Filter safety", layout="wide")
st.title("Archetype → Filter safety (history)")

st.caption(
    "This tool replays history to see, for each **filter**, which **seed archetypes** "
    "it tended to **pass** (safe) or **fail** (unsafe). Use it to decide when your large filters are safest."
)

# Controls
with st.sidebar:
    st.header("Run analysis")
    winners_path = st.text_input("Winners CSV", WINNERS_CSV)
    filters_path = st.text_input("Filters CSV", FILTERS_CSV)
    min_support  = st.number_input("Min applicable days for a signal", 1, 9999, 12)
    min_lift     = st.number_input("Min lift vs baseline for TOP signals", 1.00, 10.0, 1.10, step=0.01)
    max_lag      = st.number_input("Max lift vs baseline for DANGER signals", 0.10, 1.00, 0.90, step=0.01)
    run_now = st.button("Run / Rebuild archetype CSVs", use_container_width=True)

# Paths to expected outputs
comp_p   = OUT_DIR / "archetype_filter_composite_stats.csv"
dims_p   = OUT_DIR / "archetype_filter_dimension_stats.csv"
top_p    = OUT_DIR / "archetype_filter_top_signals.csv"
danger_p = OUT_DIR / "archetype_filter_danger_signals.csv"

# Run analysis if requested
if run_now:
    if not ANALYZER_OK:
        st.error(
            "Couldn't import `archetype_safety`. Make sure `archetype_safety.py` "
            "is in the repo. Import error:\n\n" + ANALYZER_ERR
        )
    else:
        with st.spinner("Running archetype analysis over history…"):
            paths = analyze_archetype_safety(
                winners_csv=winners_path,
                filters_csv=filters_path,
                out_dir=OUT_DIR,
                min_support_for_signal=int(min_support),
                min_lift_for_top=float(min_lift),
                max_lift_for_danger=float(max_lag),
            )
        st.success("Analysis finished.")
        st.caption("Wrote: " + ", ".join(paths.values()))

# Load tables if present
df_top    = _load_if_exists(top_p)
df_danger = _load_if_exists(danger_p)
df_dims   = _load_if_exists(dims_p)
df_comp   = _load_if_exists(comp_p)

if not any([df_top is not None, df_danger is not None, df_dims is not None, df_comp is not None]):
    st.info("No archetype CSVs found yet. Click **Run / Rebuild archetype CSVs** in the sidebar.")
    st.stop()

# Filter box
fid_query = st.text_input("Filter ID filter (optional)", "")

def _apply_filter(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if fid_query.strip():
        return df[df["filter_id"].astype(str) == fid_query.strip()]
    return df

# Top signals
st.subheader("Top **positive** signals (dimension-level)")
st.caption("Traits where a filter’s pass rate is **≥ min lift** over its own baseline (with min support).")
if df_top is not None:
    v = _apply_filter(df_top)
    st.dataframe(v, use_container_width=True, hide_index=True)
    _download_btn(v, "archetype_filter_top_signals.filtered.csv" if fid_query else top_p.name,
                  "Download TOP (CSV)")
else:
    st.caption("— none yet —")

# Danger signals
st.subheader("**Danger** signals (dimension-level)")
st.caption("Traits where a filter’s pass rate is **≤ max lift** of its baseline (with min support).")
if df_danger is not None:
    v = _apply_filter(df_danger)
    st.dataframe(v, use_container_width=True, hide_index=True)
    _download_btn(v, "archetype_filter_danger_signals.filtered.csv" if fid_query else danger_p.name,
                  "Download DANGER (CSV)")
else:
    st.caption("— none yet —")

# All dimension breakdowns
st.subheader("All dimension breakdowns")
if df_dims is not None:
    v = _apply_filter(df_dims)
    st.dataframe(v, use_container_width=True, hide_index=True)
    _download_btn(v, "archetype_filter_dimension_stats.filtered.csv" if fid_query else dims_p.name,
                  "Download DIMENSIONS (CSV)")
else:
    st.caption("— none yet —")

# Composite archetypes
st.subheader("Composite archetypes (full seed signature)")
if df_comp is not None:
    v = _apply_filter(df_comp)
    st.dataframe(v, use_container_width=True, hide_index=True)
    _download_btn(v, "archetype_filter_composite_stats.filtered.csv" if fid_query else comp_p.name,
                  "Download COMPOSITE (CSV)")
else:
    st.caption("— none yet —")
