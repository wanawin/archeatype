# Filter Picker Pro — Safe Boot Harness
# Paste this entire file as your page to recover from the blank screen.
# After it renders, wire back your compile/eval logic where marked below.

import io
import traceback
from typing import Optional

import pandas as pd
import streamlit as st

# ---- Streamlit must be configured before any other Streamlit calls
st.set_page_config(page_title="Filter Picker Pro — Safe Boot", layout="wide")

# ------------- Safe, standalone panels ---------------------------------------
def render_compile_report(skipped_df: Optional[pd.DataFrame]) -> None:
    """
    Show 'Skipped rows (reason + expression)' with safe defaults and
    always-available downloads. Will not crash if skipped_df is None
    or missing columns.
    """
    if skipped_df is None or getattr(skipped_df, "empty", True):
        with st.expander("Skipped rows (reason + expression)", expanded=True):
            st.info("No skipped rows. All expressions compiled (or none attempted).")
        return

    # Make sure the dataframe has the columns we display; if not, fill safely.
    needed = ["id", "name", "reason", "expr"]
    for col in needed:
        if col not in skipped_df.columns:
            skipped_df[col] = ""

    with st.expander("Skipped rows (reason + expression)", expanded=True):
        st.caption(f"{len(skipped_df)} row(s) could not be evaluated.")
        st.dataframe(
            skipped_df[needed], use_container_width=True, height=420
        )

        # CSV download
        csv_buf = io.StringIO()
        skipped_df[needed].to_csv(csv_buf, index=False)
        st.download_button(
            label=f"Download skipped filters CSV ({len(skipped_df)} rows)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="skipped_filters.csv",
            mime="text/csv",
            key="dl_skipped_csv",
        )

        # ID list download
        id_series = skipped_df["id"].astype(str) if "id" in skipped_df else pd.Series([])
        id_list = ", ".join(id_series.tolist())
        st.text_area("Skipped IDs (comma-separated)", id_list, height=90)
        st.download_button(
            label="Download skipped IDs (.txt)",
            data=(id_list + "\n").encode("utf-8"),
            file_name="skipped_filter_ids.txt",
            mime="text/plain",
            key="dl_skipped_ids",
        )


def render_header() -> None:
    st.title("Filter Picker Pro — Safe Boot")
    st.caption(
        "This harness renders even if compile/eval fails. "
        "Once you see this page, reconnect your compile logic below."
    )
    st.divider()


# -------------------------- Main ---------------------------------------------
def main() -> None:
    render_header()

    # --- PLACEHOLDER: Your normal inputs/UI (seed, pool, file uploads, etc.)
    # Keep them inside try/except too if they eval expressions at render time.

    # Fake status row so we prove the layout renders:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parsed pool size", "—")
    col2.metric("Filters (CSV)", "—")
    col3.metric("History rows", "—")
    col4.metric("Seed", "—")

    st.info(
        "Now wire your compile/evaluate step below. "
        "Make sure **all** heavy work happens after the UI renders."
    )

    # --- PLACEHOLDER: run your compile/eval step and produce skipped_df
    # For the first render we show a small demo dataframe so the panel works.
    demo_df = pd.DataFrame(
        [
            {"id": "NO147F054", "name": "XXX", "reason": "example", "expr": "True"},
            {"id": "N0643F056", "name": "XXX", "reason": "example", "expr": "True"},
        ]
    )

    # If you already stash the real skipped_df in session_state, use it; otherwise demo
    skipped_df = st.session_state.get("skipped_df", demo_df)

    render_compile_report(skipped_df)

    st.success(
        "Page rendered. If you reached this point, the blank-screen bug was an "
        "import-time or early-execution exception. Reconnect your logic next."
    )

# ------------- Error surfacing wrapper ----------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("App crashed during startup — here is the full traceback:")
        st.code("".join(traceback.format_exc()))
