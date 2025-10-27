# ---- at very top (after imports) ----
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Filter Picker Pro", layout="wide")

def render_compile_report(skipped_df: pd.DataFrame | None):
    import io
    if skipped_df is None or getattr(skipped_df, "empty", True):
        st.info("No skipped rows. All expressions compiled.")
        return

    with st.expander("Skipped rows (reason + expression)", expanded=True):
        st.caption(f"{len(skipped_df)} row(s) could not be evaluated.")
        st.dataframe(skipped_df, use_container_width=True, height=420)

        csv_buf = io.StringIO()
        skipped_df.to_csv(csv_buf, index=False)
        st.download_button(
            label=f"Download skipped filters ({len(skipped_df)}).csv",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="skipped_filters.csv",
            mime="text/csv",
            key="dl_skipped_filters_csv",
        )

        id_list = ", ".join(map(str, skipped_df.get("id", [])))
        st.text_area("Skipped IDs (comma-separated)", id_list, height=90)
        st.download_button(
            label="Copy/download skipped IDs (.txt)",
            data=(id_list + "\n").encode("utf-8"),
            file_name="skipped_filter_ids.txt",
            mime="text/plain",
            key="dl_skipped_ids_txt",
        )
# ---- end panel ----
