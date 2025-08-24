# --- Archetype safety (history) ---
st.markdown("---")
st.subheader("Archetype → Filter safety (history)")

comp_p   = Path("archetype_filter_composite_stats.csv")
dims_p   = Path("archetype_filter_dimension_stats.csv")
top_p    = Path("archetype_filter_top_signals.csv")
danger_p = Path("archetype_filter_danger_signals.csv")

if top_p.exists() or dims_p.exists() or comp_p.exists():
    fid_query = st.text_input("Filter ID (optional — leave blank to see all in each table)")

    if top_p.exists():
        st.markdown("**Top positive signals (dimension-level)**")
        try:
            df_top = pd.read_csv(top_p)
            if fid_query.strip():
                df_top = df_top[df_top["filter_id"].astype(str) == fid_query.strip()]
            st.dataframe(df_top.head(500), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load {top_p.name}: {e}")

    if danger_p.exists():
        st.markdown("**Danger signals (dimension-level)**")
        try:
            df_dn = pd.read_csv(danger_p)
            if fid_query.strip():
                df_dn = df_dn[df_dn["filter_id"].astype(str) == fid_query.strip()]
            st.dataframe(df_dn.head(500), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load {danger_p.name}: {e}")

    if dims_p.exists():
        st.markdown("**All dimension breakdowns**")
        try:
            df_dims = pd.read_csv(dims_p)
            if fid_query.strip():
                df_dims = df_dims[df_dims["filter_id"].astype(str) == fid_query.strip()]
            st.dataframe(df_dims.head(1000), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load {dims_p.name}: {e}")

    if comp_p.exists():
        st.markdown("**Composite archetypes (full signature)**")
        try:
            df_comp = pd.read_csv(comp_p)
            if fid_query.strip():
                df_comp = df_comp[df_comp["filter_id"].astype(str) == fid_query.strip()]
            st.dataframe(df_comp.head(1000), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load {comp_p.name}: {e}")
else:
    st.caption("Run archetype_safety.py once to generate CSVs, then reopen this page.")
