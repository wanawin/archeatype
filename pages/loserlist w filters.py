# --- keep everything above unchanged ---

# --------------------- UI ---------------------
st.title("Loser List (Least → Most Likely) — Tester-ready Export")

with st.sidebar:
    st.header("Input")
    pad4 = st.checkbox("Pad 4-digit entries", value=True)
    if st.button("Load example"):
        st.session_state["winners_text"] = (
            "74650,78845,88231,19424,37852,91664,33627,95465,53502,41621,05847,35515,81921"
        )

with st.form("winners_form", clear_on_submit=False):
    winners_text = st.text_area("13 winners (MR→Oldest)", key="winners_text", height=140)
    compute = st.form_submit_button("Compute")

def render_verification_panels(info):
    DIGITS = list("0123456789")
    LETTER_TO_NUM = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

    core_digits     = [str(LETTER_TO_NUM[L]) for L in info["core_letters"]]
    new_core_digits = [d for d in DIGITS if info["digit_current_letters"][d] not in info["core_letters"]]
    cooled_digits   = [d for d in DIGITS if info["rank_curr_map"][d] > info["rank_prev_map"][d]]
    loser_7_9       = info["ranking"][7:10]

    st.session_state["core_digits"]     = core_digits
    st.session_state["new_core_digits"] = new_core_digits
    st.session_state["cooled_digits"]   = cooled_digits
    st.session_state["loser_7_9"]       = loser_7_9

    st.subheader("Loser list (Least → Most Likely)")
    st.code(" ".join(info["ranking"]))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Current heatmap (digit → letter)**")
        st.dataframe(
            pd.DataFrame({"digit": DIGITS, "letter": [info["digit_current_letters"][d] for d in DIGITS]}),
            use_container_width=True, hide_index=True
        )
    with colB:
        st.markdown("**Previous heatmap (digit → letter)**")
        st.dataframe(
            pd.DataFrame({"digit": DIGITS, "letter": [info["digit_prev_letters"][d] for d in DIGITS]}),
            use_container_width=True, hide_index=True
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**prev_core_letters**"); st.code(", ".join(sorted(info["core_letters"])) or "∅")
        st.markdown("**loser_7_9 (digits)**");  st.code(", ".join(loser_7_9) or "∅")
    with c2:
        st.markdown("**cooled_digits (digits)**");   st.code(", ".join(sorted(cooled_digits)) or "∅")
        st.markdown("**new_core_digits (digits)**"); st.code(", ".join(sorted(new_core_digits)) or "∅")
    with c3:
        ring = _ring_digits(set(info["core_letters"]), info["digit_current_letters"])
        st.markdown("**ring_digits (digits)**");     st.code(", ".join(ring) or "∅")

if compute:
    try:
        winners = parse_winners_text(st.session_state["winners_text"], pad4=pad4)[:13]
        ranking, info = loser_list(winners)
        info["ranking"] = ranking
        st.session_state["info"] = info
        render_verification_panels(info)
    except Exception as e:
        st.error(str(e))

# ------ everything below will show after ANY successful Compute ------
if "info" in st.session_state:
    # Keep panels visible on reruns
    render_verification_panels(st.session_state["info"])

    st.markdown("### CSV Source")
    with st.form("csv_form", clear_on_submit=False):
        st.text_area("Paste MEGA CSV (3-col or 5-col)", key="mega_csv",
                     height=180, value=st.session_state.get("mega_csv",""))
        build = st.form_submit_button("Build Tester CSV")

    if build:
        info = st.session_state["info"]
        df3 = digits_only_df(
            in_csv_text=st.session_state.get("mega_csv",""),
            digit_current_letters=info["digit_current_letters"],
            digit_prev_letters=info["digit_prev_letters"],
            prev_core_letters=set(info["core_letters"]),
            cooled_digits=set(st.session_state["cooled_digits"]),
            new_core_digits=set(st.session_state["new_core_digits"]),
            loser_7_9=list(st.session_state["loser_7_9"]),
        )
        tester_df = to_tester_schema(df3)

        st.markdown("### Tester-ready CSV (copy/paste)")
        csv_bytes = tester_df.to_csv(index=False).encode("utf-8")
        st.code(csv_bytes.decode("utf-8"), language="csv")
        st.download_button("Download tester CSV",
                           data=csv_bytes,
                           file_name="filters_for_tester.csv",
                           mime="text/csv")
else:
    st.info("Enter winners and click **Compute** first.")
