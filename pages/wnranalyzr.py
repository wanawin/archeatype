# wnranalyzr.py
# DC5 Hot/Cold/Due Analyzer — v0.6
# - Single Run button
# - Robust TXT parsing (00458 and 0-0-4-5-8 / 0–0–4–5–8)
# - Rolling hot/cold/due computed from prior draws only
# - Expanded Summary (always expanded) + distribution tables
# - CSV / TXT exports
# - No auto-refresh while downloading

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st


# ---------- parsing ----------
WIN_CONTIGUOUS = re.compile(r"(?<!\d)(\d{5})(?!\d)")
WIN_HYPHENATED = re.compile(r"(?<!\d)(\d)[\-\–](\d)[\-\–](\d)[\-\–](\d)[\-\–](\d)(?!\d)")

def _normalize_winner_token(tok: str) -> str:
    tok = tok.strip()
    return tok if len(tok) == 5 else tok.zfill(5)

def parse_winners_from_text(text: str) -> List[str]:
    winners: List[Tuple[int, str]] = []
    for m in WIN_HYPHENATED.finditer(text):
        winners.append((m.start(), _normalize_winner_token("".join(m.groups()))))
    for m in WIN_CONTIGUOUS.finditer(text):
        winners.append((m.start(), _normalize_winner_token(m.group(1))))
    winners.sort(key=lambda x: x[0])
    return [w for _, w in winners]

def parse_uploaded_file(file) -> List[str]:
    name = (file.name or "").lower()
    data = file.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="ignore")

    if name.endswith(".txt") or not name.endswith(".csv"):
        return parse_winners_from_text(data)

    # CSV path
    df = pd.read_csv(io.StringIO(data))
    maybe_cols = list(df.columns)
    preferred = ["winner", "winning", "result", "results", "combo", "number", "numbers"]
    for p in preferred:
        for c in list(maybe_cols):
            if p in str(c).lower():
                maybe_cols.insert(0, maybe_cols.pop(maybe_cols.index(c)))
                break

    for c in maybe_cols:
        vals = df[c].astype(str).fillna("").tolist()
        tmp, ok = [], 0
        for v in vals:
            v = v.strip()
            m1 = WIN_HYPHENATED.fullmatch(v)
            m2 = WIN_CONTIGUOUS.fullmatch(v)
            if m1:
                tmp.append("".join(m1.groups())); ok += 1
            elif m2:
                tmp.append(_normalize_winner_token(m2.group(1))); ok += 1
            else:
                found = parse_winners_from_text(v)
                if len(found) == 1:
                    tmp.append(found[0]); ok += 1
                elif len(found) > 1:
                    ok = -999; break
        if ok > 0:
            return [_normalize_winner_token(t) for t in tmp if t]

    return parse_winners_from_text(data)

def detect_order(_w: List[str]) -> str:
    # Simple default; user can override
    return "Newest first"


# ---------- labeling ----------
@dataclass
class Params:
    window_for_hotcold: int
    pct_hot: int
    pct_cold: int
    due_threshold: int

def freq_counts(history: List[str]) -> Dict[str, int]:
    counts = {str(d): 0 for d in range(10)}
    for w in history:
        for ch in w:
            counts[ch] += 1
    return counts

def label_hot_cold(counts: Dict[str, int], pct_hot: int, pct_cold: int) -> Tuple[set, set]:
    full = {str(d): counts.get(str(d), 0) for d in range(10)}
    ser = pd.Series(full).sort_values(ascending=False)
    n = len(ser)
    n_hot = int(np.floor(n * (pct_hot / 100.0)))
    n_cold = int(np.floor(n * (pct_cold / 100.0)))
    hot = set(ser.index[:n_hot]) if n_hot > 0 else set()
    cold = set(ser.sort_values(ascending=True).index[:n_cold]) if n_cold > 0 else set()
    cold -= hot
    return hot, cold

def due_set(history: List[str], due_threshold: int) -> set:
    if due_threshold <= 0:
        return set()
    recent = history[-due_threshold:] if due_threshold <= len(history) else history[:]
    seen = set(ch for w in recent for ch in w)
    return set(str(d) for d in range(10)) - seen


# ---------- analysis ----------
def analyze(old_to_new: List[str], params: Params):
    if not old_to_new:
        return pd.DataFrame(), {}, [], []

    # quick hottest→coldest using latest window/all
    refer = old_to_new if params.window_for_hotcold == 0 else old_to_new[-params.window_for_hotcold:]
    ordered_digits = list(pd.Series(freq_counts(refer)).sort_values(ascending=False).index)

    rows = []
    for i, winner in enumerate(old_to_new):
        prior = old_to_new[:i]
        if not prior:
            rows.append({
                "idx": i, "winner": winner, "sum": sum(int(c) for c in winner),
                "hot_hits": 0, "cold_hits": 0, "due_hits": 0,
                "neutral_hits": 5, "hot_and_due_hits": 0,
                "hot_digits": "", "cold_digits": "", "due_digits": "",
            })
            continue

        window = prior[-params.window_for_hotcold:] if params.window_for_hotcold > 0 else prior
        counts = freq_counts(window)
        hot, cold = label_hot_cold(counts, params.pct_hot, params.pct_cold)
        due = due_set(prior, params.due_threshold)

        digits = list(winner)
        hot_hits  = sum(d in hot  for d in digits)
        cold_hits = sum(d in cold for d in digits)
        due_hits  = sum(d in due  for d in digits)
        labeled = hot | cold | due
        neutral_hits = sum(d not in labeled for d in digits)
        hot_and_due_hits = sum(d in hot and d in due for d in digits)

        rows.append({
            "idx": i, "winner": winner, "sum": sum(int(c) for c in winner),
            "hot_hits": hot_hits, "cold_hits": cold_hits, "due_hits": due_hits,
            "neutral_hits": neutral_hits, "hot_and_due_hits": hot_and_due_hits,
            "hot_digits": ",".join(sorted(hot)),
            "cold_digits": ",".join(sorted(cold)),
            "due_digits": ",".join(sorted(due)),
        })

    df = pd.DataFrame(rows)

    # rich summary
    def dist(col):
        s = df[col].value_counts().sort_index()
        pct = (s / len(df) * 100).round(2)
        out = pd.DataFrame({"count": s, "pct": pct})
        for k in range(0, 6):  # ensure 0..5 present
            if k not in out.index:
                out.loc[k] = {"count": 0, "pct": 0.0}
        return out.sort_index()

    summary = {
        "n_winners": int(len(df)),
        "window_for_hotcold": int(params.window_for_hotcold),
        "pct_hot": int(params.pct_hot),
        "pct_cold": int(params.pct_cold),
        "due_threshold": int(params.due_threshold),
        "means": {
            "hot_hits": float(df["hot_hits"].mean()),
            "cold_hits": float(df["cold_hits"].mean()),
            "due_hits": float(df["due_hits"].mean()),
            "neutral_hits": float(df["neutral_hits"].mean()),
            "hot_and_due_hits": float(df["hot_and_due_hits"].mean()),
            "sum": float(df["sum"].mean()),
        },
    }

    dist_tables = {
        "hot_hits_dist":  dist("hot_hits"),
        "cold_hits_dist": dist("cold_hits"),
        "due_hits_dist":  dist("due_hits"),
        "neutral_hits_dist": dist("neutral_hits"),
        "hot_and_due_hits_dist": dist("hot_and_due_hits"),
    }

    return df, summary, dist_tables, ordered_digits


# ---------- UI ----------
st.set_page_config(page_title="DC5 Hot/Cold/Due Analyzer", layout="wide")
st.title("DC5 Hot/Cold/Due Analyzer — v0.6")

with st.sidebar:
    st.header("1) Input")
    uploaded = st.file_uploader("Upload winners (.csv or .txt)", type=["csv", "txt"])

    st.header("2) File order")
    order_choice = st.radio(
        "Row order in file:",
        ["Auto (guess)", "Newest first", "Oldest first"],
        index=1,
        help="We compute labels for each winner using only prior draws."
    )

    st.header("3) Run controls")
    run_clicked = st.button("▶️ Run analysis")

    st.header("4) Hot/Cold/Due params")
    window = st.number_input("Window (trailing draws) for hot/cold ranking (0 = all)",
                             min_value=0, max_value=2000, value=10, step=1)
    pct_hot = st.slider("% of digits labeled Hot (by frequency)", 0, 50, 30)
    pct_cold = st.slider("% of digits labeled Cold (by frequency)", 0, 50, 30)
    due_threshold = st.number_input("Due threshold: 'not seen in last N draws'",
                                    min_value=0, max_value=200, value=2, step=1)

    st.header("5) Export control")
    export_prefix = st.text_input("Export file prefix", value="dc5_analysis")

params = Params(int(window), int(pct_hot), int(pct_cold), int(due_threshold))

info_box = st.empty()
quick_box = st.empty()
table_placeholder = st.empty()
sum_header = st.markdown("### Summary")
summary_json = st.empty()
dist_cols = st.columns(3)
dist_more_cols = st.columns(2)
export_cols = st.columns(2)

if uploaded is None:
    st.info("Upload a winners file and click **Run analysis**.")
else:
    try:
        uploaded.seek(0)
        winners_raw = parse_uploaded_file(uploaded)
        info_box.info(f"Parsed: **{len(winners_raw)}** winners found "
                      f"(accepts 5-digit like `00458` and hyphenated `0-0-4-5-8`).")

        # order handling
        guessed = detect_order(winners_raw)
        chosen = order_choice if order_choice != "Auto (guess)" else guessed
        winners = winners_raw[:] if chosen == "Newest first" else winners_raw[::-1]
        old_to_new = winners[::-1] if chosen == "Newest first" else winners[:]

        if run_clicked:
            df, summary, dist_tables, ordered_digits = analyze(old_to_new, params)

            quick_box.markdown(f"### Hottest → coldest (quick view)\n\n`{', '.join(ordered_digits)}`")

            if df.empty:
                table_placeholder.warning("No winners parsed from the file.")
            else:
                table_placeholder.dataframe(df, use_container_width=True)

                # Expanded summary (no '{ ... }' collapse)
                summary_json.json(summary, expanded=True)

                # Distributions
                with dist_cols[0]:
                    st.markdown("**Hot hits**"); st.dataframe(dist_tables["hot_hits_dist"])
                with dist_cols[1]:
                    st.markdown("**Cold hits**"); st.dataframe(dist_tables["cold_hits_dist"])
                with dist_cols[2]:
                    st.markdown("**Due hits**"); st.dataframe(dist_tables["due_hits_dist"])
                with dist_more_cols[0]:
                    st.markdown("**Neutral hits**"); st.dataframe(dist_tables["neutral_hits_dist"])
                with dist_more_cols[1]:
                    st.markdown("**Hot ∩ Due hits**"); st.dataframe(dist_tables["hot_and_due_hits_dist"])

                # Exports (no auto-refresh)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                txt_buf = io.StringIO()
                txt_buf.write("DC5 Hot/Cold/Due Analyzer Results\n")
                txt_buf.write(f"Total winners: {summary['n_winners']}\n")
                txt_buf.write(f"Window for hot/cold: {summary['window_for_hotcold']}\n")
                txt_buf.write(f"% Hot: {summary['pct_hot']} | % Cold: {summary['pct_cold']}\n")
                txt_buf.write(f"Due threshold: last {summary['due_threshold']} draws\n\n")
                txt_buf.write("Hottest→Coldest: " + ", ".join(ordered_digits) + "\n\n")
                for k, v in summary["means"].items():
                    txt_buf.write(f"mean_{k}: {v}\n")

                with export_cols[0]:
                    st.download_button("Download per-winner CSV",
                        data=csv_bytes,
                        file_name=f"{export_prefix}_per_winner.csv",
                        mime="text/csv",
                        use_container_width=True)
                with export_cols[1]:
                    st.download_button("Download TXT summary",
                        data=txt_buf.getvalue().encode("utf-8"),
                        file_name=f"{export_prefix}_summary.txt",
                        mime="text/plain",
                        use_container_width=True)

    except Exception as e:
        st.error(f"Failed to read file: {e}")
