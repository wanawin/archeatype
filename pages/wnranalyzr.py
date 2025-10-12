# --- Drop-in fixes for your Streamlit page (wnanalyzr.py) ---
# 1) Make composition_counts() robust to missing columns
# 2) Ensure per_winner always has a fixed schema (hot/cold/due/neutral),
#    even when a category is empty for the chosen window/thresholds.
# 3) (Optional) Guardrails so Hot/Cold sets never end up empty with short windows.

# ============================
# 1) Robust composition_counts
# ============================

def composition_counts(df, col):
    """Return distribution of how many of tag `col` each winner has (0..5).
    Never raises KeyError if the column is missing; treats it as all zeros.
    """
    import pandas as pd

    col = str(col).lower()
    if col in df.columns:
        s = df[col]
    else:
        # Column missing -> behave as all zeros
        s = pd.Series([0] * len(df), index=df.index)

    # Winners are 5-digit; possible counts 0..5
    dist = s.value_counts().reindex(range(6), fill_value=0).sort_index()
    out = dist.rename_axis("count").reset_index(name="winners")
    total = int(out["winners"].sum())
    out["pct"] = out["winners"] / total if total else 0.0
    return out


# =====================================================
# 2) Enforce fixed per_winner schema before summarizing
# =====================================================
# After you build `per_winner` (the dataframe with tag counts per winner),
# add this block ONCE before any calls to composition_counts(...):

# --- BEGIN: fixed schema block ---
FIXED_TAGS = ["hot", "cold", "due", "neutral"]
for _col in FIXED_TAGS:
    if _col not in per_winner.columns:
        per_winner[_col] = 0
# If you also compute overlaps like 'hot_due', 'hot_cold', etc., include them:
# OVERLAPS = ["hot_due", "hot_cold", "cold_due", "hot_cold_due"]
# for _col in OVERLAPS:
#     if _col not in per_winner.columns:
#         per_winner[_col] = 0
# --- END: fixed schema block ---


# ==================================================================
# 3) Optional: ensure Hot/Cold sets are never empty for short windows
# ==================================================================
# Where you derive hot_digits / cold_digits from digit frequencies,
# replace your head(...) logic with this pattern:

def pick_hot_cold(freq_series, hot_pct=30, cold_pct=30):
    """Return (hot_digits, cold_digits) lists from a frequency series of digits 0..9.
    Guarantees at least 1 hot and 1 cold digit.
    """
    import math

    # Percent of 10 digits -> count, but never less than 1
    k_hot = max(1, int(round(10 * hot_pct / 100.0)))
    k_cold = max(1, int(round(10 * cold_pct / 100.0)))

    hot_digits = (freq_series
                  .sort_values(ascending=False)
                  .head(k_hot)
                  .index.tolist())

    cold_digits = (freq_series
                   .sort_values(ascending=True)
                   .head(k_cold)
                   .index.tolist())

    return hot_digits, cold_digits

# Usage example (pseudocode context):
# freq = winners_digits.value_counts().reindex(range(10), fill_value=0)
# hot_digits, cold_digits = pick_hot_cold(freq, hot_pct=st.session_state.hot_pct, cold_pct=st.session_state.cold_pct)
# ... then label winners and build `per_winner` as before ...


# ==========================
# Notes for where to paste:
# ==========================
# • Put `composition_counts` at module scope (replacing the old one if present).
# • After constructing `per_winner`, paste the fixed schema block.
# • If you want the guardrails, swap in pick_hot_cold(...) where you pick hot/cold.
# • No other parts of your app need to change. The Pattern summaries section
#   can keep calling composition_counts(per_winner, "hot"), etc., safely.
