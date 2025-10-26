# filter_picker_pro.py
# Full self-contained Streamlit app (no patches needed)

import io
import math
import csv
import re
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Filter Picker — Tester Parity", layout="wide")

# -----------------------------
# Utilities: parsing & cleaning
# -----------------------------

def normalize_expr(s: str) -> str:
    """Make CSV expression text valid Python like your Tester app expects."""
    if s is None:
        return ""
    s = str(s)

    # Normalize curly quotes / em dashes etc.
    s = (s.replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("—", "-")
           .replace("\u00A0", " ")
           .strip())

    # Collapse doubled quotes repeatedly (handles ""foo"", """"bar"""", etc.)
    # We do it iteratively to squash nested duplications.
    prev = None
    while prev != s:
        prev = s
        s = s.replace('""', '"')
        s = re.sub(r"^\"(.*)\"$", r"\1", s.strip())  # remove outer quotes if still present

    return s.strip()


def parse_pool_text(txt: str) -> List[str]:
    """
    Accepts continuous digits, comma/space/line separated.
    Returns 5-char zero-padded strings unique and sorted.
    """
    if not txt:
        return []

    # If looks like continuous digits (only digits + maybe commas/spaces), try to split by non-digits
    # Support both "0123498765..." and "01234,98765 12345"
    # We'll collect any 5-length digit groups; if there are longer runs, cut into chunks of 5.
    tokens = re.findall(r"\d+", txt)
    combos = []
    for t in tokens:
        if len(t) == 5:
            combos.append(t)
        elif len(t) > 5:
            # chop into 5s without overlap
            for i in range(0, len(t) - 4, 5):
                combos.append(t[i:i+5])

    # If nothing matched length 5, we fall back and try splitting by commas/spaces/newlines
    if not combos:
        for chunk in re.split(r"[,\s]+", txt):
            chunk = chunk.strip()
            if chunk.isdigit():
                chunk = chunk.zfill(5)
                if len(chunk) == 5:
                    combos.append(chunk)

    # Deduplicate (preserve order) and return
    seen = set()
    out = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_filters_csv(file: io.BytesIO) -> pd.DataFrame:
    """Load tester-style filter CSV; drop Unnamed columns; normalize expressions."""
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # harmonize common headers
    cols = {c.lower(): c for c in df.columns}
    if "expression" not in cols:
        # try to find the expression column heuristically
        guess = [c for c in df.columns if "expr" in c.lower()]
        if guess:
            df["expression"] = df[guess[0]]
        else:
            st.error("CSV is missing an 'expression' column.")
            df["expression"] = ""
    else:
        df["expression"] = df[cols["expression"]]

    # Normalize expressions
    df["expression"] = df["expression"].map(normalize_expr)

    # Provide a clean id/name/desc
    if "id" not in df.columns:
        # fallback: try 'name' unique or build index ids
        if "name" in df.columns:
            df["id"] = df["name"].fillna("").str.strip()
        else:
            df["id"] = [f"F{i+1:04d}" for i in range(len(df))]
    if "name" not in df.columns:
        df["name"] = df["id"]

    # Strip trailing/leading spaces
    df["id"] = df["id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    return df


# -----------------------------
# Safe evaluation environment
# -----------------------------

SAFE_FUNCS = {
    "set": set, "sum": sum, "len": len, "any": any, "all": all,
    "max": max, "min": min, "sorted": sorted,
}

def build_env(seed_str: str, combo_str: str) -> Dict:
    """Match Tester’s env: seed_digits as ints; combo_digits as strings; combo_sum as int."""
    combo_digits = list(combo_str)                   # ['8','8','0','0','1']
    combo_sum = sum(int(d) for d in combo_digits)    # int
    seed_digits = set(map(int, seed_str))            # {8,0,1}
    env = {
        "combo_digits": combo_digits,
        "combo_sum": combo_sum,
        "seed_digits": seed_digits,
    }
    env.update(SAFE_FUNCS)
    return env

def eval_rule(expr: str, env: Dict) -> bool:
    """
    Evaluate the CSV expression. Return True if the rule triggers (i.e., eliminate).
    'True' / empty means always-on eliminator.
    """
    if not expr:
        return False
    s = expr.strip()
    if s.lower() == "true":
        return True
    try:
        return bool(eval(s, {"__builtins__": {}}, env))
    except Exception:
        # mark as skipped by raising; caller handles
        raise


# -----------------------------
# Metrics & scoring
# -----------------------------

def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * math.sqrt((p_hat*(1 - p_hat) + z**2/(4*n))/n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def compute_filter_metrics(
    df_filters: pd.DataFrame,
    pool: List[str],
    seed: str,
    history_rows: List[str],
    history_weighting: str = "none"
) -> Tuple[pd.DataFrame, Dict[str, set]]:
    """
    For each filter: estimate keep% on winners and elim% on pool.
    - Keep%: fraction of history winners NOT eliminated when this rule is applied.
    - Elim%: fraction of the current pool eliminated (by sampling or full scan).
    Returns a dataframe + dict of {filter_id -> set(eliminated_indices)} for redundancy calc.
    """
    n_pool = len(pool)
    # precompute envs for pool (speed)
    pool_envs = [build_env(seed, c) for c in pool]

    elim_sets: Dict[str, set] = {}
    rows = []

    # history weighting (simple options)
    weights = np.ones(len(history_rows), dtype=float)
    if history_weighting == "time_decay":
        # newest gets weight 1.0, oldest ~ 0.5
        if len(weights) > 1:
            w = np.linspace(1.0, 0.5, num=len(weights))
            weights = w
    # Build envs for history winners
    hist_envs = [build_env(seed, w) for w in history_rows]

    for _, r in df_filters.iterrows():
        fid = r["id"]
        fname = r["name"]
        expr = r["expression"]
        # Evaluate elim set on pool
        eliminated_idx = set()
        skipped = False
        for i, env in enumerate(pool_envs):
            try:
                if eval_rule(expr, env):
                    eliminated_idx.add(i)
            except Exception:
                skipped = True
                break
        if skipped:
            keep_pct = np.nan
            elim_pct = np.nan
            wpp = np.nan
        else:
            # elim% on pool
            elim_pct = len(eliminated_idx) / max(1, n_pool)

            # Keep% on history winners
            keeps = []
            for env, w in zip(hist_envs, weights):
                try:
                    fired = eval_rule(expr, env)
                except Exception:
                    fired = False  # count as kept if rule can't eval on that winner
                keeps.append((0.0 if fired else 1.0, w))
            if keeps:
                # weighted keep%
                num = sum(k*w for k, w in keeps)
                den = sum(w for _, w in keeps)
                keep_pct = num / den if den > 0 else 0.0
            else:
                keep_pct = np.nan
            # a simple Win-Preserving Power metric
            alpha = st.session_state.get("alpha_wpp", 1.0)
            wpp = (keep_pct if not np.isnan(keep_pct) else 0.0) * (elim_pct ** alpha)

        elim_sets[fid] = eliminated_idx
        lb, ub = wilson_ci(keep_pct if not np.isnan(keep_pct) else 0.0, len(history_rows))
        rows.append({
            "id": fid,
            "name": fname,
            "keep%": keep_pct,
            "keep%_lb": lb,
            "keep%_ub": ub,
            "elim%": elim_pct,
            "WPP": wpp,
            "expression": expr
        })

    out = pd.DataFrame(rows)
    return out, elim_sets


def greedy_bundle(
    metrics: pd.DataFrame,
    elim_sets: Dict[str, set],
    n_pool: int,
    min_survival: float,
    target_survivors: int,
    redundancy_penalty: float
) -> Tuple[List[str], float, int, List[Tuple[str, float, int]]]:
    """
    Build a bundle of filters greedily:
    - Always maintain projected winner survival >= min_survival
    - Favor higher WPP and low redundancy with chosen
    - Stop when projected survivors <= target_survivors or nothing improves
    """
    chosen: List[str] = []
    chosen_elims: set = set()
    steps: List[Tuple[str, float, int]] = []

    # work DF with safe fills
    m = metrics.copy()
    m["keep%"] = m["keep%"].fillna(1.0)
    m["elim%"] = m["elim%"].fillna(0.0)
    m["WPP"] = m["WPP"].fillna(0.0)

    survivors = n_pool
    winner_survival = 1.0

    # Try at most len(metrics) iterations
    for _ in range(len(m)):
        best = None
        best_score = -1e9

        for _, r in m.iterrows():
            fid = r["id"]
            if fid in chosen:
                continue

            # candidate new survivors if we add this filter
            new_elims = elim_sets.get(fid, set())
            new_total_elims = len(chosen_elims | new_elims)
            new_survivors = max(0, n_pool - new_total_elims)

            # candidate survival (approx multiply keeps)
            keep = float(r["keep%"])
            cand_survival = winner_survival * keep

            if cand_survival < min_survival:
                continue

            # redundancy: compute jaccard with already chosen union
            red = 0.0
            if chosen_elims:
                red = jaccard(new_elims, chosen_elims)
            # objective: favor reducing survivors and high WPP; penalize redundancy
            # You can tune weights here
            score = ( (survivors - new_survivors)   # how many more we remove
                      + 1000 * float(r["WPP"])      # reward for win-preserving thinning
                      - 200 * redundancy_penalty * red )

            if score > best_score:
                best_score = score
                best = (fid, new_survivors, cand_survival)

        if not best:
            break

        fid, new_survivors, cand_survival = best
        chosen.append(fid)
        chosen_elims |= elim_sets.get(fid, set())
        survivors = new_survivors
        winner_survival = cand_survival
        steps.append((fid, winner_survival, survivors))

        if survivors <= target_survivors:
            break

    return chosen, winner_survival, survivors, steps


# -----------------------------
# UI
# -----------------------------

st.title("Filter Picker — Tester Parity")

with st.expander("Paste current pool — supports continuous digits", expanded=True):
    pool_text = st.text_area(
        "Pool (you can paste continuous digits, or comma/space/newline-separated)",
        height=150,
        placeholder="01234 98765 12345, 54321 … or continuous 0123498765…"
    )
    pool = parse_pool_text(pool_text)
    st.caption(f"Parsed pool size: **{len(pool)}**")

col_top = st.columns(3)
with col_top[0]:
    seed = st.text_input("Seed (last 5)", value="88001", max_chars=5).strip()
    seed = re.sub(r"\D", "", seed).zfill(5)[:5]
with col_top[1]:
    st.session_state["alpha_wpp"] = st.number_input("WPP α (elim exponent)", 0.1, 3.0, 1.0, 0.1)
with col_top[2]:
    chronology = st.radio(
        "History chronology",
        ["Newest→Oldest", "Oldest→Newest"],
        index=0,
        horizontal=True
    )

up1, up2 = st.columns(2)
with up1:
    mf = st.file_uploader("Upload master filter CSV (tester-style)", type=["csv"])
with up2:
    hist_file = st.file_uploader("Upload history winners (csv/txt)", type=["csv", "txt"])

# Gather history
history_rows: List[str] = []
if hist_file:
    raw = hist_file.read().decode("utf-8", errors="ignore")
    # accept either one per line or CSV in first column
    # pull all digit runs of length >=5 and chop into 5s
    tokens = re.findall(r"\d+", raw)
    for t in tokens:
        if len(t) == 5:
            history_rows.append(t)
        elif len(t) > 5:
            for i in range(0, len(t) - 4, 5):
                history_rows.append(t[i:i+5])
    if chronology == "Oldest→Newest":
        pass
    else:
        history_rows = list(reversed(history_rows))

st.caption(f"History rows loaded: **{len(history_rows)}**")

# Load filters CSV
filters_df: Optional[pd.DataFrame] = None
if mf:
    try:
        filters_df = load_filters_csv(io.BytesIO(mf.getvalue()))
        st.success(f"Loaded filters (CSV): {len(filters_df)} rows")
    except Exception as e:
        st.error(f"Failed to load filters CSV: {e}")

# Run button
run = st.button("Compute metrics & build suggestions", type="primary")

if run and filters_df is not None and pool and seed and history_rows:
    with st.spinner("Evaluating filters…"):
        metrics_df, elim_sets = compute_filter_metrics(
            filters_df, pool, seed, history_rows,
            history_weighting="time_decay"
        )

    # Skips report
    compiled = metrics_df["keep%"].notna().sum()
    skipped = len(metrics_df) - compiled
    st.success(f"Compiled {compiled} expressions; **{skipped}** skipped.")
    with st.expander("Compile report (skipped filters)"):
        skipped_df = metrics_df[metrics_df["keep%"].isna()][["id", "name", "expression"]]
        st.dataframe(skipped_df, use_container_width=True)

    st.subheader("Per-filter metrics")
    show_cols = ["id", "name", "keep%", "keep%_lb", "keep%_ub", "elim%", "WPP", "expression"]
    st.dataframe(metrics_df[show_cols].sort_values("WPP", ascending=False), use_container_width=True)

    st.subheader("Advanced: Redundancy & Greedy Bundle")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        min_survival = st.slider("Min winner survival (bundle)", 0.5, 1.0, 0.75, 0.01)
    with c2:
        target_survivors = st.number_input("Target survivors (bundle)", 1, len(pool), 50, 1)
    with c3:
        red_pen = st.slider("Redundancy penalty (0=off)", 0.0, 1.0, 0.2, 0.01)

    if st.button("Build greedy bundle", use_container_width=True):
        chosen, surv_prob, survivors, steps = greedy_bundle(
            metrics_df, elim_sets, len(pool), min_survival, target_survivors, red_pen
        )
        st.markdown("**Selected bundle (IDs)**:")
        st.write(", ".join(chosen) if chosen else "—")
        st.write(f"Projected winner survival: **{surv_prob:.2%}**")
        st.write(f"Projected survivors (pool): **{survivors:,}**")

        if steps:
            step_df = pd.DataFrame(steps, columns=["added", "survival", "survivors"])
            st.dataframe(step_df, use_container_width=True)
            # Download buttons
            ids_txt = ", ".join(chosen)
            st.download_button("Download bundle IDs (.txt)", ids_txt.encode(), file_name=f"bundle_ids_{survivors}.txt")
            # Survivors preview list (top 200 for speed)
            chosen_elims = set()
            for fid in chosen:
                chosen_elims |= elim_sets.get(fid, set())
            survivors_idx = [i for i in range(len(pool)) if i not in chosen_elims]
            survivors_list = [pool[i] for i in survivors_idx]
            preview = ", ".join(survivors_list[:200])
            st.text_area("Survivors (preview, up to 200)", value=preview, height=120)
            st.download_button("Download projected survivors (.txt)",
                               ("\n".join(survivors_list)).encode(),
                               file_name=f"survivors_{survivors}.txt")

else:
    st.info("Load master filter CSV, history, paste pool, and enter a seed, then click **Compute metrics & build suggestions**.")
