# filter_picker_pro.py
# Full self-contained Streamlit app (adds "Applicable IDs" paste box)
# Evaluates tester-style filters safely and builds a greedy bundle.

import io
import math
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Filter Picker — Tester Parity", layout="wide")

# -----------------------------
# Expression cleaning & parsing
# -----------------------------
def normalize_expr(s: str) -> str:
    """Normalize CSV expression text into valid Python like the Tester app expects."""
    if s is None:
        return ""
    s = str(s)
    # unify curly quotes/dashes & nbsp
    s = (s.replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("—", "-")
           .replace("\u00A0", " ")
           .strip())
    # collapse doubled quotes repeatedly (handles ""foo"" and """"bar"""")
    prev = None
    while prev != s:
        prev = s
        s = s.replace('""', '"')
        # remove ONE layer of outer quotes if present
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        s = s.strip()
    return s

def parse_pool_text(txt: str) -> List[str]:
    """
    Accepts continuous digits or comma/space/line separated. Returns unique 5-digit combos.
    """
    if not txt:
        return []
    tokens = re.findall(r"\d+", txt)
    combos = []
    for t in tokens:
        if len(t) == 5:
            combos.append(t)
        elif len(t) > 5:
            # chop into 5s without overlap
            for i in range(0, len(t) - 4, 5):
                combos.append(t[i:i+5])
    if not combos:
        for chunk in re.split(r"[,\s]+", txt):
            chunk = chunk.strip()
            if chunk.isdigit():
                chunk = chunk.zfill(5)
                if len(chunk) == 5:
                    combos.append(chunk)
    # de-dupe preserving order
    seen, out = set(), []
    for c in combos:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def load_filters_csv(file: io.BytesIO) -> pd.DataFrame:
    """Load tester-style filter CSV; drop Unnamed columns; normalize expressions."""
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # pick columns robustly
    cols = {c.lower().strip(): c for c in df.columns}
    if "expression" not in cols:
        guess = [c for c in df.columns if "expr" in c.lower()]
        if guess:
            df["expression"] = df[guess[0]]
        else:
            st.error("CSV is missing an 'expression' column."); df["expression"] = ""
    else:
        df["expression"] = df[cols["expression"]]
    # id / name
    if "id" in cols:
        df["id"] = df[cols["id"]].astype(str).str.strip()
    else:
        if "name" in cols:
            df["id"] = df[cols["name"]].astype(str).str.strip()
        else:
            df["id"] = [f"F{i+1:04d}" for i in range(len(df))]
    if "name" in cols:
        df["name"] = df[cols["name"]].astype(str).str.strip()
    else:
        df["name"] = df["id"]
    # normalize expression text
    df["expression"] = df["expression"].map(normalize_expr)
    return df[["id","name","expression"]]

def parse_applicable_ids(txt: str) -> List[str]:
    if not txt or not txt.strip():
        return []
    parts = re.split(r"[,\s]+", txt.strip())
    return [p.strip() for p in parts if p.strip()]

# -----------------------------
# Safe evaluation environment
# -----------------------------
SAFE_FUNCS = {
    "set": set, "sum": sum, "len": len, "any": any, "all": all,
    "max": max, "min": min, "sorted": sorted,
}
def build_env(seed_str: str, combo_str: str) -> Dict:
    """
    Match the Tester’s env:
      - seed_digits as a set of INTs (e.g., {8,0,1})
      - combo_digits as list of STRs (['8','8','0','0','1'])
      - combo_sum as INT
    """
    combo_digits = list(combo_str)
    combo_sum = sum(int(d) for d in combo_digits)
    seed_digits = set(map(int, seed_str)) if seed_str else set()
    env = {
        "combo_digits": combo_digits,
        "combo_sum": combo_sum,
        "seed_digits": seed_digits,
    }
    env.update(SAFE_FUNCS)
    return env

def eval_rule(expr: str, env: Dict) -> bool:
    """
    Evaluate the CSV expression. Return True if the rule fires (eliminate).
    Empty => False; 'True' => always-on eliminated.
    """
    if not expr:
        return False
    s = expr.strip()
    if s.lower() == "true":
        return True
    try:
        return bool(eval(s, {"__builtins__": {}}, env))
    except Exception:
        # mark as skipped by raising outward
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

def compute_filter_metrics(
    df_filters: pd.DataFrame,
    pool: List[str],
    seed: str,
    history_rows: List[str],
    weight_mode: str = "time_decay",
    alpha_wpp: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, set]]:
    """
    For each filter: compute keep% on winners and elim% on pool (hard apply).
    Returns metrics DF + map {filter_id -> set(eliminated_pool_indices)}.
    """
    n_pool = len(pool)
    pool_envs = [build_env(seed, c) for c in pool]
    # history envs & weights
    hist_envs = [build_env(seed, w) for w in history_rows]
    Nw = len(history_rows)
    if Nw > 1 and weight_mode == "time_decay":
        weights = np.linspace(1.0, 0.5, num=Nw)
    else:
        weights = np.ones(Nw, dtype=float)

    rows = []; elim_sets: Dict[str,set] = {}
    for _, r in df_filters.iterrows():
        fid, fname, expr = r["id"], r["name"], r["expression"]
        eliminated_idx = set()
        # apply to pool
        compile_failed = False
        for i, env in enumerate(pool_envs):
            try:
                if eval_rule(expr, env):
                    eliminated_idx.add(i)
            except Exception:
                compile_failed = True
                break
        if compile_failed:
            keep_pct = np.nan; elim_pct = np.nan; WPP = np.nan
            elim_sets[fid] = set()
            rows.append({
                "id": fid, "name": fname, "keep%": keep_pct,
                "keep%_lb": np.nan, "keep%_ub": np.nan,
                "elim%": elim_pct, "WPP": WPP, "expression": expr
            })
            continue

        elim_sets[fid] = eliminated_idx
        elim_pct = len(eliminated_idx)/max(1, n_pool)

        # keep% on history
        if Nw == 0:
            keep_pct = np.nan
            lb = ub = np.nan
        else:
            kept_mass = 0.0
            for env, w in zip(hist_envs, weights):
                try:
                    fired = eval_rule(expr, env)
                except Exception:
                    fired = False  # treat as kept
                if not fired:
                    kept_mass += w
            keep_pct = kept_mass / weights.sum()
            lb, ub = wilson_ci(keep_pct, Nw)

        WPP = (0.0 if np.isnan(keep_pct) else keep_pct) * (elim_pct ** float(alpha_wpp))
        rows.append({
            "id": fid, "name": fname, "keep%": keep_pct,
            "keep%_lb": lb if not np.isnan(keep_pct) else np.nan,
            "keep%_ub": ub if not np.isnan(keep_pct) else np.nan,
            "elim%": elim_pct, "WPP": WPP, "expression": expr
        })

    out = pd.DataFrame(rows).sort_values(["WPP","keep%","elim%"], ascending=[False,False,False])
    return out.reset_index(drop=True), elim_sets

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def greedy_bundle(
    metrics: pd.DataFrame,
    elim_sets: Dict[str, set],
    n_pool: int,
    min_survival: float,
    target_survivors: int,
    redundancy_penalty: float
) -> Tuple[List[str], float, int, List[Tuple[str, float, int]]]:
    """
    Greedy bundle builder: respects min_survival; stops at target_survivors or exhaustion.
    """
    chosen: List[str] = []
    union_elims: set = set()
    steps: List[Tuple[str, float, int]] = []
    survivors = n_pool
    win_survival = 1.0

    m = metrics.copy()
    m["keep%"] = m["keep%"].fillna(1.0)
    m["elim%"] = m["elim%"].fillna(0.0)
    m["WPP"]   = m["WPP"].fillna(0.0)

    for _ in range(len(m)):
        best = None; best_score = -1e18
        for _, r in m.iterrows():
            fid = r["id"]
            if fid in chosen: continue
            S = elim_sets.get(fid, set())
            new_union = union_elims | S
            new_survivors = max(0, n_pool - len(new_union))
            cand_keep = float(r["keep%"])
            cand_survival = win_survival * cand_keep
            if cand_survival < min_survival:
                continue
            # redundancy
            red = 0.0 if not union_elims else jaccard(S, union_elims)
            score = ( (survivors - new_survivors)         # more eliminated is better
                      + 1000.0 * float(r["WPP"])          # reward win-preserving thinning
                      - 200.0  * redundancy_penalty * red )
            if score > best_score:
                best_score = score
                best = (fid, new_union, new_survivors, cand_survival)
        if not best:
            break
        fid, union_elims, survivors, win_survival = best
        chosen.append(fid)
        steps.append((fid, win_survival, survivors))
        if survivors <= target_survivors:
            break

    return chosen, win_survival, survivors, steps

# -----------------------------
# UI
# -----------------------------
st.title("Filter Picker — Tester Parity (with Applicable IDs)")

with st.expander("Paste current pool — supports continuous digits", expanded=True):
    pool_text = st.text_area(
        "Pool (continuous digits or comma/space/newline-separated)",
        height=140,
        placeholder="01234 98765 12345, 54321 … or continuous 0123498765…"
    )
pool = parse_pool_text(pool_text)
st.caption(f"Parsed pool size: **{len(pool)}**")

c_top1, c_top2, c_top3 = st.columns(3)
with c_top1:
    seed = st.text_input("Seed (last 5 digits)", value="", max_chars=5).strip()
    seed = re.sub(r"\D", "", seed).zfill(5)[:5]
with c_top2:
    alpha_wpp = st.number_input("WPP α (thinning weight)", 0.1, 3.0, 1.5, 0.1)
with c_top3:
    chronology = st.radio("History chronology", ["Newest→Oldest", "Oldest→Newest"], horizontal=True, index=0)

up1, up2 = st.columns(2)
with up1:
    mf = st.file_uploader("Upload master filter CSV (tester-style)", type=["csv"])
with up2:
    hist_file = st.file_uploader("Upload history winners (csv/txt)", type=["csv","txt"])

# Applicable IDs paste box
st.markdown("### Applicable filter IDs (optional)")
app_ids_text = st.text_area(
    "Paste IDs to limit evaluation (comma / space / newline). Leave blank to use ALL from CSV.",
    height=110,
    placeholder="LL002f LL003b, LL018_unique_eq5, …"
)
app_ids = parse_applicable_ids(app_ids_text)

# Load history
history_rows: List[str] = []
if hist_file:
    raw = hist_file.read().decode("utf-8", errors="ignore")
    tokens = re.findall(r"\d+", raw)
    for t in tokens:
        if len(t) == 5:
            history_rows.append(t)
        elif len(t) > 5:
            for i in range(0, len(t) - 4, 5):
                history_rows.append(t[i:i+5])
    if chronology == "Newest→Oldest":
        pass
    else:
        history_rows = list(reversed(history_rows))
st.caption(f"History rows loaded: **{len(history_rows)}**")

# Load filters CSV
filters_df: Optional[pd.DataFrame] = None
if mf:
    try:
        filters_df = load_filters_csv(io.BytesIO(mf.getvalue()))
        st.success(f"Loaded filters: {len(filters_df)} total rows")
    except Exception as e:
        st.error(f"Failed to load filters CSV: {e}")

# RUN
run = st.button("🚀 Compute metrics", type="primary", use_container_width=True)

if run:
    if not pool:
        st.warning("Please paste the current pool."); st.stop()
    if not seed or len(seed) != 5:
        st.warning("Please enter a 5-digit seed."); st.stop()
    if filters_df is None or filters_df.empty:
        st.warning("Please upload the master filter CSV."); st.stop()
    if not history_rows:
        st.warning("Please upload winners history (csv/txt)."); st.stop()

    # Subset by Applicable IDs if provided
    use_df = filters_df.copy()
    matched = None
    if app_ids:
        # match case-insensitively on 'id'
        idset = {x.lower() for x in app_ids}
        use_df = use_df[use_df["id"].str.lower().isin(idset)].reset_index(drop=True)
        matched = len(use_df)
        if use_df.empty:
            st.error("None of the pasted IDs matched the CSV 'id' column."); st.stop()

    with st.spinner("Evaluating filters (safe)…"):
        metrics_df, elim_sets = compute_filter_metrics(
            use_df, pool, seed, history_rows,
            weight_mode="time_decay",
            alpha_wpp=alpha_wpp
        )

    compiled = metrics_df["keep%"].notna().sum()
    skipped = len(metrics_df) - compiled
    msg = f"Compiled **{compiled}**; skipped **{skipped}**."
    if matched is not None:
        msg += f" Matched Applicable IDs: **{matched}** / {len(app_ids)}."
    st.success(msg)

    # Show skipped list for quick debugging
    if skipped:
        with st.expander("Skipped filters (could not evaluate)"):
            st.dataframe(
                metrics_df[metrics_df["keep%"].isna()][["id","name","expression"]],
                use_container_width=True
            )

    st.subheader("Per-filter metrics")
    show_cols = ["id","name","keep%","keep%_lb","keep%_ub","elim%","WPP","expression"]
    st.dataframe(metrics_df[show_cols].sort_values("WPP", ascending=False), use_container_width=True)

    # ---------------- Greedy bundle ----------------
    st.markdown("---")
    st.subheader("Build greedy bundle")
    b1, b2, b3 = st.columns(3)
    with b1:
        min_survival = st.slider("Min winner survival", 0.50, 0.99, 0.72, 0.01)
    with b2:
        target_survivors = st.number_input("Target survivors", min_value=1, value=50, step=1)
    with b3:
        red_pen = st.slider("Redundancy penalty", 0.0, 1.0, 0.20, 0.05)

    if st.button("Build", type="primary"):
        chosen, surv_prob, survivors, steps = greedy_bundle(
            metrics_df, elim_sets, len(pool),
            min_survival=float(min_survival),
            target_survivors=int(target_survivors),
            redundancy_penalty=float(red_pen)
        )
        st.markdown("**Selected bundle (IDs):**")
        st.write(", ".join(chosen) if chosen else "—")
        st.write(f"Projected winner survival: **{surv_prob:.2%}**")
        st.write(f"Projected survivors: **{survivors:,}**")

        if steps:
            step_df = pd.DataFrame(steps, columns=["added","survival","survivors"])
            st.dataframe(step_df, use_container_width=True)

        # Survivors preview & download
        union = set()
        for fid in chosen:
            union |= elim_sets.get(fid, set())
        survivors_idx = [i for i in range(len(pool)) if i not in union]
        survivors_list = [pool[i] for i in survivors_idx]
        st.text_area("Survivors (preview, up to 2,000 chars)",
                     value=", ".join(survivors_list)[:2000],
                     height=120)
        st.download_button(
            "Download survivors (.txt)",
            ("\n".join(survivors_list)).encode("utf-8"),
            file_name=f"survivors_{survivors}.txt",
            use_container_width=True
        )

else:
    st.info("Paste the pool, upload master CSV + history, enter seed, optionally paste Applicable IDs, then click **Compute metrics**.")
