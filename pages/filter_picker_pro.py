# pages/filter_picker_pro.py
# Streamlit ≥ 1.28

import io
import re
import time
import math
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Filter Picker (Hybrid I/O) — Advanced", layout="wide")

# =============================
# Session init
# =============================
def init_state():
    ss = st.session_state
    ss.setdefault("filters_df", None)            # master filters [id, name, expression]
    ss.setdefault("history_winners", [])         # list[str] 5-digit winners
    ss.setdefault("pool_combos", [])             # list[str] current pool 5-digit combos
    ss.setdefault("seed_str", "")
    ss.setdefault("chronology", "Newest → Oldest")
    ss.setdefault("alpha_wpp", 1.0)              # WPP exponent
    ss.setdefault("gamma_decay", 0.0)            # time-decay weight for winners
    ss.setdefault("result_df", None)             # per-filter scored DF
    ss.setdefault("fired_on_winners", dict())    # id -> set(indexes eliminated)
    ss.setdefault("fired_on_pool", dict())       # id -> set(indexes eliminated)
    ss.setdefault("last_compute_ts", None)
    ss.setdefault("bundle_result", None)         # dict with bundle info
    ss.setdefault("applicable_ids_raw", "")

init_state()

# =============================
# Parsing helpers
# =============================
def parse_pool_text(text: str) -> List[str]:
    if not text:
        return []
    if "," in text:
        parts = [re.sub(r"\D", "", p) for p in text.split(",")]
        return [p for p in parts if len(p) == 5 and p.isdigit()]
    return re.findall(r"(?<!\d)(\d{5})(?!\d)", text)

def parse_history_file(file, chronology_label: str) -> List[str]:
    raw = file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    winners = re.findall(r"(?<!\d)(\d{5})(?!\d)", text)
    return winners if chronology_label == "Newest → Oldest" else winners[::-1]

def load_filters_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    dcl = {c.lower(): c for c in df.columns}
    def has(*cols): return all(c in dcl for c in cols)

    if has("id", "name", "expression"):
        return df.rename(columns={dcl["id"]:"id", dcl["name"]:"name", dcl["expression"]:"expression"})[["id","name","expression"]]

    # Try to guess expression column
    expr_col = None
    for c in df.columns:
        sample = " ".join(df[c].astype(str).head(25).tolist())
        if any(tok in sample for tok in ["sum(", "any(", "all(", "combo_digits", "seed_digits", "ord("]):
            expr_col = c; break
    if expr_col is None:
        cols = list(df.columns)
        take = cols[:3] if len(cols)>=3 else cols + ["", ""]
        out = pd.DataFrame({
            "id": df[take[0]].astype(str) if take else "",
            "name": df[take[1]].astype(str) if len(take)>1 else "",
            "expression": df[take[2]].astype(str) if len(take)>2 else ""
        })
        return out

    rest = [c for c in df.columns if c != expr_col]
    id_col = rest[0] if rest else expr_col
    name_col = rest[1] if len(rest)>1 else id_col
    return pd.DataFrame({
        "id": df[id_col].astype(str),
        "name": df[name_col].astype(str),
        "expression": df[expr_col].astype(str)
    })

def parse_applicable_ids(text: str) -> Set[str]:
    if not text.strip(): return set()
    parts = re.split(r"[,\s]+", text.strip())
    return {p.strip() for p in parts if p.strip()}

# =============================
# Eval environment
# =============================
SAFE_FUNCS = {
    "sum": sum, "any": any, "all": all, "len": len, "set": set, "min": min, "max": max,
    "range": range, "ord": ord, "int": int, "float": float, "abs": abs
}

def apply_expr_to_combo(expr: str, combo: str, seed: str, extra_vars: Optional[Dict]=None) -> bool:
    """Return True if filter fires (eliminate combo)."""
    combo_digits = list(combo)
    seed_digits  = list(seed) if seed else []
    env = dict(SAFE_FUNCS)
    env.update({"combo_digits": combo_digits, "seed_digits": seed_digits, "primes":[2,3,5,7]})
    if extra_vars: env.update(extra_vars)
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        # Treat errors as non-firing to be conservative
        return False

# =============================
# Stats
# =============================
def wilson_ci(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k/n
    denom = 1 + (z*z)/n
    centre = p + (z*z)/(2*n)
    margin = z * math.sqrt((p*(1-p) + (z*z)/(4*n))/n)
    lo = (centre - margin)/denom
    hi = (centre + margin)/denom
    return max(0.0, lo), min(1.0, hi)

def time_decay_weights(n: int, gamma: float) -> np.ndarray:
    """Newer winners get higher weight if gamma>0 (index 0 is newest)."""
    if n == 0: return np.array([])
    idx = np.arange(n)  # 0..n-1
    w = np.exp(-gamma * idx)
    return w / w.sum()

# =============================
# Scoring + firing sets
# =============================
def compute_scores(filters_df: pd.DataFrame,
                   pool: List[str],
                   winners: List[str],
                   seed: str,
                   alpha: float,
                   gamma: float) -> Tuple[pd.DataFrame, Dict[str,Set[int]], Dict[str,Set[int]]]:
    """Return scored DF and firing sets (winners & pool)."""
    if filters_df is None or filters_df.empty:
        return pd.DataFrame(), {}, {}

    df = filters_df.fillna("").copy()
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["expression"] = df["expression"].astype(str)

    Nw, Np = len(winners), len(pool)
    w_w = time_decay_weights(Nw, gamma) if Nw>0 else np.array([])

    keep_rate, keep_lo, keep_hi = [], [], []
    elim_rate = []
    winners_fire_sets: Dict[str,Set[int]] = {}
    pool_fire_sets: Dict[str,Set[int]] = {}

    for _, row in df.iterrows():
        fid = row["id"]; expr = row["expression"]

        # Winners
        fired_ix = set()
        kept_weight = 0.0
        for i, w in enumerate(winners):
            fires = apply_expr_to_combo(expr, w, seed)
            if fires:
                fired_ix.add(i)
            else:
                kept_weight += (w_w[i] if Nw>0 else 1.0)
        winners_fire_sets[fid] = fired_ix
        if Nw > 0:
            k_rate = kept_weight if Nw>0 else (Nw-len(fired_ix))/Nw
            keep_rate.append(k_rate)
            lo, hi = wilson_ci(Nw-len(fired_ix), Nw)
            keep_lo.append(lo); keep_hi.append(hi)
        else:
            keep_rate.append(0.0); keep_lo.append(0.0); keep_hi.append(0.0)

        # Pool
        pool_fired = set()
        for j, c in enumerate(pool):
            if apply_expr_to_combo(expr, c, seed):
                pool_fired.add(j)
        pool_fire_sets[fid] = pool_fired
        elim_rate.append(len(pool_fired)/Np if Np>0 else 0.0)

    out = df.copy()
    out["keep_rate"] = keep_rate
    out["keep_lo"] = keep_lo
    out["keep_hi"] = keep_hi
    out["elim_rate"] = elim_rate
    out["WPP"] = out["keep_rate"] * (out["elim_rate"] ** float(alpha))
    out = out.sort_values(["WPP", "keep_rate", "elim_rate"], ascending=[False, False, False]).reset_index(drop=True)
    return out, winners_fire_sets, pool_fire_sets

def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

# =============================
# Greedy bundle builder
# =============================
def build_bundle(scored: pd.DataFrame,
                 winners_fire_sets: Dict[str,Set[int]],
                 pool_fire_sets: Dict[str,Set[int]],
                 N_winners: int,
                 w_weights: np.ndarray,
                 min_survival: float,
                 target_survivors: Optional[int],
                 redundancy_penalty: float = 0.5) -> Dict:
    if scored.empty:
        return {"selected": [], "survival": 0.0, "survivors": None, "history": []}

    selected: List[str] = []
    selected_union_w = set()
    selected_union_p = set()
    history_steps = []

    ids = scored["id"].tolist()

    def weighted_survival(union_elims_w: Set[int]) -> float:
        if N_winners == 0: return 0.0
        if w_weights.size == 0:
            return 1.0 - (len(union_elims_w) / N_winners)
        else:
            mask = np.ones(N_winners, dtype=bool)
            if union_elims_w:
                idx = np.fromiter(union_elims_w, dtype=int)
                mask[idx] = False
            kept_mass = w_weights[mask].sum()
            return kept_mass

    def projected_survivors(union_elims_p: Set[int], pool_size: int) -> int:
        return pool_size - len(union_elims_p)

    pool_size = max(0, max([max(s) if s else -1 for s in pool_fire_sets.values()] + [-1]) + 1)
    current_survival = weighted_survival(selected_union_w)

    while True:
        best_id = None
        best_score = -1e9
        best_survival = current_survival
        best_union_w = selected_union_w
        best_union_p = selected_union_p

        for fid in ids:
            if fid in selected: continue
            cand_union_w = selected_union_w | winners_fire_sets.get(fid, set())
            cand_union_p = selected_union_p | pool_fire_sets.get(fid, set())
            cand_survival = weighted_survival(cand_union_w)
            if cand_survival < min_survival:
                continue

            # Redundancy penalty on eliminated-winner overlap
            if selected:
                max_j = 0.0
                S = winners_fire_sets.get(fid, set())
                for sid in selected:
                    max_j = max(max_j, jaccard(S, winners_fire_sets.get(sid, set())))
            else:
                max_j = 0.0

            delta_thin = len(cand_union_p) - len(selected_union_p)
            score = delta_thin - redundancy_penalty * max_j * len(cand_union_p)

            if score > best_score:
                best_score = score
                best_id = fid
                best_survival = cand_survival
                best_union_w = cand_union_w
                best_union_p = cand_union_p

        if best_id is None:
            break

        selected.append(best_id)
        selected_union_w = best_union_w
        selected_union_p = best_union_p
        current_survival = best_survival

        cur_survivors = projected_survivors(selected_union_p, pool_size)
        history_steps.append({"added": best_id, "survival": current_survival, "survivors": cur_survivors})

        if target_survivors is not None and cur_survivors <= target_survivors:
            break

    return {
        "selected": selected,
        "survival": current_survival,
        "survivors": projected_survivors(selected_union_p, pool_size),
        "history": history_steps
    }

# =============================
# UI
# =============================
st.title("Filter Picker (Hybrid I/O) — Advanced Scoring")

with st.expander("How to use"):
    st.markdown("""
1) **Paste current pool** (comma-separated or continuous digits).  
2) **Upload master filter CSV** (must contain a Python `expression` column; `id` and `name` are detected).  
3) (Optional) **Paste applicable filter IDs** to limit to a subset (comma/space/newline separated).  
4) **Upload history** (CSV/TXT with any 5-digit winners); choose **chronology**.  
5) Set **Seed**, **WPP α**, and optional **time-decay** γ.  
6) Click **RUN** (or the Compute button) to score filters. Use the **Greedy bundle** builder to pick a set.
""")

# Inputs
top_l, top_r = st.columns([1.6, 1.0])

with top_l:
    pool_text = st.text_area(
        "Paste current pool (commas or continuous)",
        height=180,
        placeholder="88001,87055,04510,…  or a single blob like 8800187055…"
    )
    applicable_ids_raw = st.text_area(
        "Paste applicable filter IDs (optional — comma/space/newline-separated)",
        value=st.session_state.applicable_ids_raw,
        height=120,
        placeholder="LL002, LL002a, LL004d  (or one per line)"
    )
    st.session_state.applicable_ids_raw = applicable_ids_raw

with top_r:
    chronology = st.radio("History chronology", ["Newest → Oldest", "Oldest → Newest"])
    seed_str = st.text_input("Seed (last 5 digits)", value=st.session_state.seed_str, max_chars=5)
    alpha = st.slider("WPP α (thin weight)", 0.2, 2.0, float(st.session_state.alpha_wpp), 0.1)
    gamma = st.slider("Time-decay γ (history)", 0.0, 0.10, float(st.session_state.gamma_decay), 0.01,
                      help="Higher γ ⇒ newer winners get more weight")

st.session_state.chronology = chronology
st.session_state.seed_str = re.sub(r"\D","",seed_str)[:5]
st.session_state.alpha_wpp = alpha
st.session_state.gamma_decay = gamma

u1, u2, u3 = st.columns([1,1,1])
with u1:
    filters_file = st.file_uploader("Upload master filter CSV", type=["csv"])
with u2:
    history_file = st.file_uploader("Upload winners history (CSV or TXT)", type=["csv","txt"])
with u3:
    st.write("")  # spacer
    run_clicked = st.button("🚀 RUN", type="primary", use_container_width=True)

# Persist uploads
if pool_text:
    st.session_state.pool_combos = parse_pool_text(pool_text)
if filters_file is not None:
    st.session_state.filters_df = load_filters_csv(filters_file)
if history_file is not None:
    st.session_state.history_winners = parse_history_file(history_file, st.session_state.chronology)

# Metrics
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Parsed pool size", len(st.session_state.pool_combos))
m2.metric("Filters (master)", 0 if st.session_state.filters_df is None else len(st.session_state.filters_df))
m3.metric("History rows", len(st.session_state.history_winners))
m4.metric("Seed", st.session_state.seed_str if st.session_state.seed_str else "—")

# Compute trigger
compute_clicked = run_clicked or st.button("Compute per-filter metrics", use_container_width=True)

if compute_clicked:
    if not st.session_state.pool_combos:
        st.warning("Please paste the current pool.")
    elif st.session_state.filters_df is None or st.session_state.filters_df.empty:
        st.warning("Please upload a master filter CSV.")
    elif not st.session_state.history_winners:
        st.warning("Please upload a winners history file.")
    elif not st.session_state.seed_str or len(st.session_state.seed_str) != 5:
        st.warning("Please enter a 5-digit seed.")
    else:
        # Limit to applicable IDs if provided
        use_df = st.session_state.filters_df.copy()
        subset_ids = parse_applicable_ids(st.session_state.applicable_ids_raw)
        if subset_ids:
            use_df = use_df[use_df["id"].astype(str).isin(subset_ids)].reset_index(drop=True)
            if use_df.empty:
                st.error("None of the pasted IDs matched the master CSV.")
        with st.spinner("Evaluating filters…"):
            res, w_sets, p_sets = compute_scores(
                use_df,
                st.session_state.pool_combos,
                st.session_state.history_winners,
                st.session_state.seed_str,
                alpha=st.session_state.alpha_wpp,
                gamma=st.session_state.gamma_decay
            )
            st.session_state.result_df = res
            st.session_state.fired_on_winners = w_sets
            st.session_state.fired_on_pool = p_sets
            st.session_state.last_compute_ts = time.time()

# Results: table + Pareto
st.markdown("### Per-filter metrics")
if st.session_state.result_df is None or st.session_state.result_df.empty:
    st.info("No results yet. Provide inputs and click **RUN** (or **Compute**).")
else:
    df = st.session_state.result_df.copy()
    show_cols = ["id","name","keep_rate","keep_lo","keep_hi","elim_rate","WPP","expression"]
    for c in show_cols:
        if c not in df.columns: df[c] = ""
    df["keep_rate"] = (df["keep_rate"].astype(float)*100).round(2)
    df["keep_lo"]   = (df["keep_lo"].astype(float)*100).round(2)
    df["keep_hi"]   = (df["keep_hi"].astype(float)*100).round(2)
    df["elim_rate"] = (df["elim_rate"].astype(float)*100).round(2)
    df["WPP"]       = df["WPP"].astype(float).round(6)

    st.dataframe(df[show_cols], hide_index=True, use_container_width=True)

    chart_df = st.session_state.result_df.copy()
    chart_df["keep%"] = chart_df["keep_rate"]*100
    chart_df["elim%"] = chart_df["elim_rate"]*100
    base = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X("elim%:Q", title="Elimination Rate on Pool (%)"),
        y=alt.Y("keep%:Q", title="Winner Survival (Keep %)"),
        tooltip=["id","name","keep%","elim%","WPP"]
    ).interactive()
    st.altair_chart(base.properties(height=360), use_container_width=True)

    st.markdown("#### Copy / Download (per-filter table)")
    ids_only = "\n".join(chart_df.sort_values("WPP", ascending=False)["id"].astype(str).tolist())
    st.code(ids_only, language="text")
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button("Download IDs (.txt)", ids_only.encode("utf-8"),
                           file_name="recommended_filter_ids.txt", use_container_width=True)
    with cdl2:
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download table (.csv)", buf.getvalue().encode("utf-8"),
                           file_name="filters_scored.csv", use_container_width=True)

# Advanced bundle
st.markdown("---")
st.markdown("## Advanced: Redundancy & Greedy Bundle")

if st.session_state.result_df is None or st.session_state.result_df.empty:
    st.info("Compute per-filter metrics first.")
else:
    adv1, adv2, adv3 = st.columns([1.0,1.0,1.0])
    with adv1:
        min_survival = st.slider("Min winner survival (bundle)", 0.50, 0.99, 0.75, 0.01)
    with adv2:
        target_survivors = st.number_input("Target survivors (bundle)", min_value=0, value=1000, step=10)
    with adv3:
        red_pen = st.slider("Redundancy penalty (0=off)", 0.0, 1.0, 0.5, 0.05)

    bgo = st.button("Build greedy bundle", type="primary", use_container_width=True)

    if bgo:
        winners = st.session_state.history_winners
        Nw = len(winners)
        w_weights = time_decay_weights(Nw, st.session_state.gamma_decay) if Nw>0 else np.array([])
        bundle = build_bundle(
            st.session_state.result_df,
            st.session_state.fired_on_winners,
            st.session_state.fired_on_pool,
            Nw,
            w_weights,
            min_survival=float(min_survival),
            target_survivors=int(target_survivors),
            redundancy_penalty=float(red_pen)
        )
        st.session_state.bundle_result = bundle

    if st.session_state.bundle_result:
        b = st.session_state.bundle_result
        st.markdown("### Selected bundle")
        st.write(f"**Filters chosen ({len(b['selected'])})**: {', '.join(b['selected']) or '—'}")
        st.write(f"**Projected winner survival**: {b['survival']*100:.2f}%")
        st.write(f"**Projected survivors (pool)**: {b['survivors']:,}")

        if b["history"]:
            hdf = pd.DataFrame(b["history"])
            st.dataframe(hdf, hide_index=True, use_container_width=True)

        selected_ids = set(b["selected"])
        union_pool = set()
        for fid in selected_ids:
            union_pool |= st.session_state.fired_on_pool.get(fid, set())
        pool_size = len(st.session_state.pool_combos)
        survivors_idx = sorted(set(range(pool_size)) - union_pool)
        survivors_list = [st.session_state.pool_combos[i] for i in survivors_idx]
        surv_blob = ",".join(survivors_list)

        bdl1, bdl2 = st.columns(2)
        with bdl1:
            st.download_button("Download bundle IDs (.txt)",
                               "\n".join(b["selected"]).encode("utf-8"),
                               file_name="bundle_ids.txt",
                               use_container_width=True)
        with bdl2:
            st.download_button(f"Download projected survivors ({len(survivors_list)}).txt",
                               surv_blob.encode("utf-8"),
                               file_name="projected_survivors_bundle.txt",
                               use_container_width=True)

        st.markdown("#### Survivors (preview)")
        st.code(", ".join(survivors_list[:2000]), language="text")
