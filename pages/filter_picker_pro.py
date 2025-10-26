# filter_picker_pro.py
# Complete Streamlit app – universal tester CSV + history + pool (comma/continuous),
# with a safe eval shim so your tester expressions compile here too.

import io
import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st


######################################################################
# ------------------------- UI HELPERS ------------------------------ #
######################################################################

st.set_page_config(page_title="Filter Picker (Hybrid I/O)", layout="wide")

st.title("Filter Picker — Universal CSV + History + Pool")

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        **Inputs**
        1) **Pool (paste)**: comma-separated or continuous digits; 5-digit combos (e.g., `01234, 56789, ...`).
        2) **Master filter CSV**: Your *tester* CSV (any extra columns OK). Must include columns:
           - `id` (or `filter_id`)
           - `name` (optional but recommended)
           - `expression` (Python expression that is `True` when a combo should be **eliminated**)
           - `applicable_if` (optional guard; if omitted or empty -> treated as `True`)
        3) **History winners**: CSV or TXT, any order. Choose chronology below. The app builds
           rolling (prev2, prev1, seed) → **next** pairs and evaluates filters by asking:
           > *Would the filter have eliminated the **actual next winner**?*
        4) (Optional) **Applicable filter IDs**: paste a comma-separated subset; the app will restrict to those IDs.

        **Advanced**
        - Toggle history chronology (old→new or new→old).
        - Sliders control greedy bundle:
          * Min winner survival (bundle)
          * Target survivors (bundle)
          * Redundancy penalty (discourages redundant filters)
        """
    )

######################################################################
# ------------------------ INPUT PANELS ----------------------------- #
######################################################################

st.subheader("Inputs")

col_pool, col_csv, col_hist = st.columns([1.1, 1, 1])

with col_pool:
    pool_text = st.text_area(
        "Paste current pool (comma-separated or continuous), 5-digit combos",
        height=190,
        placeholder="01234, 56789, 00001, 12345 ...",
    )
    seed_override = st.text_input(
        "Seed (5 digits, optional – leave blank to use last history seed)",
        value="",
        max_chars=5,
        help="If blank, the last seed from the chosen chronology is used.",
    )
    ids_text = st.text_area(
        "Applicable filter IDs (optional, comma-separated)",
        placeholder="LL002f, LL003b, NO64F356 ...",
        height=88,
    )

with col_csv:
    filt_file = st.file_uploader(
        "Upload **master filter CSV** (tester CSV)", type=["csv"]
    )
    chronology = st.radio(
        "History chronology",
        options=["Oldest → Newest", "Newest → Oldest"],
        index=0,
        help="Choose how to interpret the order of winners in your file.",
    )

with col_hist:
    hist_file = st.file_uploader(
        "Upload **history winners** (CSV or TXT)", type=["csv", "txt"]
    )
    run_btn = st.button("Compute", type="primary", use_container_width=True)

######################################################################
# ------------------------ PARSERS & UTILS -------------------------- #
######################################################################

def parse_pool(text: str) -> List[str]:
    if not text:
        return []
    # Accept comma separated or any non-digit separators; find 5-digit chunks.
    # First try comma/space split:
    tokens = re.split(r"[,\s]+", text.strip())
    combos = []
    for t in tokens:
        if not t:
            continue
        if re.fullmatch(r"\d{5}", t):
            combos.append(t)
        else:
            # If token is continuous digits, break into 5s
            digits = re.sub(r"\D+", "", t)
            while len(digits) >= 5:
                chunk, digits = digits[:5], digits[5:]
                combos.append(chunk)
    # Dedup preserving order
    seen = set()
    out = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_filters_csv(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("filter_id")
    name_col = cols.get("name")
    expr_col = cols.get("expression")
    appl_col = cols.get("applicable_if")
    if id_col is None or expr_col is None:
        raise ValueError("CSV must have columns: id (or filter_id) and expression")
    out = pd.DataFrame({
        "id": df[id_col].astype(str).str.strip(),
        "name": df[name_col].astype(str).str.strip() if name_col else df[id_col].astype(str),
        "applicable_if": df[appl_col].fillna("True").astype(str) if appl_col else "True",
        "expression": df[expr_col].astype(str),
    })
    # Clean blanks to True (guards)
    out.loc[out["applicable_if"].str.strip().eq(""), "applicable_if"] = "True"
    return out


def load_history(file: io.BytesIO, chronology_label: str) -> List[str]:
    raw = file.read()
    # Try CSV then fallback to text digits
    try:
        df = pd.read_csv(io.BytesIO(raw))
        # Find a column with 5-digit combos
        col = None
        for c in df.columns:
            if df[c].astype(str).str.fullmatch(r"\d{5}").all():
                col = c
                break
        if col is None:
            # Try to extract from first column
            col = df.columns[0]
        seq = [str(x).zfill(5) for x in df[col].astype(str)]
    except Exception:
        # TXT: pull all 5-digit sequences
        txt = raw.decode("utf-8", errors="ignore")
        seq = re.findall(r"\b\d{5}\b", txt)

    if chronology_label.startswith("Newest"):
        seq = list(reversed(seq))
    return seq


def paste_applicable_ids(ids_text: str) -> Optional[set]:
    if not ids_text.strip():
        return None
    ids = [x.strip() for x in re.split(r"[,\s]+", ids_text.strip()) if x.strip()]
    return set([x.upper() for x in ids])


######################################################################
# --------------- EVAL SHIM: VARIABLES + BUILTINS ------------------- #
######################################################################

MIRROR_MAP = {'0':'5','1':'6','2':'7','3':'8','4':'9','5':'0','6':'1','7':'2','8':'3','9':'4'}

SAFE_BUILTINS = {
    "int": int,
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "set": set,
    "ord": ord,
    "list": list,
    "tuple": tuple,
    "min": min,
    "max": max,
    "abs": abs,
    "Counter": Counter,
    # math helpers (rarely used but safe)
    "math": math,
}

def make_env(seed: str, prev1: str, prev2: str, combo: str) -> Dict[str, object]:
    # All as strings
    seed_digits = list(seed)
    prev1_digits = list(prev1) if prev1 else []
    prev2_digits = list(prev2) if prev2 else []
    combo_digits = list(combo)

    last2_union = set(prev1_digits) | set(prev2_digits)
    last2_intersection = set(prev1_digits) & set(prev2_digits)
    new_seed_digits = set(seed_digits) - set(prev1_digits)
    seed_sum = sum(map(int, seed_digits))
    combo_sum = sum(map(int, combo_digits))

    # Useful aliases to match tester CSV
    env = dict(
        # current combo context
        combo_digits=combo_digits,
        combo_sum=combo_sum,

        # seed & previous winners
        seed_digits=seed_digits,
        prev1_digits=prev1_digits,
        prev2_digits=prev2_digits,

        # last2 helpers and synonyms
        last2_union=last2_union,
        last2_intersection=last2_intersection,
        last2=last2_union,                     # synonym used by some rows
        common_to_both=last2_intersection,     # synonym used by some rows

        # seed value helpers and synonyms
        seed_sum=seed_sum,
        seed_value=seed_sum,                   # synonym used by some rows

        # new seed vs MR
        new_seed_digits=new_seed_digits,

        # mirror mapping + helper
        mirror_map=MIRROR_MAP,
        mirror=lambda d: MIRROR_MAP[str(d)],

        # add builtins
        **SAFE_BUILTINS
    )
    return env


def eval_guard(expr: str, env: Dict[str, object]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        # If guard fails to evaluate, treat as not applicable (False)
        return False


def eval_elim(expr: str, env: Dict[str, object]) -> Optional[bool]:
    try:
        res = eval(expr, {"__builtins__": {}}, env)
        return bool(res)
    except Exception:
        return None


######################################################################
# --------------------- SCORING / METRICS --------------------------- #
######################################################################

@dataclass
class Pair:
    prev2: str
    prev1: str
    seed: str    # current seed (context)
    next: str    # the actual next winner


def build_history_pairs(history: List[str]) -> List[Pair]:
    # For indices i >= 2, seed = history[i], next = history[i+1] (if exists)
    pairs = []
    for i in range(2, len(history)-1):
        prev2, prev1, seed, nxt = history[i-2], history[i-1], history[i], history[i+1]
        pairs.append(Pair(prev2=prev2, prev1=prev1, seed=seed, next=nxt))
    return pairs


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z*math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def per_filter_metrics(filters_df: pd.DataFrame,
                       pool: List[str],
                       pairs: List[Pair],
                       seed_now: str,
                       applicable_ids: Optional[set]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      metrics_df: id, name, keep%, elim%, WPP, CI, expression (compiled)
      skipped_df: id, name, reason, expr
    """
    recs = []
    skipped = []
    applied = 0

    # Build mapping for quick access if subset of IDs given
    if applicable_ids is not None:
        sub = filters_df[filters_df["id"].str.upper().isin(applicable_ids)].copy()
    else:
        sub = filters_df.copy()

    sub = sub.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    for _, row in sub.iterrows():
        fid = str(row["id"]).strip()
        name = str(row["name"]).strip()
        guard = str(row["applicable_if"]).strip()
        expr = str(row["expression"]).strip()

        # Quick sanity – often empty guards/expr in some rows
        if not expr or expr.lower() == "nan":
            skipped.append({"id": fid, "name": name, "reason": "empty expression", "expr": expr})
            continue

        # Evaluate on current seed to confirm it's executable in this environment
        # and measure pool elimination rate + history keep rate.
        # 1) Check guard in *current* context (prev1/prev2 from last two winners)
        # For current seed context, use the last two winners in history:
        prev1 = pairs[-1].prev1 if pairs else ""
        prev2 = pairs[-1].prev2 if pairs else ""
        sample_env = make_env(seed_now, prev1, prev2, pool[0] if pool else seed_now)

        # If guard fails to evaluate, mark as skipped
        try:
            applicable_now = eval_guard(guard, sample_env)
        except Exception as e:
            skipped.append({"id": fid, "name": name,
                            "reason": f"guard eval error: {e.__class__.__name__}",
                            "expr": guard})
            continue

        if not applicable_now:
            # Not applicable for current seed context; we still keep it,
            # because it might be applicable for some pairs in scoring.
            pass

        # 2) Elim rate on current pool
        elim_cnt = 0
        eval_ok = True
        for c in pool:
            env = make_env(seed_now, prev1, prev2, c)
            # Must satisfy guard in *this* combo context too (common in your CSV)
            if not eval_guard(guard, env):
                continue
            res = eval_elim(expr, env)
            if res is None:
                eval_ok = False
                break
            if res:
                elim_cnt += 1
        if not eval_ok:
            skipped.append({"id": fid, "name": name, "reason": "runtime: expression did not evaluate",
                            "expr": expr})
            continue

        elim_rate = (elim_cnt / len(pool)) if pool else 0.0

        # 3) Keep rate on history pairs (winner survival)
        k = 0
        n = 0
        for p in pairs:
            env = make_env(p.seed, p.prev1, p.prev2, p.next)
            if not eval_guard(guard, env):
                continue
            res = eval_elim(expr, env)
            if res is None:
                # treat as not applicable for that pair
                continue
            n += 1
            if not res:  # not eliminated => winner survives
                k += 1
        keep = (k / n) if n > 0 else 0.0
        lb, ub = wilson_ci(k, n) if n > 0 else (0.0, 0.0)

        # 4) Combined score (WPP) – K * (elim_rate^alpha). Alpha=1 default; you can tune later.
        alpha = 1.0
        wpp = keep * (elim_rate ** alpha)

        recs.append({
            "id": fid, "name": name, "keep%": keep, "keep%_lb": lb, "keep%_ub": ub,
            "elim%": elim_rate, "WPP": wpp, "expression": expr, "guard": guard,
            "pairs_used": n
        })
        applied += 1

    metrics_df = pd.DataFrame(recs).sort_values(["WPP", "keep%"], ascending=[False, False]).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped)
    return metrics_df, skipped_df


######################################################################
# --------------------- GREEDY BUNDLE BUILDER ----------------------- #
######################################################################

def apply_filter_to_pool(expr: str, guard: str, seed: str, prev1: str, prev2: str, pool: Iterable[str]) -> List[str]:
    survivors = []
    for c in pool:
        env = make_env(seed, prev1, prev2, c)
        if not eval_guard(guard, env):
            survivors.append(c)
            continue
        res = eval_elim(expr, env)
        if not res:
            survivors.append(c)
    return survivors


def bundle_winner_survival(selected: List[Dict], pairs: List[Pair]) -> float:
    """
    Probability that the winner survives all filters (on history pairs).
    For each pair where the filter set is applicable, the next winner must not be eliminated by any filter.
    """
    k, n = 0, 0
    for p in pairs:
        # Evaluate all filters on the true next winner
        applicable_some = False
        eliminated = False
        for f in selected:
            env = make_env(p.seed, p.prev1, p.prev2, p.next)
            if not eval_guard(f["guard"], env):
                continue
            applicable_some = True
            res = eval_elim(f["expression"], env)
            if res:
                eliminated = True
                break
        if applicable_some:
            n += 1
            if not eliminated:
                k += 1
    return (k / n) if n > 0 else 0.0


def greedy_bundle(metrics: pd.DataFrame,
                  pool: List[str],
                  pairs: List[Pair],
                  seed_now: str,
                  min_win_survival: float,
                  target_survivors: int,
                  redundancy_penalty: float) -> Tuple[List[Dict], List[str], float]:
    """
    Basic forward greedy:
      - At each step, pick the filter that yields the largest drop in survivors
        while keeping bundle winner-survival ≥ min_win_survival (using history).
      - Redundancy penalty reduces value for filters that overlap (by identical surviving set),
        approximated here by Jaccard on **pool** survivors.
    """
    if metrics.empty or not pool:
        return [], pool, 0.0

    # Pre-compute survivors for each filter on the current pool
    prev1 = pairs[-1].prev1 if pairs else ""
    prev2 = pairs[-1].prev2 if pairs else ""

    pool_by_filter = {}
    for _, r in metrics.iterrows():
        survivors = apply_filter_to_pool(r["expression"], r["guard"], seed_now, prev1, prev2, pool)
        pool_by_filter[r["id"]] = set(survivors)

    selected = []
    survivors = set(pool)
    # Iteratively add filters
    while True:
        best = None
        best_score = -1e9
        best_survivors = None

        for _, r in metrics.iterrows():
            if any(f["id"] == r["id"] for f in selected):
                continue
            cand_survivors = pool_by_filter[r["id"]] & survivors
            # Score = drop - redundancy_penalty * overlap
            drop = len(survivors) - len(cand_survivors)
            # redundancy via Jaccard of survivors sets vs already selected
            red = 0.0
            if selected:
                overlaps = []
                for f in selected:
                    sA = pool_by_filter[f["id"]]
                    sB = pool_by_filter[r["id"]]
                    inter = len(sA & sB)
                    union = len(sA | sB)
                    overlaps.append(inter / union if union else 0.0)
                red = max(overlaps) if overlaps else 0.0
            score = drop - redundancy_penalty * red * len(survivors)

            if score > best_score:
                best = r
                best_score = score
                best_survivors = cand_survivors

        if best is None:
            break

        # Tentatively add and check bundle winner survival
        tentative = selected + [best.to_dict()]
        win_surv = bundle_winner_survival(tentative, pairs)
        if win_surv >= min_win_survival:
            selected = tentative
            survivors = best_survivors
        # Stop if we reached target
        if len(survivors) <= target_survivors:
            break

        # If adding this filter didn’t change survivors or worsens survival below min, stop if no more progress
        if best_score <= 0 and win_surv < min_win_survival:
            break

        # Safeguard from infinite loop
        if len(selected) > 400:
            break

    final_win = bundle_winner_survival(selected, pairs)
    return selected, sorted(list(survivors)), final_win


######################################################################
# --------------------------- MAIN RUN ------------------------------ #
######################################################################

if run_btn:
    # Parse pool
    pool = parse_pool(pool_text)
    if not pool:
        st.error("Please paste a valid 5-digit pool.")
        st.stop()

    # Load filters
    if filt_file is None:
        st.error("Please upload your **master filter CSV** (tester CSV).")
        st.stop()
    try:
        filters_df = load_filters_csv(filt_file)
    except Exception as e:
        st.error(f"Failed to read filter CSV: {e}")
        st.stop()

    # Load history
    if hist_file is None:
        st.error("Please upload a **history winners** file.")
        st.stop()
    try:
        history = load_history(hist_file, chronology)
    except Exception as e:
        st.error(f"Failed to read history: {e}")
        st.stop()

    if len(history) < 4:
        st.error("Need at least 4 winners in history to form pairs.")
        st.stop()

    # Build history pairs
    pairs = build_history_pairs(history)

    # Seed now: from override or last in chosen chronology
    seed_now = seed_override.strip()
    if not re.fullmatch(r"\d{5}", seed_now):
        seed_now = history[-2]  # last seed (the one before final next)
    st.success(f"Parsed pool size: **{len(pool)}**  |  History rows: **{len(history)}**  |  Seed: **{seed_now}**")

    # Restrict to IDs if provided
    ids_set = paste_applicable_ids(ids_text)

    st.write("---")
    st.subheader("Compile and score filters")

    metrics_df, skipped_df = per_filter_metrics(filters_df, pool, pairs, seed_now, ids_set)

    st.info(f"Compiled **{len(metrics_df)}** expressions; Skipped **{len(skipped_df)}**.")
    with st.expander("Skipped rows (reason + expression)", expanded=False):
        if skipped_df.empty:
            st.write("None 🎉")
        else:
            st.dataframe(skipped_df, use_container_width=True, height=280)

    if metrics_df.empty:
        st.warning("No executable filters after compilation.")
        st.stop()

    st.dataframe(
        metrics_df[["id", "name", "keep%", "keep%_lb", "keep%_ub", "elim%", "WPP", "pairs_used", "expression"]],
        use_container_width=True,
        height=480
    )

    # Downloads (don’t reset page)
    st.download_button(
        "Download per-filter metrics (.csv)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="per_filter_metrics.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download skipped rows (.csv)",
        data=skipped_df.to_csv(index=False).encode("utf-8"),
        file_name="skipped_filters.csv",
        mime="text/csv"
    )

    st.write("---")
    st.subheading = st.subheader("Advanced: Redundancy & Greedy Bundle")

    colA, colB, colC = st.columns(3)
    with colA:
        min_win_survival = st.slider("Min winner survival (bundle)", 0.50, 0.99, 0.75, 0.01)
    with colB:
        target_survivors = st.number_input("Target survivors (bundle)", min_value=1, max_value=len(pool),
                                           value=min(50, len(pool)), step=1)
    with colC:
        redundancy_penalty = st.slider("Redundancy penalty (0=off)", 0.0, 1.0, 0.20, 0.01)

    if st.button("Build greedy bundle", use_container_width=True):
        selected, survivors, bundle_win = greedy_bundle(
            metrics_df, pool, pairs, seed_now,
            min_win_survival=min_win_survival,
            target_survivors=int(target_survivors),
            redundancy_penalty=redundancy_penalty
        )

        st.markdown("### Selected bundle")
        if not selected:
            st.warning("No filters selected (constraints too strict or pool cannot be reduced by compiled filters).")
        else:
            chosen_ids = [f["id"] for f in selected]
            st.write(f"**Filters chosen ({len(chosen_ids)}):**", ", ".join(chosen_ids))
            st.write(f"**Projected winner survival (bundle):** {bundle_win:.2%}")
            st.write(f"**Projected survivors (pool):** {len(survivors)}")

            # Table of incremental effect
            rows = []
            prev1 = pairs[-1].prev1 if pairs else ""
            prev2 = pairs[-1].prev2 if pairs else ""
            running = set(pool)
            for f in selected:
                new_running = set(apply_filter_to_pool(f["expression"], f["guard"], seed_now, prev1, prev2, running))
                rows.append({"added": f["id"], "survival": bundle_win, "survivors": len(new_running)})
                running = new_running
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

            # Downloads
            st.download_button(
                "Download bundle IDs (.txt)",
                data=("\n".join(chosen_ids)).encode("utf-8"),
                file_name="bundle_ids.txt",
                mime="text/plain"
            )
            st.download_button(
                f"Download projected survivors ({len(survivors)}) (.txt)",
                data=(", ".join(survivors)).encode("utf-8"),
                file_name=f"survivors_{len(survivors)}.txt",
                mime="text/plain"
            )

            with st.expander("Survivors (preview)", expanded=False):
                st.write(", ".join(survivors))
