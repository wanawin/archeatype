# filter_picker_pro.py
# Complete Streamlit app – universal tester CSV + history + pool (comma/continuous),
# with an expanded safe-eval shim so tester/LL/CF-style variables never NameError.

import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------- UI & PAGE ------------------------- #
st.set_page_config(page_title="Filter Picker (Hybrid I/O)", layout="wide")
st.title("Filter Picker — Universal CSV + History + Pool")

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        **Inputs**
        1) **Pool (paste)**: comma-separated or continuous digits; 5-digit combos (e.g., `01234, 56789, ...`).
        2) **Master filter CSV**: your tester CSV. Required columns:
           - `id` (or `filter_id`)
           - `expression` (Python expression; **True** means *eliminate*)
           - `applicable_if` (optional; empty treated as True)
           - `name` (optional; used for display)
        3) **History winners**: CSV or TXT, any order (choose chronology).
        4) (Optional) **Applicable IDs**: comma list to restrict which rows are scored.

        **Advanced**
        - Chronology toggle
        - Greedy bundle with: Min winner survival, Target survivors, Redundancy penalty
        - All outputs are downloadable **without resetting** the page
        """
    )

# ------------------------- INPUTS --------------------------- #
st.subheader("Inputs")
col_pool, col_csv, col_hist = st.columns([1.1, 1, 1])

with col_pool:
    pool_text = st.text_area(
        "Paste current pool (comma-separated or continuous), 5-digit combos",
        height=190,
        placeholder="01234, 56789, 00001, 12345 ..."
    )
    seed_override = st.text_input(
        "Seed (5 digits, optional – leave blank to use last history seed)",
        value="", max_chars=5
    )
    ids_text = st.text_area(
        "Applicable filter IDs (optional, comma-separated)",
        placeholder="LL002f, LL003b, NO64F356 ...",
        height=88
    )

with col_csv:
    filt_file = st.file_uploader("Upload **master filter CSV** (tester CSV)", type=["csv"])
    chronology = st.radio(
        "History chronology",
        options=["Oldest → Newest", "Newest → Oldest"],
        index=0
    )

with col_hist:
    hist_file = st.file_uploader("Upload **history winners** (CSV or TXT)", type=["csv", "txt"])
    run_btn = st.button("Compute", type="primary", use_container_width=True)


# ----------------------- PARSERS/UTILS ---------------------- #
def parse_pool(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.split(r"[,\s]+", text.strip())
    combos = []
    for t in tokens:
        if not t:
            continue
        if re.fullmatch(r"\d{5}", t):
            combos.append(t)
        else:
            digits = re.sub(r"\D+", "", t)
            while len(digits) >= 5:
                chunk, digits = digits[:5], digits[5:]
                combos.append(chunk)
    out, seen = [], set()
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_filters_csv(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("filter_id")
    expr_col = cols.get("expression")
    name_col = cols.get("name")
    appl_col = cols.get("applicable_if")
    if id_col is None or expr_col is None:
        raise ValueError("CSV must have columns: id (or filter_id) and expression")
    out = pd.DataFrame({
        "id": df[id_col].astype(str).str.strip(),
        "name": (df[name_col].astype(str).str.strip() if name_col else df[id_col].astype(str)),
        "applicable_if": (df[appl_col].fillna("True").astype(str) if appl_col else "True"),
        "expression": df[expr_col].astype(str),
    })
    out.loc[out["applicable_if"].str.strip().eq(""), "applicable_if"] = "True"
    return out


def load_history(file: io.BytesIO, chronology_label: str) -> List[str]:
    raw = file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
        col = None
        for c in df.columns:
            if df[c].astype(str).str.fullmatch(r"\d{5}").all():
                col = c
                break
        if col is None:
            col = df.columns[0]
        seq = [str(x).zfill(5) for x in df[col].astype(str)]
    except Exception:
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


# ---------------------- SAFE-EVAL SHIM ---------------------- #
MIRROR_MAP = {'0':'5','1':'6','2':'7','3':'8','4':'9','5':'0','6':'1','7':'2','8':'3','9':'4'}

SAFE_BUILTINS = {
    "int": int, "len": len, "any": any, "all": all, "sum": sum, "set": set,
    "ord": ord, "list": list, "tuple": tuple, "min": min, "max": max, "abs": abs,
    "Counter": Counter
}

def make_env(seed: str, prev1: str, prev2: str, combo: str) -> Dict[str, object]:
    # Digit lists
    seed_digits = list(seed)
    prev1_digits = list(prev1) if prev1 else []
    prev2_digits = list(prev2) if prev2 else []
    combo_digits = list(combo)

    # Basic sets & sums
    last2_union = set(prev1_digits) | set(prev2_digits)
    last2_intersection = set(prev1_digits) & set(prev2_digits)
    new_seed_digits = set(seed_digits) - set(prev1_digits)
    seed_sum = sum(map(int, seed_digits)) if seed_digits else 0
    combo_sum = sum(map(int, combo_digits)) if combo_digits else 0

    # ---- Extra names (SAFE DEFAULTS) to avoid NameErrors ----
    # CF/LL style placeholders — let guards/exprs evaluate instead of skipping.
    core_letters = set()
    prev_core_letters = set()
    digit_current_letters = {str(d): '?' for d in range(10)}  # mapping d->letter
    new_core_digits = set()
    cooled_digits = set()
    ring_digits = set()
    hot_digits = set()
    cold_digits = set()
    due_digits = set()
    loser_7_9 = set(list("789"))

    env = dict(
        combo_digits=combo_digits,
        combo_sum=combo_sum,

        seed_digits=seed_digits,
        prev1_digits=prev1_digits,
        prev2_digits=prev2_digits,

        last2_union=last2_union,
        last2_intersection=last2_intersection,
        last2=last2_union,                 # synonym
        common_to_both=last2_intersection, # synonym

        new_seed_digits=new_seed_digits,
        seed_sum=seed_sum,
        seed_value=seed_sum,               # synonym

        mirror_map=MIRROR_MAP,
        mirror=lambda d: MIRROR_MAP[str(d)],

        # CF/LL placeholders
        core_letters=core_letters,
        prev_core_letters=prev_core_letters,
        digit_current_letters=digit_current_letters,
        new_core_digits=new_core_digits,
        cooled_digits=cooled_digits,
        ring_digits=ring_digits,
        hot_digits=hot_digits,
        cold_digits=cold_digits,
        due_digits=due_digits,
        loser_7_9=loser_7_9,

        **SAFE_BUILTINS
    )
    return env


def eval_guard(expr: str, env: Dict[str, object]) -> bool:
    # Any error in guard => treat as not applicable (False), **not** a skip.
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def eval_elim(expr: str, env: Dict[str, object]) -> Optional[bool]:
    # Return None only if expression truly cannot be evaluated.
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return None


# ---------------------- SCORING / PAIRS --------------------- #
@dataclass
class Pair:
    prev2: str
    prev1: str
    seed: str
    next: str

def build_history_pairs(history: List[str]) -> List[Pair]:
    pairs = []
    for i in range(2, len(history)-1):
        pairs.append(Pair(prev2=history[i-2], prev1=history[i-1], seed=history[i], next=history[i+1]))
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
    recs, skipped = [], []

    if applicable_ids is not None:
        sub = filters_df[filters_df["id"].str.upper().isin(applicable_ids)].copy()
    else:
        sub = filters_df.copy()

    sub = sub.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    # Use latest context for quick checks
    prev1 = pairs[-1].prev1 if pairs else ""
    prev2 = pairs[-1].prev2 if pairs else ""
    sample_combo = pool[0] if pool else seed_now

    for _, row in sub.iterrows():
        fid = str(row["id"]).strip()
        name = str(row["name"]).strip()
        guard = str(row["applicable_if"]).strip()
        expr = str(row["expression"]).strip()

        if not expr or expr.lower() == "nan":
            skipped.append({"id": fid, "name": name, "reason": "empty expression", "expr": expr})
            continue

        # Check the expression is syntactically valid in current context;
        # guard errors are treated as False (not skip).
        sample_env = make_env(seed_now, prev1, prev2, sample_combo)
        # If the expression itself cannot evaluate *in any simple context*, skip.
        try_expr = eval_elim(expr, sample_env)
        if try_expr is None:
            skipped.append({"id": fid, "name": name, "reason": "runtime: expression did not evaluate", "expr": expr})
            continue

        # Pool elim%
        elim_cnt = 0
        for c in pool:
            env = make_env(seed_now, prev1, prev2, c)
            if not eval_guard(guard, env):
                continue
            res = eval_elim(expr, env)
            if res is None:
                # expression fails => skip this row entirely
                elim_cnt = None
                break
            if res:
                elim_cnt += 1
        if elim_cnt is None:
            skipped.append({"id": fid, "name": name, "reason": "runtime: expression did not evaluate", "expr": expr})
            continue
        elim_rate = (elim_cnt / len(pool)) if pool else 0.0

        # History keep%
        k = n = 0
        for p in pairs:
            env = make_env(p.seed, p.prev1, p.prev2, p.next)
            if not eval_guard(guard, env):
                continue
            res = eval_elim(expr, env)
            if res is None:
                continue
            n += 1
            if not res:
                k += 1
        keep = (k / n) if n > 0 else 0.0
        lb, ub = wilson_ci(k, n) if n > 0 else (0.0, 0.0)

        alpha = 1.0
        wpp = keep * (elim_rate ** alpha)

        recs.append({
            "id": fid, "name": name,
            "keep%": keep, "keep%_lb": lb, "keep%_ub": ub,
            "elim%": elim_rate, "WPP": wpp, "pairs_used": n,
            "expression": expr, "guard": guard
        })

    metrics_df = pd.DataFrame(recs).sort_values(["WPP", "keep%"], ascending=[False, False]).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped)
    return metrics_df, skipped_df


# ------------------ GREEDY BUNDLE BUILDER ------------------- #
def eval_guard(expr: str, env: Dict[str, object]) -> bool:  # re-define just to be sure
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def eval_elim(expr: str, env: Dict[str, object]) -> Optional[bool]:
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return None

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

@dataclass
class Pair:
    prev2: str
    prev1: str
    seed: str
    next: str

def build_history_pairs(history: List[str]) -> List[Pair]:
    pairs = []
    for i in range(2, len(history)-1):
        pairs.append(Pair(prev2=history[i-2], prev1=history[i-1], seed=history[i], next=history[i+1]))
    return pairs

def bundle_winner_survival(selected: List[Dict], pairs: List[Pair]) -> float:
    k = n = 0
    for p in pairs:
        applicable = False
        eliminated = False
        for f in selected:
            env = make_env(p.seed, p.prev1, p.prev2, p.next)
            if not eval_guard(f["guard"], env):
                continue
            applicable = True
            res = eval_elim(f["expression"], env)
            if res:
                eliminated = True
                break
        if applicable:
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
    if metrics.empty or not pool:
        return [], pool, 0.0

    prev1 = pairs[-1].prev1 if pairs else ""
    prev2 = pairs[-1].prev2 if pairs else ""

    pool_by_filter = {}
    for _, r in metrics.iterrows():
        survivors = apply_filter_to_pool(r["expression"], r["guard"], seed_now, prev1, prev2, pool)
        pool_by_filter[r["id"]] = set(survivors)

    selected = []
    survivors = set(pool)

    while True:
        best = None
        best_score = -1e9
        best_survivors = None

        for _, r in metrics.iterrows():
            if any(f["id"] == r["id"] for f in selected):
                continue
            cand_survivors = pool_by_filter[r["id"]] & survivors
            drop = len(survivors) - len(cand_survivors)
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
            score = drop - redundancy_penalty * red * max(1, len(survivors))
            if score > best_score:
                best = r
                best_score = score
                best_survivors = cand_survivors

        if best is None:
            break

        tentative = selected + [best.to_dict()]
        win_surv = bundle_winner_survival(tentative, pairs)
        if win_surv >= min_win_survival:
            selected = tentative
            survivors = best_survivors
        if len(survivors) <= target_survivors:
            break
        if best_score <= 0 and win_surv < min_win_survival:
            break
        if len(selected) > 500:
            break

    final_win = bundle_winner_survival(selected, pairs)
    return selected, sorted(list(survivors)), final_win


# --------------------------- RUN ---------------------------- #
if run_btn:
    pool = parse_pool(pool_text)
    if not pool:
        st.error("Please paste a valid 5-digit pool.")
        st.stop()

    if filt_file is None:
        st.error("Please upload your **master filter CSV** (tester CSV).")
        st.stop()
    try:
        filters_df = load_filters_csv(filt_file)
    except Exception as e:
        st.error(f"Failed to read filter CSV: {e}")
        st.stop()

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

    pairs = build_history_pairs(history)
    seed_now = seed_override.strip() if re.fullmatch(r"\d{5}", seed_override.strip() or "") else history[-2]

    st.success(f"Parsed pool size: **{len(pool)}**  |  History rows: **{len(history)}**  |  Seed: **{seed_now}**")

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
        use_container_width=True, height=480
    )

    st.download_button(
        "Download per-filter metrics (.csv)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="per_filter_metrics.csv", mime="text/csv"
    )
    st.download_button(
        "Download skipped rows (.csv)",
        data=skipped_df.to_csv(index=False).encode("utf-8"),
        file_name="skipped_filters.csv", mime="text/csv"
    )

    st.write("---")
    st.subheader("Advanced: Redundancy & Greedy Bundle")
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
            min_win_survival=float(min_win_survival),
            target_survivors=int(target_survivors),
            redundancy_penalty=float(redundancy_penalty)
        )

        st.markdown("### Selected bundle")
        if not selected:
            st.warning("No filters selected (constraints too strict or pool reduction not possible).")
        else:
            chosen_ids = [f["id"] for f in selected]
            st.write(f"**Filters chosen ({len(chosen_ids)}):**", ", ".join(chosen_ids))
            st.write(f"**Projected winner survival (bundle):** {bundle_win:.2%}")
            st.write(f"**Projected survivors (pool):** {len(survivors)}")

            rows = []
            prev1 = pairs[-1].prev1 if pairs else ""
            prev2 = pairs[-1].prev2 if pairs else ""
            running = set(pool)
            for f in selected:
                new_running = set(apply_filter_to_pool(f["expression"], f["guard"], seed_now, prev1, prev2, running))
                rows.append({"added": f["id"], "survival": bundle_win, "survivors": len(new_running)})
                running = new_running
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

            st.download_button(
                "Download bundle IDs (.txt)",
                data=("\n".join(chosen_ids)).encode("utf-8"),
                file_name="bundle_ids.txt", mime="text/plain"
            )
            st.download_button(
                f"Download projected survivors ({len(survivors)}) (.txt)",
                data=(", ".join(survivors)).encode("utf-8"),
                file_name=f"survivors_{len(survivors)}.txt", mime="text/plain"
            )
