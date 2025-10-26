# filter_picker_universal.py
# Full, self-contained Streamlit app that ingests your "universal" CSV as-is.
# It auto-detects logic across columns (incl. Unnamed), normalizes expressions,
# supports both int and string digit semantics, and builds a greedy bundle.

import io
import math
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Filter Picker — Universal CSV", layout="wide")


# -----------------------------
# Helpers: expression harvesting & normalization
# -----------------------------

BOOL_HINTS = (
    "combo_digits", "combo_sum", "seed_digits", "prev_seed",
    "any(", "all(", "len(", " in ", "==", "!=", "<", ">", "<=", ">=", "%", " and ", " or ", " not "
)

def looks_like_bool_code(s: str) -> bool:
    s = s.strip()
    if not s or s.lower() in ("n/a", "na", "none"):
        return False
    if s.lower() == "true":
        return True
    # If it has operators or known names, treat as boolean-ish
    return any(h in s for h in BOOL_HINTS)

def normalize_quotes_and_wrapping(s: str) -> str:
    """Unify curly quotes, collapse doubled quotes, strip one layer of outer quotes."""
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("—", "-")
           .replace("\u00A0", " ")
           .strip())
    prev = None
    # Iteratively collapse "" → " and strip outer quotes once per pass
    while prev != s:
        prev = s
        s = s.replace('""', '"').replace("''", "'").strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1].strip()
    return s.strip()

def auto_quote_digits_in_sets_and_lists(expr: str, prefer_strings: bool) -> str:
    """
    Convert {1,2,9} → {'1','2','9'} (and [1,2] → ['1','2']) when expressions
    likely operate on string digits. If prefer_strings=False, leave them as ints.
    """
    if not expr or not prefer_strings:
        return expr

    def conv_items(text, opener, closer):
        parts = [p.strip() for p in text.split(",")]
        out = []
        for p in parts:
            if re.fullmatch(r"[\"'][0-9][\"']", p):  # already '7' or "7"
                out.append(p)
            elif re.fullmatch(r"[0-9]", p):
                out.append(f"'{p}'")
            else:
                out.append(p)
        return opener + ",".join(out) + closer

    # Sets
    expr = re.sub(r"\{([^{}]+)\}", lambda m: conv_items(m.group(1), "{", "}"), expr)
    # Lists
    expr = re.sub(r"\[([^\[\]]+)\]", lambda m: conv_items(m.group(1), "[", "]"), expr)
    return expr

def harvest_logic_from_row(row: pd.Series) -> Tuple[str, str]:
    """
    Return (applicable_if, expression) harvested from any relevant columns in the row.
    Priority:
      - explicit 'applicable_if' and 'expression'
      - any columns whose names contain those strings
      - any 'Unnamed:*' columns that look like boolean code
    """
    cols = {c.lower().strip(): c for c in row.index}

    # Canonicals if present
    app_if = ""
    expr   = ""

    # 1) direct matches
    if "applicable_if" in cols:
        app_if = str(row[cols["applicable_if"]])
    if "expression" in cols:
        expr = str(row[cols["expression"]])

    # 2) near matches by name
    if not app_if:
        for c in row.index:
            if "applicable" in c.lower():
                app_if = str(row[c])
                break
    if not expr:
        # prefer a column literally named 'expression' or with 'expr'
        for c in row.index:
            lc = c.lower()
            if "expression" in lc or "expr" in lc:
                val = str(row[c])
                if val and val.strip() and looks_like_bool_code(val):
                    expr = val
                    break

    # 3) scan Unnamed for leftover logic
    # We assign leftover boolean-looking strings as expression if expr is empty,
    # otherwise append them to applicable_if as extra gating, whichever makes more sense.
    if not app_if or not expr:
        leftovers = []
        for c in row.index:
            if c.lower().startswith("unnamed"):
                val = str(row[c]).strip()
                if val and looks_like_bool_code(val):
                    leftovers.append(val)
        if leftovers:
            extra = " and ".join(f"({normalize_quotes_and_wrapping(v)})" for v in leftovers)
            if not expr:
                expr = extra
            else:
                # if expr already present, treat leftovers as gating
                app_if = f"({app_if}) and ({extra})" if app_if else extra

    # final tidy
    app_if = normalize_quotes_and_wrapping(app_if)
    expr   = normalize_quotes_and_wrapping(expr)
    return app_if, expr


# -----------------------------
# CSV loader (universal)
# -----------------------------
def load_universal_csv(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    # We keep all columns (do not drop Unnamed) because logic may hide there.

    # Derive id/name
    cols = {c.lower().strip(): c for c in df.columns}
    if "id" in cols:
        df["id"] = df[cols["id"]].astype(str).str.strip()
    elif "name" in cols:
        df["id"] = df[cols["name"]].astype(str).str.strip()
    else:
        df["id"] = [f"F{i+1:04d}" for i in range(len(df))]

    if "name" in cols:
        df["name"] = df[cols["name"]].astype(str).str.strip()
    else:
        df["name"] = df["id"]

    # Harvest logic per row
    app_ifs, exprs = [], []
    for _, r in df.iterrows():
        ai, ex = harvest_logic_from_row(r)
        app_ifs.append(ai)
        exprs.append(ex)

    df["applicable_if_raw"] = app_ifs
    df["expression_raw"]    = exprs

    # Compose final expression: (app_if) and (expr) if both exist
    finals, reasons, prefer_strings_flags = [], [], []
    for ai, ex in zip(app_ifs, exprs):
        ai = ai.strip()
        ex = ex.strip()
        # Decide if we should auto-quote digit sets/lists (prefer strings)
        # Heuristic: if expression mentions combo_digits or uses quotes around digits anywhere, prefer strings
        prefer_strings = ("combo_digits" in ex) or ("'" in ai+ex) or ('"' in ai+ex)
        prefer_strings_flags.append(prefer_strings)

        # Normalize digits where needed (string-digit context)
        ai_n = auto_quote_digits_in_sets_and_lists(ai, prefer_strings)
        ex_n = auto_quote_digits_in_sets_and_lists(ex, prefer_strings)

        if ex_n and ai_n:
            finals.append(f"(({ai_n})) and (({ex_n}))")
            reasons.append("")
        elif ex_n:
            finals.append(ex_n)
            reasons.append("")
        elif ai_n:
            # An applicable_if with no expression: treat as the actual rule (rare)
            finals.append(ai_n)
            reasons.append("")
        else:
            finals.append("")
            reasons.append("no_logic_found")

    out = pd.DataFrame({
        "id": df["id"],
        "name": df["name"],
        "applicable_if": df["applicable_if_raw"],
        "expression": df["expression_raw"],
        "final_expr": finals,
        "why_empty": reasons,
        "prefer_strings": prefer_strings_flags
    })
    # Drop rows with truly no logic
    out = out[out["final_expr"].str.strip() != ""].reset_index(drop=True)
    return out


# -----------------------------
# Evaluation environment (universal)
# -----------------------------
SAFE_FUNCS = {
    "set": set, "sum": sum, "len": len, "any": any, "all": all,
    "max": max, "min": min, "sorted": sorted,
}

def build_env(seed: str, combo: str) -> Dict:
    combo_digits = list(combo)                    # ['8','8','0','0','1']
    combo_digits_int = [int(d) for d in combo]    # [8,8,0,0,1]
    combo_sum = sum(combo_digits_int)
    seed_digits = set(int(d) for d in seed) if seed else set()
    seed_digits_str = set(seed) if seed else set()

    env = {
        # both flavors for maximum compatibility
        "combo_digits": combo_digits,               # strings
        "combo_digits_int": combo_digits_int,       # ints
        "combo_sum": combo_sum,
        "seed_digits": seed_digits,                 # ints (tester style)
        "seed_digits_str": seed_digits_str,         # strings (LL/text style)
    }
    env.update(SAFE_FUNCS)
    return env

def eval_rule(expr: str, env: Dict) -> bool:
    s = expr.strip()
    if not s:
        return False
    if s.lower() == "true":
        return True
    # Safety: no builtins; this matches your tester’s style
    return bool(eval(s, {"__builtins__": {}}, env))


# -----------------------------
# Metrics, scoring & greedy bundle
# -----------------------------
def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * math.sqrt((p_hat*(1 - p_hat) + z**2/(4*n))/n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def compute_metrics(
    df: pd.DataFrame,
    pool: List[str],
    seed: str,
    winners: List[str],
    alpha_wpp: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str,set], pd.DataFrame]:
    """
    Evaluate each final_expr on pool and history, producing:
      - metrics dataframe
      - elim_sets: {id -> set(pool indices eliminated)}
      - skipped dataframe with reasons
    """
    n_pool = len(pool)
    pool_envs = [build_env(seed, c) for c in pool]
    hist_envs = [build_env(seed, w) for w in winners]

    rows, skipped_rows = [], []
    elim_sets: Dict[str,set] = {}

    for _, r in df.iterrows():
        fid, name, expr = str(r["id"]), str(r["name"]), str(r["final_expr"])

        # Evaluate on pool
        eliminated = set()
        compile_failed = False
        try:
            # quick sanity parse (let Python raise on syntax if broken)
            _ = compile(expr, "<expr>", "eval")
        except Exception as e:
            skipped_rows.append({"id": fid, "name": name, "reason": f"syntax: {e}", "expr": expr})
            compile_failed = True

        if not compile_failed:
            for i, env in enumerate(pool_envs):
                try:
                    if eval_rule(expr, env):
                        eliminated.add(i)
                except Exception as e:
                    skipped_rows.append({"id": fid, "name": name, "reason": f"runtime: {e}", "expr": expr})
                    compile_failed = True
                    break

        if compile_failed:
            elim_sets[fid] = set()
            rows.append({
                "id": fid, "name": name, "keep%": np.nan, "keep%_lb": np.nan, "keep%_ub": np.nan,
                "elim%": np.nan, "WPP": np.nan, "final_expr": expr
            })
            continue

        elim_sets[fid] = eliminated
        elim_pct = len(eliminated) / max(1, n_pool)

        # Keep% on winners
        keeps = 0
        for env in hist_envs:
            try:
                fires = eval_rule(expr, env)
            except Exception:
                fires = False  # treat failure as "kept"
            keeps += (0 if fires else 1)
        n_w = len(hist_envs)
        keep_pct = (keeps / n_w) if n_w > 0 else float("nan")
        lb, ub = (wilson_ci(keep_pct, n_w) if n_w > 0 else (float("nan"), float("nan")))

        WPP = (0.0 if (keep_pct != keep_pct) else keep_pct) * (elim_pct ** float(alpha_wpp))
        rows.append({
            "id": fid, "name": name,
            "keep%": keep_pct, "keep%_lb": lb, "keep%_ub": ub,
            "elim%": elim_pct, "WPP": WPP,
            "final_expr": expr
        })

    metrics = pd.DataFrame(rows).sort_values(["WPP","keep%","elim%"], ascending=[False,False,False]).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows)
    return metrics, elim_sets, skipped

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def greedy_bundle(
    metrics: pd.DataFrame,
    elim_sets: Dict[str,set],
    n_pool: int,
    min_survival: float,
    target_survivors: int,
    redundancy_penalty: float
) -> Tuple[List[str], float, int, List[Tuple[str,float,int]]]:
    chosen, union, steps = [], set(), []
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
            if fid in chosen: 
                continue
            S = elim_sets.get(fid, set())
            new_union = union | S
            new_survivors = max(0, n_pool - len(new_union))
            cand_keep = float(r["keep%"])
            cand_survival = win_survival * cand_keep
            if cand_survival < min_survival:
                continue
            red = 0.0 if not union else jaccard(S, union)
            score = ((survivors - new_survivors) + 1000.0*float(r["WPP"]) - 200.0*redundancy_penalty*red)
            if score > best_score:
                best_score = score
                best = (fid, new_union, new_survivors, cand_survival)
        if not best:
            break
        fid, union, survivors, win_survival = best
        chosen.append(fid)
        steps.append((fid, win_survival, survivors))
        if survivors <= target_survivors:
            break

    return chosen, win_survival, survivors, steps


# -----------------------------
# UI
# -----------------------------
st.title("Filter Picker — Universal CSV")

with st.expander("Paste current pool — continuous or comma/space/newline-separated", expanded=True):
    pool_text = st.text_area("Pool", height=140, placeholder="01234, 98765, ... or a long digit blob 0123498765...")
    # Parse pool
    tokens = re.findall(r"\d+", pool_text or "")
    pool: List[str] = []
    for t in tokens:
        if len(t) == 5:
            pool.append(t)
        elif len(t) > 5:
            for i in range(0, len(t) - 4, 5):
                pool.append(t[i:i+5])
    # de-dupe preserving order
    seen = set(); pool_clean = []
    for c in pool:
        if c not in seen:
            seen.add(c); pool_clean.append(c)
    pool = pool_clean
st.caption(f"Parsed pool size: **{len(pool)}**")

c1, c2, c3 = st.columns(3)
with c1:
    seed = st.text_input("Seed (last 5)", value="", max_chars=5)
    seed = re.sub(r"\D", "", seed).zfill(5)[:5]
with c2:
    alpha = st.number_input("WPP α (elim exponent)", 0.1, 3.0, 1.5, 0.1)
with c3:
    chronology = st.radio("History chronology", ["Newest→Oldest", "Oldest→Newest"], index=0, horizontal=True)

u1, u2 = st.columns(2)
with u1:
    filters_file = st.file_uploader("Upload universal CSV (same one used by your other apps)", type=["csv"])
with u2:
    history_file = st.file_uploader("Upload winners history (csv/txt)", type=["csv","txt"])

st.markdown("### Applicable filter IDs (optional)")
ids_text = st.text_area("Paste IDs (comma/space/newline); leave blank to use ALL", height=110)
app_ids = [t.strip() for t in re.split(r"[,\s]+", ids_text.strip()) if t.strip()] if ids_text.strip() else []

run = st.button("🚀 Compute metrics", type="primary", use_container_width=True)

# Load history
winners: List[str] = []
if history_file:
    raw = history_file.read().decode("utf-8", errors="ignore")
    toks = re.findall(r"\d+", raw)
    for t in toks:
        if len(t) == 5:
            winners.append(t)
        elif len(t) > 5:
            for i in range(0, len(t) - 4, 5):
                winners.append(t[i:i+5])
    if chronology == "Newest→Oldest":
        pass
    else:
        winners = list(reversed(winners))

st.caption(f"History rows: **{len(winners)}**")

if run:
    if not pool:
        st.warning("Please paste the current pool."); st.stop()
    if not seed or len(seed) != 5:
        st.warning("Please enter a 5-digit seed."); st.stop()
    if not winners:
        st.warning("Please upload a winners history file."); st.stop()
    if filters_file is None:
        st.warning("Please upload the universal CSV."); st.stop()

    # Load & harvest logic
    with st.spinner("Parsing universal CSV and harvesting expressions…"):
        try:
            harvested = load_universal_csv(io.BytesIO(filters_file.getvalue()))
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            st.stop()

    st.success(f"Found {len(harvested)} rows with executable logic.")

    # Apply Applicable IDs subset
    use_df = harvested
    matched = None
    if app_ids:
        idset = {x.lower() for x in app_ids}
        use_df = harvested[harvested["id"].str.lower().isin(idset)].reset_index(drop=True)
        matched = len(use_df)
        if use_df.empty:
            st.error("None of the pasted IDs matched the CSV.")
            st.stop()

    # Evaluate
    with st.spinner("Evaluating rules against pool and history…"):
        metrics, elim_sets, skipped = compute_metrics(use_df, pool, seed, winners, alpha_wpp=alpha)

    compiled = metrics["keep%"].notna().sum()
    skipped_n = 0 if skipped is None else len(skipped)
    msg = f"Compiled **{compiled}**; Skipped **{skipped_n}**."
    if matched is not None:
        msg += f" Matched Applicable IDs: **{matched}** / {len(app_ids)}."
    st.success(msg)

    if skipped_n:
        with st.expander("Skipped rows (reason + expression)"):
            st.dataframe(skipped[["id","name","reason","expr"]], use_container_width=True, hide_index=True)

    st.subheader("Per-filter metrics")
    show = metrics.copy()
    show["keep%"] = (show["keep%"]*100).round(2)
    show["elim%"] = (show["elim%"]*100).round(2)
    show["WPP"] = show["WPP"].astype(float).round(6)
    st.dataframe(show[["id","name","keep%","keep%_lb","keep%_ub","elim%","WPP","final_expr"]],
                 use_container_width=True, hide_index=True)

    # ---------------- Greedy bundle ----------------
    st.markdown("---")
    st.subheader("Build greedy bundle")
    b1, b2, b3 = st.columns(3)
    with b1:
        min_survival = st.slider("Min winner survival", 0.50, 0.99, 0.75, 0.01)
    with b2:
        target_survivors = st.number_input("Target survivors", 1, len(pool), 50, 1)
    with b3:
        red_pen = st.slider("Redundancy penalty", 0.0, 1.0, 0.30, 0.05)

    if st.button("Build bundle", type="primary"):
        chosen, surv_prob, survivors, steps = greedy_bundle(
            metrics, elim_sets, len(pool),
            float(min_survival), int(target_survivors), float(red_pen)
        )
        st.write("**Selected IDs:**", ", ".join(chosen) if chosen else "—")
        st.write(f"Projected winner survival: **{surv_prob:.2%}**")
        st.write(f"Projected survivors: **{survivors:,}**")

        if steps:
            st.dataframe(pd.DataFrame(steps, columns=["added","survival","survivors"]),
                         use_container_width=True, hide_index=True)

        # Survivors for download
        union = set()
        for fid in chosen:
            union |= elim_sets.get(fid, set())
        survivors_idx = [i for i in range(len(pool)) if i not in union]
        survivors_list = [pool[i] for i in survivors_idx]
        st.text_area("Survivors (preview)", value=", ".join(survivors_list)[:2000], height=110)
        st.download_button("Download survivors (.txt)",
                           ("\n".join(survivors_list)).encode("utf-8"),
                           file_name=f"survivors_{survivors}.txt", use_container_width=True)
else:
    st.info("Paste pool, upload universal CSV + history, enter seed, optionally paste Applicable IDs, then click **Compute metrics**.")
