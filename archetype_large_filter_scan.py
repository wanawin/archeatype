# archetype_large_filter_scan.py
from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Core helpers & signals
# -----------------------
VTRAC = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}
MIRROR = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def spread_band(spread: int) -> str:
    if spread <= 3: return "0–3"
    if spread <= 5: return "4–5"
    if spread <= 7: return "6–7"
    if spread <= 9: return "8–9"
    return "10+"

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def digits_of(s: str):
    return [int(ch) for ch in str(s)]

def hot_cold_due(winners_digits, k=10):
    if not winners_digits:
        return set(), set(), set(range(10))
    hist = winners_digits[-k:] if len(winners_digits) >= k else winners_digits
    flat = [d for row in hist for d in row]
    cnt = Counter(flat)
    if not cnt:
        return set(), set(), set(range(10))
    most = cnt.most_common()
    topk = 6
    thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
    hot = {d for d,c in most if c >= thresh}

    least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
    coldk = 4
    cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
    cold = {d for d,c in least if c <= cth}

    last2 = set(d for row in winners_digits[-2:] for d in row)
    due  = set(range(10)) - last2
    return hot, cold, due

def build_day_env(winners_list, i):
    """
    Environment for historical day i (i>=1):
    seed = winners[i-1], winner/combo = winners[i]
    """
    seed = winners_list[i-1]
    winner = winners_list[i]
    sd = digits_of(seed)
    cd = sorted(digits_of(winner))
    hist_digits = [digits_of(x) for x in winners_list[:i]]
    hot, cold, due = hot_cold_due(hist_digits, k=10)

    prev_seed = winners_list[i-2] if i-2 >= 0 else ""
    prev_prev = winners_list[i-3] if i-3 >= 0 else ""
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev) if prev_prev else []

    prev_pattern = []
    for digs in (ppdigs, pdigs, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    env = {
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': set(sd) - set(pdigs),
        'seed_counts': Counter(sd),
        'seed_sum': sum(sd),
        'prev_sum_cat': sum_category(sum(sd)),
        'prev_pattern': tuple(prev_pattern),

        'combo': winner,
        'combo_digits': cd,
        'combo_digits_list': cd,
        'combo_sum': sum(cd),
        'combo_sum_cat': sum_category(sum(cd)),
        'combo_sum_category': sum_category(sum(cd)),

        'seed_vtracs': set(VTRAC[d] for d in sd),
        'combo_vtracs': set(VTRAC[d] for d in cd),
        'mirror': MIRROR,

        'hot_digits': sorted(hot),
        'cold_digits': sorted(cold),
        'due_digits': sorted(due),

        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted, 'Counter': Counter
    }

    # --- added for DC-5 CSV compatibility ---
    env['seed_value'] = int(seed)
    env['nan'] = float('nan')
    env['winner_structure'] = classify_structure(sd)
    env['combo_structure'] = classify_structure(cd)
    # ----------------------------------------

    env['combo_sum_category'] = env['combo_sum_cat']
    return env

def build_ctx_for_pool(seed, prev_seed, prev_prev):
    sd = digits_of(seed)
    pdigs = digits_of(prev_seed) if prev_seed else []
    ppdigs = digits_of(prev_prev) if prev_prev else []

    new_digits = set(sd) - set(pdigs)
    seed_counts = Counter(sd)
    seed_vtracs = set(VTRAC[d] for d in sd)
    prev_sum_cat = sum_category(sum(sd))
    prev_pattern = []
    for digs in (ppdigs, pdigs, sd):
        if digs:
            parity = 'Even' if sum(digs) % 2 == 0 else 'Odd'
            prev_pattern.extend([sum_category(sum(digs)), parity])
        else:
            prev_pattern.extend(['', ''])

    base = {
        'seed_digits': sd,
        'prev_seed_digits': pdigs,
        'prev_prev_seed_digits': ppdigs,
        'new_seed_digits': new_digits,
        'prev_pattern': tuple(prev_pattern),
        'hot_digits': [],
        'cold_digits': [],
        'due_digits': [d for d in range(10) if d not in pdigs and d not in ppdigs],
        'seed_counts': seed_counts,
        'seed_sum': sum(sd),
        'prev_sum_cat': prev_sum_cat,
        'seed_vtracs': seed_vtracs,
        'mirror': MIRROR,
        'Counter': Counter,
        'any': any, 'all': all, 'len': len, 'sum': sum,
        'max': max, 'min': min, 'set': set, 'sorted': sorted
    }

    # --- added for DC-5 CSV compatibility ---
    base['seed_value'] = int(seed)
    base['nan'] = float('nan')
    base['winner_structure'] = classify_structure(sd)
    base['combo_structure'] = classify_structure(sd)  # placeholder when combos unknown
    # ----------------------------------------

    return base
