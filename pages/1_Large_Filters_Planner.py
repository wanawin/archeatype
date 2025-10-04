def combo_env(base_env: Dict, combo: str) -> Dict:
    cd = digits_of(combo)
    env = dict(base_env)
    combo_set = set(cd)                     # NEW

    # NEW: mirror calculations for the combo
    combo_mirror_digits = sorted({base_env["mirror"][d] for d in cd}) if cd else []
    mirror_pairs = {tuple(sorted((d, base_env["mirror"][d]))) for d in cd
                    if base_env["mirror"][d] in combo_set and base_env["mirror"][d] != d}
    mirror_pair_count = len(mirror_pairs)
    has_mirror_pair = mirror_pair_count > 0
    mirror_overlap_with_seed = len(combo_set & set(base_env.get("seed_mirror_digits", [])))

    env.update({
        "combo": combo,
        "combo_digits": sorted(cd),
        "combo_digits_list": sorted(cd),
        "combo_set": combo_set,                         # NEW
        "combo_sum": sum(cd),
        "combo_sum_cat": sum_category(sum(cd)),
        "combo_sum_category": sum_category(sum(cd)),
        "combo_vtracs": set(VTRAC[d] for d in cd),
        "combo_structure": classify_structure(cd),
        "last_digit": cd[-1] if cd else None,
        "spread": (max(cd) - min(cd)) if cd else 0,
        "seed_spread": (max(base_env["seed_digits"]) - min(base_env["seed_digits"])) if base_env["seed_digits"] else 0,

        # NEW: expose mirror values
        "combo_mirror_digits": combo_mirror_digits,
        "has_mirror_pair": has_mirror_pair,
        "mirror_pair_count": mirror_pair_count,
        "mirror_overlap_with_seed": mirror_overlap_with_seed,
    })
    return env
