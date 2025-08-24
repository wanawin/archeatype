def historical_safety_for_archetype(
    winners: List[str],
    filters: List[FilterDef],
    arch_name_now: str,
    return_audit: bool = False,
):
    N = len(winners)
    out: Dict[str, Dict[str, float]] = {f.fid: {"days": 0, "app": 0, "kept": 0} for f in filters if f.enabled}
    audit_rows = []
    if N < 2:
        return (out, pd.DataFrame()) if return_audit else out

    envs = [build_env_for_draw(i, winners) for i in range(1, N)]
    arch_keys = [day_arche_key(winners, i) for i in range(1, N)]

    for i, env in enumerate(envs, start=1):
        if arch_keys[i-1] != arch_name_now:
            continue
        for f in filters:
            if not f.enabled:
                continue
            out[f.fid]["days"] += 1
            applicable = safe_eval(f.applicable_if, env)
            blocked = False
            if applicable:
                out[f.fid]["app"] += 1
                blocked = safe_eval(f.expression, env)
                if not blocked:
                    out[f.fid]["kept"] += 1
            if return_audit:
                audit_rows.append({
                    "day_index": i,
                    "seed": env["seed"],
                    "winner": env["combo"],
                    "archetype": arch_keys[i-1],
                    "filter_id": f.fid,
                    "filter_name": f.name,
                    "applicable": bool(applicable),
                    "blocked_winner": bool(blocked),
                })

    for fid, d in out.items():
        d["kept_pct"] = (d["kept"] / d["app"] * 100.0) if d["app"] > 0 else None

    if return_audit:
        return out, pd.DataFrame(audit_rows)
    return out
