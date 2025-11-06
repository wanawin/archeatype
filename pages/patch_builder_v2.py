# save as patch_builder_v2.py and run it with:
#   python patch_builder_v2.py "loserlist_w_filters_17_UNI_fix.py"
# It will write: loserlist_w_filters.py

import sys, re, hashlib, pathlib

src_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "loserlist_w_filters_17_UNI_fix.py")
if not src_path.exists():
    raise FileNotFoundError(f"Source file not found: {src_path}")

code = src_path.read_text(encoding="utf-8", errors="ignore")
orig_sha = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()

# --- A) Quote digits in fmt_digits_list so outputs are strings like ['5','0'] ---
def patch_fmt_digits_list(s: str) -> str:
    pat = re.compile(r"(def\s+fmt_digits_list\s*\(\s*xs\s*\)\s*:\s*\n)(\s*return[^\n]*\n)", re.MULTILINE)
    def repl(m):
        indent = re.match(r"(\s*)", m.group(2)).group(1)
        return m.group(1) + indent + "return \"[\" + \",\".join(f\"'{str(int(d))}'\" for d in xs) + \"]\"\n"
    return pat.sub(repl, s)

patched = patch_fmt_digits_list(code)

# --- B) Restrict substitutions to membership only (avoid bare-name replacement) ---
loop_pat = re.compile(
    r"(for\s+name,\s*arr\s+in\s*list_vars\.items\(\)\s*:\s*\n\s*lit\s*=\s*fmt_digits_list\(arr\)\s*\n)"
    r"\s*x\s*=\s*re\.sub\([^\n]*\n\s*x\s*=\s*re\.sub\([^\n]*\n",
    re.MULTILINE
)

def repl_loop(m):
    prefix = m.group(1)
    # keep indentation consistent
    indent = " " * 8
    lines = [
        indent + r'x = re.sub(rf"\bin\s+{name}\b",      " in " + lit, x)',
        indent + r'x = re.sub(rf"\bnot\s+in\s+{name}\b"," not in " + lit, x)',
    ]
    return prefix + "\n".join(lines) + "\n"

if loop_pat.search(patched):
    patched = loop_pat.sub(repl_loop, patched, count=1)
else:
    # fallback: remove the bare-name replacement if it exists
    patched = patched.replace(
        '    for name, arr in list_vars.items():\n        lit = fmt_digits_list(arr)\n        x = re.sub(rf"\\bin\\s+{name}\\b", " in " + lit, x)\n        x = re.sub(rf"\\b{name}\\b", lit, x)\n',
        '    for name, arr in list_vars.items():\n        lit = fmt_digits_list(arr)\n        x = re.sub(rf"\\bin\\s+{name}\\b", " in " + lit, x)\n        x = re.sub(rf"\\bnot\\s+in\\s+{name}\\b", " not in " + lit, x)\n'
    )

# --- C) Ensure UNI vars are exposed (no-op if already present) ---
res_idx = patched.find("def resolve_expression")
if res_idx != -1:
    lv_start = patched.find("list_vars", res_idx)
    lv_brace = patched.find("{", lv_start)
    if lv_start != -1 and lv_brace != -1:
        # after "due_last2" entry, add UNION_DIGITS and UNION2 if missing
        if '"UNION_DIGITS"' not in patched or '"UNION2"' not in patched:
            map_due = patched.find('"due_last2"', lv_brace)
            if map_due != -1:
                line_after_due = patched.find("\n", map_due) + 1
                extra = ''
                if '"UNION_DIGITS"' not in patched:
                    extra += '        "UNION_DIGITS":     ctx.get("UNION_DIGITS", []),\n'
                if '"UNION2"' not in patched:
                    extra += '        "UNION2":           ctx.get("UNION2", []),\n'
                patched = patched[:line_after_due] + extra + patched[line_after_due:]

# --- Write out v2 file ---
out_path = src_path.with_name(src_path.stem + "_v2.py")
out_path.write_text(patched, encoding="utf-8")
new_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

print("Patched:", src_path.name)
print("Original SHA256:", orig_sha)
print("Wrote:", out_path.name)
print("New SHA256:", new_sha)
