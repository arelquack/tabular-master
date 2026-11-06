import os
import re

root_dir = '.'
patterns = [
    re.compile(r'[a-zA-Z]:[\\/][^ \r\n"\'<>]+', re.IGNORECASE),
    re.compile(r'/home/[^ \r\n"\'<>]+', re.IGNORECASE),
    re.compile(r'/Users/[^ \r\n"\'<>]+', re.IGNORECASE)
]

ignored_dirs = {'.git', '.venv', 'node_modules', 'dist', '__pycache__', 'archive', 'experiments'}
ignored_exts = {'.png', '.jpg', '.pdf', '.npy', '.zip', '.parquet', '.log', '.csv', '.tsv', '.xlsx', '.json'}

found = []
for r, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if d not in ignored_dirs]
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ignored_exts:
            continue
        p = os.path.join(r, f)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                for idx, line in enumerate(fp, 1):
                    for pat in patterns:
                        m = pat.search(line)
                        if m:
                            val = m.group(0)
                            if any(k in val for k in ['POLBAN', 'Lenovo', 'Users/', 'Users\\', 'home/', 'home\\']):
                                found.append((p, idx, val))
        except Exception:
            pass

print(f"Total hardcoded path references found: {len(found)}")
for p, idx, val in found:
    print(f"{p}:{idx} -> {val}")
