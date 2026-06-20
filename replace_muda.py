#!/usr/bin/env python3
import os
import re

root = "StiffGIPC"
exclude_dir = os.path.join(root, "muda")

def should_process(path):
    if path.startswith(exclude_dir + os.sep) or path == exclude_dir:
        return False
    if path.endswith(".cu") or path.endswith(".cuh") or path.endswith(".h") or path.endswith(".cpp") or path.endswith(".hpp") or path.endswith(".inl"):
        return True
    return False

include_re = re.compile(r'#include\s+<muda/[^>]+>')
using_re = re.compile(r'using\s+namespace\s+muda\s*;')
muda_ref_re = re.compile(r'\bmuda::\b')

modified = []
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        path = os.path.join(dirpath, fn)
        if not should_process(path):
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        orig = content
        content = include_re.sub('#include <gipc/cuda/all.h>', content)
        content = using_re.sub('using namespace gipc::cuda;', content)
        content = muda_ref_re.sub('gipc::cuda::', content)
        if content != orig:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            modified.append(path)

print(f"Modified {len(modified)} files:")
for p in modified:
    print(p)
