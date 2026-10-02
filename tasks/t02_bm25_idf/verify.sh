#!/usr/bin/env bash
set -e
python -m pytest test_bm25.py -q >/dev/null 2>&1 || exit 1
# 回归：不得用 print/硬编码绕过（实现必须含 math.log 或等价对数）
grep -qE "log|ln" bm25.py
exit 0
