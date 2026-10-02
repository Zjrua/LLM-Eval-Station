#!/usr/bin/env bash
set -e
python -m pytest test_metrics.py -q >/dev/null 2>&1 || exit 1
# 防作弊：不得用 if 特判测试值；必须真除
grep -q "/" metrics.py
! grep -q "//" metrics.py || exit 1
exit 0
