#!/usr/bin/env bash
set -e
python -m pytest test_report.py -q >/dev/null 2>&1 || exit 1
exit 0
