#!/usr/bin/env bash
set -e
python -m pytest test_aggregate.py -q >/dev/null 2>&1 || exit 1
exit 0
