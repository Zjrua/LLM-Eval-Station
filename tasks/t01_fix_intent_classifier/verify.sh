#!/usr/bin/env bash
# 判定：修复后测试全绿（含 produce 用例 fail→pass）
# 回归：无额外文件破坏
set -e
python -m pytest test_classifier.py -q >/dev/null 2>&1 || exit 1
# 回归检查：不允许通过删改测试文件通过（校验测试文件未被改动）
grep -q "test_produce_write" test_classifier.py
grep -q "帮我写" classifier.py
exit 0
