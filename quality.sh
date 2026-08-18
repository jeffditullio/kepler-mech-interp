#!/bin/bash
# Quality gate for kepler-mech-interp: ./quality.sh — all checks must pass.
#
# Deliberately NOT here: pytest (kernels suite is `uv run pytest tests/ -q`,
# run it when touching src/kernels/), pyright (torch/numpy stubs drown it in
# false positives; run via uvx pyright if ever needed), complexity metrics (math kernels are
# legitimately long and branchy).

cd "$(dirname "$0")" || exit 1

SUPPRESSIONS_MAX=5  # inline-suppressions ceiling; prefer pyproject ignore list
FAIL=0

check() {
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    echo "✓ $name"
  else
    echo "✗ $name"
    echo "$out"
    FAIL=1
  fi
}

suppressions_check() {
  local n
  n=$(grep -rE '# noqa|# pyright: ignore|# type: ignore' --include='*.py' src papers 2>/dev/null | wc -l | tr -d ' ')
  echo "$n inline suppressions (ceiling $SUPPRESSIONS_MAX)"
  [ "$n" -le "$SUPPRESSIONS_MAX" ]
}

check "lint"          uv run ruff check .
check "format"        uv run ruff format --check .
check "suppressions"  suppressions_check
exit $FAIL
