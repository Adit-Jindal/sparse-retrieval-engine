#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "             LOCAL IR DEVELOPMENT RUN"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Course tests
# ------------------------------------------------------------

echo "[1/4] Running course tests..."
echo

python -m pytest tests -q
COURSE_STATUS=$?

if [ "$COURSE_STATUS" -ne 0 ]; then
    echo
    echo "ERROR: course tests failed."
    echo "Fix correctness issues before benchmarking retrieval."
    exit "$COURSE_STATUS"
fi

echo
echo "Course tests: PASS"
echo

# ------------------------------------------------------------
# 2. Our tests
# ------------------------------------------------------------

echo "[2/4] Running development tests..."
echo

python -m pytest dev/tests -q
DEV_STATUS=$?

if [ "$DEV_STATUS" -ne 0 ]; then
    echo
    echo "ERROR: development tests failed."
    exit "$DEV_STATUS"
fi

echo
echo "Development tests: PASS"
echo

# ------------------------------------------------------------
# 3. Run the authoritative course harness
# ------------------------------------------------------------

echo "[3/4] Running retrieval benchmark..."
echo

python -m dev.benchmark \
    --experiment "${IR_EXPERIMENT:-manual}"

BENCH_STATUS=$?

if [ "$BENCH_STATUS" -ne 0 ]; then
    echo
    echo "ERROR: benchmark failed."
    exit "$BENCH_STATUS"
fi

# ------------------------------------------------------------
# 4. Compare against accepted best
# ------------------------------------------------------------

echo
echo "[4/4] Comparing against accepted best..."
echo

python -m dev.compare

COMPARE_STATUS=$?

if [ "$COMPARE_STATUS" -ne 0 ]; then
    exit "$COMPARE_STATUS"
fi

echo
echo "============================================================"
echo "                    DEVELOPMENT RUN DONE"
echo "============================================================"