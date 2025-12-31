#!/bin/bash
# Script to run unit tests and dry-run integration tests for the KoopmanSVD project.
# Usage: ./scripts/run_tests.sh

set -e

# === Configuration ===
if [ -f .env ]; then
    set -a; source .env; set +a
fi
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "=== 1. Running Unit Tests ==="
pytest tests/

echo ""
echo "=== 2. Running Dry-Run Integration Tests (Debug Mode) ==="

run_test() {
    TEST_NAME=$1
    SCRIPT_PATH=$2

    echo "------------------------------------------------------------"
    echo "STARTING > ${TEST_NAME}..."
    echo "CMD      > bash ${SCRIPT_PATH} debug"
    echo "------------------------------------------------------------"

    if bash "$SCRIPT_PATH" debug; then
        echo ""
        echo ">>> [PASS] ${TEST_NAME}"
    else
        echo ""
        echo ">>> [FAIL] ${TEST_NAME}"
        echo "!!! Test Failed. Stopping execution. !!!"
        exit 1
    fi
    echo ""
}

# Run Integration Tests
run_test "Logistic" "./scripts/run_logistic.sh"
run_test "Ordered MNIST" "./scripts/run_orderedmnist.sh"
run_test "Molecular Dynamics" "./scripts/run_chignolin.sh"

echo "================================="
echo "=== All Tests Passed! ==="
echo "================================="
