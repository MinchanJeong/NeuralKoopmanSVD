#!/bin/bash
# scripts/common.sh

# 1. Source .env
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a; source .env; set +a
fi

# 2. Python Path
export PYTHONPATH="${PYTHONPATH}:$PWD"

# 3. Defaults
DATA_DIR=${DATA_DIR:-"./data"}
RESULT_DIR=${RESULT_DIR:-"./results"}
PROJECT_NAME="KoopmanSVD"

# 4. Cleanup
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

# 5. Generate Run ID
RUN_ID_BASE="$(date +"%Y%m%d-%H%M%S")"

# 6. Debug Flag Handler
# Usage: source scripts/common.sh [debug]
DEBUG_ARGS=""
if [[ "$1" == "debug" ]]; then
    echo ">>>>  DEBUG MODE ENABLED <<<<"
    echo "  - max_epochs: 1"
    echo "  - check_val_every_n_epoch: 1"
    DEBUG_ARGS="--config.trainer.max_epochs=1 --config.trainer.check_val_every_n_epoch=1"
    # Append suffix to Run ID to distinguish debug runs
    RUN_ID_BASE="${RUN_ID_BASE}_debug"
fi

echo "RUN_ID_BASE: $RUN_ID"
