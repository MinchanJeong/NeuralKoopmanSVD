#!/bin/bash
# scripts/common.sh

# Source .env
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a; source .env; set +a
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PYTHONPATH="${PYTHONPATH}:$PWD"

# Defaults
DATA_DIR=${DATA_DIR:-"./data"}
RESULT_DIR=${RESULT_DIR:-"./results"}
PROJECT_NAME="KoopmanSVD"

# Auto-detect NUM_GPUS if not set
if [ -z "${NUM_GPUS}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)
        # Trim whitespace
        NUM_GPUS=$(echo $NUM_GPUS | xargs)
    else
        NUM_GPUS=1
    fi
    echo ">>>> Auto-detected NUM_GPUS=${NUM_GPUS} from CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
fi

# Cleanup
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

# Generate Run ID
RUN_ID_BASE="$(date +"%Y%m%d-%H%M%S")"

# Debug Flag Handler
# Usage: source scripts/common.sh debug
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
