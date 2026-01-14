#!/bin/bash
set -e

# === Generic Training Resume Script ===
# Usage: ./scripts/run_continue_training.sh <CONFIG_FILE> <PREV_RUN_DIR> <NEW_MAX_EPOCHS> [EXTRA_ARGS...]

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <CONFIG_FILE> <PREV_RUN_DIR> <NEW_MAX_EPOCHS> [EXTRA_ARGS...]"
    echo "Example: $0 experiments/configs/mnist.py results/KoopmanSVD/mnist_run_123 200"
    exit 1
fi

CONFIG_FILE=$1
PREV_RUN_DIR=$2
NEW_MAX_EPOCHS=$3
shift 3 # Pass remaining arguments as EXTRA_ARGS

# Load common environment settings (GPU setup, etc.)
source scripts/common.sh

# 1. Validate Paths
CKPT_PATH="${PREV_RUN_DIR}/checkpoints/last.ckpt"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

# 2. Generate New Run ID (Current Timestamp)
# Extract the experiment prefix from the old directory name
OLD_RUN_ID=$(basename "$PREV_RUN_DIR")
# Remove the old timestamp (assuming format _YYYYMMDD-HHMMSS...) to append the new one
PREFIX=$(echo "$OLD_RUN_ID" | sed -E 's/_[0-9]{8}-[0-9]{6}.*//')
NEW_RUN_ID="${PREFIX}_${RUN_ID_BASE}_extended"

echo "========================================================"
echo "   RESUMING TRAINING"
echo "========================================================"
echo " Source:  $PREV_RUN_DIR"
echo " Config:  $CONFIG_FILE"
echo " Target:  $NEW_MAX_EPOCHS Epochs"
echo " New ID:  $NEW_RUN_ID"
echo "--------------------------------------------------------"

# 3. Run Training
# Execute training with the previous checkpoint and updated max_epochs
python experiments/train.py \
    --config="$CONFIG_FILE" \
    --config.trainer.max_epochs=$NEW_MAX_EPOCHS \
    --run_id="$NEW_RUN_ID" \
    --resume_from="$CKPT_PATH" \
    --workdir="${RESULT_DIR}" \
    "$@"
