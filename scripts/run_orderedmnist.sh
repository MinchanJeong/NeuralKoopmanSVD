#!/bin/bash
set -e

# === Source Common Logic ===
source scripts/common.sh "$@"

# === Experiment Specifics ===
CONFIG_FILE="experiments/configs/mnist_cnn.py"
DATA_FILE="$DATA_DIR/ordered_mnist.npy"
RUN_ID="mnist_${RUN_ID_BASE}"

echo "=== Environment Setup ==="
echo "DATA_FILE: $DATA_FILE"
echo "Run ID:    $RUN_ID"

echo "=== [1] Preprocessing ==="
python experiments/run_preprocessing.py \
    --config=$CONFIG_FILE \
    --config.data.path="$DATA_FILE" \
    # --overwrite

echo "=== [2] Training ==="
python experiments/train.py \
    --config=$CONFIG_FILE \
    --config.project_name=$PROJECT_NAME \
    --config.data.path="$DATA_FILE" \
    --workdir="$RESULT_DIR" \
    --run_id="$RUN_ID" \
    $DEBUG_ARGS

echo "=== [3] Analysis ==="
TARGET_DIR="$RESULT_DIR/$PROJECT_NAME/$RUN_ID"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Run directory not found: $TARGET_DIR"
    exit 1
fi

python experiments/run_analysis.py \
    --run_dir="$TARGET_DIR" \
    --output_dir="analysis_results"

echo "=== Done! ==="
