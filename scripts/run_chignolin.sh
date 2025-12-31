#!/bin/bash
set -e

# === Source Common Logic (Env, Defaults, Debug) ===
source scripts/common.sh "$@"

# === Experiment Specifics ===
RAW_DIR=${RAW_DIR:-"$DATA_DIR/chignolin_raw/C22star"}
PROCESSED_DIR=${PROCESSED_DIR:-"$DATA_DIR/chignolin_processed/C22star"}
CONFIG_FILE="experiments/configs/chignolin_schnet.py"
RUN_ID="chignolin_${RUN_ID_BASE}"

echo "=== Environment Setup ==="
echo "RAW_DIR:       $RAW_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
echo "Run ID:        $RUN_ID"

echo "=== [1] Preprocessing ==="
# Determine split dir name based on config defaults
SPLIT_NAME="split_80_seed0"
SPLIT_DIR="${PROCESSED_DIR}/${SPLIT_NAME}"

python experiments/run_preprocessing.py \
    --config=$CONFIG_FILE \
    --config.data.raw_path="$RAW_DIR" \
    --config.data.dataset_dir="$PROCESSED_DIR" \
    # --overwrite

echo "=== [2] Training ==="
python experiments/train.py \
    --config=$CONFIG_FILE \
    --config.trainer.devices=$NUM_GPUS \
    --config.project_name=$PROJECT_NAME \
    --config.data.raw_path="$RAW_DIR" \
    --config.data.dataset_dir="$PROCESSED_DIR" \
    --config.data.train_db_path="$SPLIT_DIR/train_full.db" \
    --config.data.val_db_path="$SPLIT_DIR/val_full.db" \
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
