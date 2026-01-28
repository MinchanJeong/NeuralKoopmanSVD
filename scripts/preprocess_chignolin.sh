#!/bin/bash
set -e

# === [1] Load Common Logic ===
source scripts/common.sh "$@"

# === [2] Path Configuration ===
CONFIG_FILE="experiments/configs/chignolin_schnet.py"
RAW_DIR=${RAW_DIR:-"$DATA_DIR/chignolin_raw/C22star"}
PROCESSED_DIR=${PROCESSED_DIR:-"$DATA_DIR/chignolin_processed/C22star"}

DATA_SEED=0

python3 experiments/run_preprocessing.py \
    --config=$CONFIG_FILE \
    --config.data.raw_path="$RAW_DIR" \
    --config.data.dataset_dir="$PROCESSED_DIR" \
    --config.data.seed=$DATA_SEED \
    --overwrite