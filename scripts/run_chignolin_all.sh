#!/bin/bash
set -e

# === [1] Load Common Logic ===
source scripts/common.sh "$@"

# === [2] Configuration ===
CONFIG_FILE="experiments/configs/chignolin_schnet.py"
RAW_DIR=${RAW_DIR:-"$DATA_DIR/chignolin_raw/C22star"}
PROCESSED_DIR=${PROCESSED_DIR:-"$DATA_DIR/chignolin_processed/C22star"}
SPLIT_NAME="split_80_seed0"
SPLIT_DIR="${PROCESSED_DIR}/${SPLIT_NAME}"

echo "========================================================"
echo "   STARTING CHIGNOLIN BENCHMARK SUITE"
echo "   Date: $(date)"
echo "========================================================"

# === [3] Preprocessing (Run Once) ===
echo ">>> [Step 0] Running Preprocessing..."
python experiments/run_preprocessing.py \
    --config=$CONFIG_FILE \
    --config.data.raw_path="$RAW_DIR" \
    --config.data.dataset_dir="$PROCESSED_DIR" # --overwrite

# === [4] Experiment Definitions ===
# Format: Name | LossType | Nesting | ExtraFlags
EXPERIMENTS=(
    "LoRA_Seq|lora|seq|--config.model.centering=true"
    "LoRA_Jnt|lora|jnt|--config.model.centering=true"
    "LoRA_Std|lora|None|--config.model.centering=true"

    "DPNet|dp|None|--config.model.loss.metric_deformation=0.01 --config.model.loss.relaxed=false --config.model.centering=false"
    "DPNet_Relaxed|dp|None|--config.model.loss.metric_deformation=0.01 --config.model.loss.relaxed=true  --config.model.centering=false"
    "DPNet_Relaxed_Cntr|dp|None|--config.model.loss.metric_deformation=0.01 --config.model.loss.relaxed=true --config.model.centering=true"

    "VAMP2|vamp|None|--config.model.loss.schatten_norm=2  --config.model.centering=false"
    "VAMP1|vamp|None|--config.model.loss.schatten_norm=1 --config.model.centering=false"
    "VAMP1_Cntr|vamp|None|--config.model.loss.schatten_norm=1 --config.model.centering=true"
)

# === [5] Execution Loop ===
for exp in "${EXPERIMENTS[@]}"; do
    # Robust Parsing using cut (Delim: |)
    # xargs removes leading/trailing whitespace
    EXP_NAME=$(echo "$exp" | cut -d'|' -f1 | xargs)
    LOSS_TYPE=$(echo "$exp" | cut -d'|' -f2 | xargs)
    NESTING=$(echo "$exp" | cut -d'|' -f3 | xargs)
    EXTRA_FLAGS=$(echo "$exp" | cut -d'|' -f4 | xargs)

    # Safety Check: Skip if parsing failed
    if [[ -z "$EXP_NAME" ]]; then
        echo "WARNING: Skipping invalid experiment line: '$exp'"
        continue
    fi

    # Generate Run ID
    # Combining Experiment Name with the global RUN_ID_BASE (timestamp from common.sh)
    # or generating a new timestamp here if you prefer unique times per run.
    CURRENT_TIME=$(date +"%H%M")
    RUN_ID="chignolin_${EXP_NAME}_${RUN_ID_BASE}_${CURRENT_TIME}"

    # Nesting Logic
    if [[ "$NESTING" == "None" ]]; then
        NESTING_FLAG="--config.model.loss.nesting=None"
    else
        NESTING_FLAG="--config.model.loss.nesting=$NESTING"
    fi

    echo "--------------------------------------------------------"
    echo ">>> Running Experiment: $EXP_NAME"
    echo "    ID: $RUN_ID"
    echo "    Flags: $EXTRA_FLAGS"
    echo "--------------------------------------------------------"

    # 1. Training
    # Note: Passing $EXTRA_FLAGS unquoted to allow shell expansion of multiple flags
    python experiments/train.py \
        --config=$CONFIG_FILE \
        --config.logging.use_wandb=True \
        --config.logging.wandb_project="KoopmanSVD" \
        --config.trainer.devices=$NUM_GPUS \
        --config.project_name=$PROJECT_NAME \
        --config.data.raw_path="$RAW_DIR" \
        --config.data.dataset_dir="$PROCESSED_DIR" \
        --config.data.train_db_path="$SPLIT_DIR/train_full.db" \
        --config.data.val_db_path="$SPLIT_DIR/val_full.db" \
        --config.model.loss.type=$LOSS_TYPE \
        $NESTING_FLAG \
        $EXTRA_FLAGS \
        --workdir="$RESULT_DIR" \
        --run_id="$RUN_ID" \
        $DEBUG_ARGS

    # 2. Analysis
    TARGET_DIR="$RESULT_DIR/$PROJECT_NAME/$RUN_ID"

    if [ -d "$TARGET_DIR" ]; then
        echo ">>> Running Analysis for $RUN_ID..."
        python experiments/run_analysis.py \
            --run_dir="$TARGET_DIR" \
            --output_dir="analysis_results"
    else
        echo "ERROR: Run directory not found. Training likely failed."
        exit 1
    fi

    echo ">>> Completed: $EXP_NAME"
    echo ""
done

echo "========================================================"
echo "   ALL EXPERIMENTS FINISHED SUCCESSFULLY"
echo "========================================================"
