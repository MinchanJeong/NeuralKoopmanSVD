#!/bin/bash
set -e

# === [1] Load Common Logic ===
source scripts/common.sh "$@"

# === [2] Path Configuration ===
CONFIG_FILE="experiments/configs/chignolin_schnet.py"
RAW_DIR=${RAW_DIR:-"$DATA_DIR/chignolin_raw/C22star"}
PROCESSED_DIR=${PROCESSED_DIR:-"$DATA_DIR/chignolin_processed/C22star"}

# !!! IMPORTANT: Check metadata.json to find the exact seed used !!!
# If your preprocessing found a different seed, update this!
EFFECTIVE_DATA_SEED=0
SPLIT_NAME="split_80_seed${EFFECTIVE_DATA_SEED}"
SPLIT_DIR="${PROCESSED_DIR}/${SPLIT_NAME}"

# Sanity Check
if [ ! -f "$SPLIT_DIR/train_full.db" ]; then
    echo "ERROR: Database not found at $SPLIT_DIR"
    echo "Please run 'scripts/preprocess_chignolin.sh' first or check the seed."
    exit 1
fi

echo "========================================================"
echo "   RUNNING CHIGNOLIN EXPERIMENTS"
echo "   Split: $SPLIT_NAME"
echo "========================================================"

# === [3] Experiment List (Simplified for Demo) ===
# Format: Name | LossType | Nesting | ExtraFlags
EXPERIMENTS=(
    "LoRA_Seq|lora|seq|--config.model.centering=true"
    #"LoRA_Jnt|lora|jnt|--config.model.centering=true"
    #"LoRA_Std|lora|None|--config.model.centering=true"
    #"DPNet_Relaxed|dp|None|--config.model.loss.metric_deformation=0.01 --config.model.loss.relaxed=true --config.model.centering=true"
    # "DPNet|dp|None|--config.model.centering=true"
    # "VAMP1|vamp|None|--config.model.loss.schatten_norm=1 --config.model.centering=true"
    # "VAMP2|vamp|None|--config.model.loss.schatten_norm=2 --config.model.centering=true"
)

# === [4] Execution Loop ===
for exp in "${EXPERIMENTS[@]}"; do
    EXP_NAME=$(echo "$exp" | cut -d'|' -f1 | xargs)
    LOSS_TYPE=$(echo "$exp" | cut -d'|' -f2 | xargs)
    NESTING=$(echo "$exp" | cut -d'|' -f3 | xargs)
    EXTRA_FLAGS=$(echo "$exp" | cut -d'|' -f4 | xargs)

    # Unique Run ID
    RUN_ID="chignolin_${EXP_NAME}_${RUN_ID_BASE}"

    # Nesting Flag Helper
    if [[ "$NESTING" == "None" ]]; then
        NESTING_FLAG="--config.model.loss.nesting=None"
    else
        NESTING_FLAG="--config.model.loss.nesting=$NESTING"
    fi

    echo ">>> Starting: $EXP_NAME (ID: $RUN_ID)"

    # 1. Training
    python experiments/train.py \
        --config=$CONFIG_FILE \
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
        python experiments/run_analysis.py \
            --run_dir="$TARGET_DIR" \
            --output_dir="analysis_results"
    fi

    echo ">>> Finished: $EXP_NAME"
    echo "--------------------------------------------------------"
done
