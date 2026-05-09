#!/bin/bash
# =============================================================================
# Full pipeline: prepare coding dataset -> generate rollouts -> build teacher data
#
# Usage:
#   bash run_pipeline.sh [step]
#   Steps: prepare | generate | evaluate | build | embed | all
#   Default: all
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TEACHER_ROOT=$(realpath "${SCRIPT_DIR}/..")

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
DATASET_DIR="${TEACHER_ROOT}/datasets/coding"
ROLLOUTS_DIR="${DATASET_DIR}/rollouts"
EMBEDDINGS_DIR="${DATASET_DIR}/embeddings"
QUESTIONS_JSON="${DATASET_DIR}/coding_questions.json"

TOTAL_QUESTIONS=10240
TACO_RATIO=0.5
SEED=42

EMBEDDING_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
REF_SIZE=2048

# ---------------------------------------------------------------------------
# Server detection & conda
# ---------------------------------------------------------------------------
if [[ "$SCRIPT_DIR" == *yz1403* ]]; then
    source ~/.bashrc
    conda activate rl_test
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [[ "$SCRIPT_DIR" == *cjin1* ]]; then
    source ~/.bashrc
    conda activate rl_test
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

STEP="${1:-all}"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Prepare dataset
# ---------------------------------------------------------------------------
# run_prepare() {
#     echo "============================================"
#     echo "Step 1: Preparing coding dataset"
#     echo "============================================"
#     python prepare_dataset.py \
#         --output_dir "$DATASET_DIR" \
#         --total "$TOTAL_QUESTIONS" \
#         --taco_ratio "$TACO_RATIO" \
#         --seed "$SEED"
# }

# ---------------------------------------------------------------------------
# Step 2: Generate rollouts (per model, multi-GPU tensor parallel)
# ---------------------------------------------------------------------------
run_generate() {
    echo "============================================"
    echo "Step 2: Generating rollouts (via _run_generate_all.sh)"
    echo "============================================"
    bash "${SCRIPT_DIR}/_run_generate_all.sh"
}

# ---------------------------------------------------------------------------
# Step 3: Build teacher data (pkl + parquet)
# ---------------------------------------------------------------------------
run_build() {
    echo "============================================"
    echo "Step 3: Building teacher data"
    echo "============================================"
    python build_teacher_data.py \
        --questions_path "$QUESTIONS_JSON" \
        --rollouts_dir "$ROLLOUTS_DIR" \
        --output_dir "$DATASET_DIR" \
        --ref_size "$REF_SIZE" \
        --seed "$SEED"
}

# ---------------------------------------------------------------------------
# Step 4: Generate embeddings
# ---------------------------------------------------------------------------
run_embed() {
    echo "============================================"
    echo "Step 4: Generating embeddings"
    echo "============================================"
    python save_coding_embeddings.py \
        --model_name "$EMBEDDING_MODEL" \
        --questions_json "$QUESTIONS_JSON" \
        --output_dir "$EMBEDDINGS_DIR" \
        --batch_size 32
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$STEP" in
    prepare)    run_prepare ;;
    generate)   run_generate ;;
    build)      run_build ;;
    embed)      run_embed ;;
    all)
        run_prepare
        run_generate
        run_build
        run_embed
        echo ""
        echo "============================================"
        echo "Pipeline complete!"
        echo "  Questions: ${QUESTIONS_JSON}"
        echo "  Teacher data: ${DATASET_DIR}/data_coding_{train,ref}.pkl"
        echo "  Embeddings: ${EMBEDDINGS_DIR}/"
        echo "============================================"
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: $0 {prepare|generate|build|embed|all}"
        exit 1
        ;;
esac
