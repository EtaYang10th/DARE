#!/bin/bash
# ==========================================================================
# Generate rollouts for all teacher models, one at a time.
# Each model uses maximum tensor-parallel across available GPUs.
#
# Usage:
#   bash _run_generate_all.sh            # run all models
#   bash _run_generate_all.sh 3          # start from model index 3 (0-based)
# ==========================================================================
set -euo pipefail
source ~/.bashrc
conda activate rl_test

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

DATASET_DIR="../datasets/coding"
ROLLOUTS_DIR="${DATASET_DIR}/rollouts"
QUESTIONS_JSON="${DATASET_DIR}/coding_questions.json"
LOG_DIR="${DATASET_DIR}/logs"
mkdir -p "$ROLLOUTS_DIR" "$LOG_DIR"

ALL_GPUS="0,1,2,3,4,5,6,7"

# ── model list: "name:tensor_parallel_size" ──
# tp is determined by num_attention_heads divisibility (max that fits 8 GPUs)
#   Qwen-7B / Coder-7B:  28 heads → max tp=4
#   Qwen-3B:             16 heads → max tp=8
#   Qwen-1.5B / Coder-1.5B: 12 heads → max tp=4
#   deepseek-6.7B:       32 heads → max tp=8
#   deepseek-1.3B:       16 heads → max tp=8
MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct:4"
    "deepseek-ai/deepseek-coder-6.7b-instruct:8"
    "Qwen/Qwen2.5-7B-Instruct:4"
    "Qwen/Qwen2.5-3B-Instruct:8"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct:4"
    "Qwen/Qwen2.5-1.5B-Instruct:4"
    "deepseek-ai/deepseek-coder-1.3b-instruct:8"
)

N_ROLLOUTS=8
MAX_TOKENS=4096
TEMPERATURE=0.6
GPU_MEM_UTIL=0.85
TIMEOUT_PER_CASE=20
MAX_TEST_CASES=50

START_IDX="${1:-0}"

echo "=========================================="
echo "  Coding rollout generation"
echo "  Models: ${#MODELS[@]}, starting from index ${START_IDX}"
echo "  GPUs: ${ALL_GPUS}"
echo "  Questions: ${QUESTIONS_JSON}"
echo "  Start time: $(date)"
echo "=========================================="

for i in "${!MODELS[@]}"; do
    (( i < START_IDX )) && continue

    IFS=':' read -r MODEL TP <<< "${MODELS[$i]}"
    SAFE_NAME="${MODEL//\//_}"
    LOG_FILE="${LOG_DIR}/${SAFE_NAME}.log"
    REWARDS_FILE="${ROLLOUTS_DIR}/rewards_${SAFE_NAME}.json"

    if [[ -f "$REWARDS_FILE" ]]; then
        echo "[${i}/${#MODELS[@]}] SKIP ${MODEL} (rewards file exists)"
        continue
    fi

    # pick first $TP GPUs from ALL_GPUS
    GPUS=$(echo "$ALL_GPUS" | tr ',' '\n' | head -n "$TP" | paste -sd,)

    echo ""
    echo "=========================================="
    echo "  [${i}/${#MODELS[@]}] ${MODEL}"
    echo "  tp=${TP}  GPUs=${GPUS}"
    echo "  $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="$GPUS" python generate_rollouts.py \
        --questions_path "$QUESTIONS_JSON" \
        --output_dir "$ROLLOUTS_DIR" \
        --model "$MODEL" \
        --n_rollouts "$N_ROLLOUTS" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --tensor_parallel_size "$TP" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --timeout_per_case "$TIMEOUT_PER_CASE" \
        --max_test_cases "$MAX_TEST_CASES" \
        --resume \
        2>&1 | tee "$LOG_FILE"

    echo "[${i}/${#MODELS[@]}] DONE ${MODEL}  $(date)"
done

echo ""
echo "=========================================="
echo "  ALL MODELS COMPLETE  $(date)"
echo "=========================================="
