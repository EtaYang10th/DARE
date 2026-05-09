#!/bin/bash
# =============================================================================
# Train the coding teacher model (adaptive difficulty predictor).
# Mirrors run_train.sh but points to coding datasets/embeddings.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TEACHER_ROOT=$(realpath "${SCRIPT_DIR}/..")

source ~/.bashrc
conda activate rl_test

export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"

DATASET_DIR="${TEACHER_ROOT}/datasets/coding"
DATA_TRAIN_PATH="${DATASET_DIR}/data_coding_train.pkl"
DATA_REF_PATH="${DATASET_DIR}/data_coding_ref.pkl"
EMBEDDINGS_DIR="${DATASET_DIR}/embeddings"

REF_SIZE=256
LOSS_TYPE="binary_cross_entropy"
NUM_LAYERS=3
SCALING="group_logit_temp"
BATCH_SIZE=256
EPOCHS=20
LR=1e-3

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
cd "$TEACHER_ROOT"

accelerate launch \
  --num_processes 1 \
  train.py \
  --loss_type "$LOSS_TYPE" \
  --model_name "$EMBEDDING_MODEL" \
  --batch_size_per_gpu "$BATCH_SIZE" \
  --ref_size "$REF_SIZE" \
  --data_path "$DATASET_DIR" \
  --use_embeddings \
  --epochs "$EPOCHS" \
  --seed 1 \
  --lr "$LR" \
  --num_layers "$NUM_LAYERS" \
  --use_scheduler \
  --save_predictions \
  --use_layernorm \
  --output_dir "outputs_coding" \
  --scaling "$SCALING" \
  --data_train_path "$DATA_TRAIN_PATH" \
  --data_ref_path "$DATA_REF_PATH" \
  --left_padding \
  --method residual
