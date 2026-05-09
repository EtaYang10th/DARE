#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
export PROJECT_ROOT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${SCRIPT_DIR}"

if [[ "${1:-}" == "--check" ]]; then
    python3 - <<'PY'
import importlib.util
import sys

required = ["torch", "ray", "verl"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing packages: " + ", ".join(missing))
    sys.exit(1)
print("Environment check passed.")
PY
    exit 0
fi

cat <<'EOF'
This anonymous launcher is intentionally conservative.

It does not contain cluster accounts, local filesystem paths, W&B URLs, or author identifiers.
Use it as a safe starting point for reviewers or public release users.

Examples:
  bash rl_training/run_anonymous_example.sh --check
  CUDA_VISIBLE_DEVICES=0 bash rl_training/run_anonymous_example.sh --check

For full training, copy this file locally and add site-specific paths outside the public repository.
EOF
