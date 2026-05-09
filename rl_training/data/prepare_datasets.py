"""
Prepare training datasets for RL fine-tuning, matching the paper's setup.

Datasets:
  1. MATH (Level 3-5, train+test merged, excluding MATH500)
  2. DeepScaleR-40K (sample 10240)          -- already exists as deepscaler.parquet
  3. Open-Reasoner-Zero-57K / ORZ (sample 8192)
  4. DeepMath-103K (sample 8192)

Output format (verl parquet):
  - data_source: str
  - prompt: list[dict]  (system + user messages)
  - ability: "math"
  - reward_model: {"style": "rule", "ground_truth": <answer>}
  - extra_info: {"question": <raw_question>, "answer": <answer>, "index": int, "split": "train"}

Usage:
    python prepare_datasets.py --datasets all --merge
    python prepare_datasets.py --datasets math orz deepmath  [--seed 42]
"""

import argparse
import os
import random

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# MATH500 问题列表 (用于去重)
# ---------------------------------------------------------------------------
def load_math500_problems():
    """Load MATH500 benchmark problems to exclude from training."""
    try:
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        return set(p.strip() for p in ds["problem"])
    except Exception as e:
        print(f"[WARN] Could not load MATH-500 for decontamination: {e}")
        print("       Proceeding without MATH500 exclusion.")
        return set()


# ---------------------------------------------------------------------------
# Helper: build a single row in the target schema
# ---------------------------------------------------------------------------
def make_row(question: str, answer: str, index: int, data_source: str):
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "question": question,
            "answer": answer,
            "index": index,
            "split": "train",
        },
    }


# ---------------------------------------------------------------------------
# 1. MATH  (Level 3-5, merged train+test, excluding MATH500)
# ---------------------------------------------------------------------------
def prepare_math(seed: int, output_dir: str):
    out_path = os.path.join(output_dir, "math.parquet")
    if os.path.exists(out_path):
        print(f"\n=== MATH already exists at {out_path}, skipping ===")
        return out_path

    print("\n=== Preparing MATH (Level 3-5) ===")
    from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

    # Load all 7 subject configs and merge
    configs = ['algebra', 'counting_and_probability', 'geometry',
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_splits = []
    for cfg in configs:
        ds = datasets.load_dataset("EleutherAI/hendrycks_math", cfg)
        all_splits.append(ds["train"])
        all_splits.append(ds["test"])
    all_examples = datasets.concatenate_datasets(all_splits)
    print(f"  Total examples (all levels): {len(all_examples)}")

    # filter Level 3-5
    all_examples = all_examples.filter(lambda x: x["level"] in ["Level 3", "Level 4", "Level 5"])
    print(f"  After Level 3-5 filter: {len(all_examples)}")

    # exclude MATH500
    math500 = load_math500_problems()
    if math500:
        before = len(all_examples)
        all_examples = all_examples.filter(lambda x: x["problem"].strip() not in math500)
        print(f"  After MATH500 exclusion: {len(all_examples)} (removed {before - len(all_examples)})")

    rows = []
    skipped = 0
    for idx, ex in enumerate(all_examples):
        question = ex["problem"]
        try:
            answer = remove_boxed(last_boxed_only_string(ex["solution"]))
        except Exception:
            skipped += 1
            continue
        rows.append(make_row(question, answer, idx, "math_level3to5"))

    print(f"  Final count: {len(rows)} (skipped {skipped} unparseable)")
    save_parquet(rows, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2. DeepScaleR  (already exists — skip or regenerate)
# ---------------------------------------------------------------------------
def prepare_deepscaler(seed: int, output_dir: str, n_sample: int = 10240):
    out_path = os.path.join(output_dir, "deepscaler.parquet")
    if os.path.exists(out_path):
        print(f"\n=== DeepScaleR already exists at {out_path}, skipping ===")
        return out_path

    print(f"\n=== Preparing DeepScaleR (sample {n_sample}) ===")
    ds = datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    print(f"  Total examples: {len(ds)}")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n_sample]

    rows = []
    for new_idx, orig_idx in enumerate(indices):
        ex = ds[orig_idx]
        question = ex["problem"]
        answer = ex["answer"]
        rows.append(make_row(question, answer, new_idx, f"deepscaler_{n_sample}"))

    print(f"  Final count: {len(rows)}")
    save_parquet(rows, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Open-Reasoner-Zero-57K  (ORZ, sample 8192)
# ---------------------------------------------------------------------------
def prepare_orz(seed: int, output_dir: str, n_sample: int = 8192):
    out_path = os.path.join(output_dir, "orz.parquet")
    if os.path.exists(out_path):
        print(f"\n=== ORZ already exists at {out_path}, skipping ===")
        return out_path

    print(f"\n=== Preparing ORZ (sample {n_sample}) ===")
    ds = datasets.load_dataset("Tonic/OpenReasonerZero", split="train")
    print(f"  Total examples: {len(ds)}")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n_sample]

    rows = []
    for new_idx, orig_idx in enumerate(indices):
        ex = ds[orig_idx]
        question = ex["0"]["value"]       # human turn
        answer = ex["1"]["ground_truth"]["value"]  # assistant ground truth
        if not question or not answer:
            continue
        rows.append(make_row(question, answer, new_idx, f"orz_{n_sample}"))

    print(f"  Final count: {len(rows)}")
    save_parquet(rows, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 4. DeepMath-103K  (sample 8192)
# ---------------------------------------------------------------------------
def prepare_deepmath(seed: int, output_dir: str, n_sample: int = 8192):
    out_path = os.path.join(output_dir, "deepmath.parquet")
    if os.path.exists(out_path):
        print(f"\n=== DeepMath already exists at {out_path}, skipping ===")
        return out_path

    print(f"\n=== Preparing DeepMath-103K (sample {n_sample}) ===")
    ds = datasets.load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  Total examples: {len(ds)}")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n_sample]

    rows = []
    for new_idx, orig_idx in enumerate(indices):
        ex = ds[orig_idx]
        question = ex["question"]
        answer = ex["final_answer"]
        if not question or not answer:
            continue
        rows.append(make_row(question, answer, new_idx, f"deepmath_{n_sample}"))

    print(f"  Final count: {len(rows)}")
    save_parquet(rows, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Save / merge helpers
# ---------------------------------------------------------------------------
def save_parquet(rows: list[dict], path: str):
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  Saved to {path}  ({len(df)} rows)")


def merge_parquets(input_paths: list[str], output_path: str):
    existing_paths = [path for path in input_paths if path and os.path.exists(path)]
    missing_paths = [path for path in input_paths if path and not os.path.exists(path)]

    if missing_paths:
        print("\n[WARN] Skipping missing parquet files during merge:")
        for path in missing_paths:
            print(f"  - {path}")

    if not existing_paths:
        raise FileNotFoundError("No parquet files found to merge.")

    merged_df = pd.concat((pd.read_parquet(path) for path in existing_paths), ignore_index=True)
    merged_df.to_parquet(output_path, index=False)
    print(f"\n=== Merged {len(existing_paths)} datasets into {output_path} ({len(merged_df)} rows) ===")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RL training datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["math", "orz", "deepmath"],
        choices=["math", "deepscaler", "orz", "deepmath", "all"],
        help="Which datasets to prepare",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same dir as this script)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge the selected parquet files into one combined parquet file")
    parser.add_argument("--merged_filename", type=str, default="merged.parquet",
                        help="Filename for the merged parquet output")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    targets = set(args.datasets)
    if "all" in targets:
        targets = {"math", "deepscaler", "orz", "deepmath"}

    prepared_paths = []
    if "math" in targets:
        prepared_paths.append(prepare_math(args.seed, output_dir))
    if "deepscaler" in targets:
        prepared_paths.append(prepare_deepscaler(args.seed, output_dir))
    if "orz" in targets:
        prepared_paths.append(prepare_orz(args.seed, output_dir))
    if "deepmath" in targets:
        prepared_paths.append(prepare_deepmath(args.seed, output_dir))

    if args.merge:
        merge_output_path = os.path.join(output_dir, args.merged_filename)
        merge_parquets(prepared_paths, merge_output_path)

    print("\n✓ Done!")
