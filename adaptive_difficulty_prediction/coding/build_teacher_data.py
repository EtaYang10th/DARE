"""
Step 3: Build teacher training data (pickle + parquet) from evaluated rollouts.

Reads reward files from multiple models, splits into train/ref, and packages
into the format expected by adaptive_difficulty_prediction/train.py.

Usage:
    python build_teacher_data.py \
        --questions_path ../datasets/coding/coding_questions.json \
        --rollouts_dir ../datasets/coding/rollouts \
        --output_dir ../datasets/coding \
        --ref_size 2048
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter

sys.set_int_max_str_digits(100000)

import pandas as pd


def load_reward_files(rollouts_dir: str) -> dict[str, dict]:
    """Load all rewards_*.json files from the rollouts directory."""
    models = {}
    for fname in sorted(os.listdir(rollouts_dir)):
        if fname.startswith("rewards_") and fname.endswith(".json"):
            fpath = os.path.join(rollouts_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            model_name = data["model"]
            models[model_name] = data["results"]
            n_qs = len(data["results"])
            avg_sr = sum(v["success_rate"] for v in data["results"].values()) / max(n_qs, 1)
            n_signal = sum(1 for v in data["results"].values() if 0 < v["success_rate"] < 1)
            print(f"  Loaded {model_name}: {n_qs} questions, "
                  f"avg_sr={avg_sr:.4f}, signal={n_signal}")
    return models


def build_teacher_pkl(
    questions: list[dict],
    model_results: dict[str, dict],
    ref_size: int = 2048,
    seed: int = 42,
    output_dir: str = ".",
):
    """
    Build data_train.pkl and data_ref.pkl in the format expected by train.py.

    Format:
        {model_name: {"questions": [str, ...], "rewards": [float, ...]}}
    where rewards[i] = success_rate for question[i] under model_name.
    """
    q_ids_ordered = [q["id"] for q in questions]
    q_text_map = {q["id"]: q["question"] for q in questions}

    common_ids = set(q_ids_ordered)
    for model_name, results in model_results.items():
        common_ids &= set(results.keys())
    common_ids = sorted(common_ids, key=lambda x: q_ids_ordered.index(x)
                        if x in q_ids_ordered else float("inf"))
    print(f"\nQuestions with results from ALL models: {len(common_ids)}")

    if len(common_ids) < ref_size + 100:
        print(f"WARNING: only {len(common_ids)} common questions, "
              f"need at least {ref_size} for ref set + training set")

    rng = random.Random(seed)
    shuffled = list(common_ids)
    rng.shuffle(shuffled)

    ref_ids = set(shuffled[:ref_size])
    train_ids = [qid for qid in shuffled[ref_size:]]

    print(f"Split: {len(train_ids)} train, {len(ref_ids)} ref")

    ref_questions = [q_text_map[qid] for qid in shuffled[:ref_size]]
    train_questions = [q_text_map[qid] for qid in train_ids]

    data_train = {}
    data_ref = {}

    for model_name, results in model_results.items():
        safe_model = model_name.split("/")[-1]

        train_rewards = [results[qid]["success_rate"] for qid in train_ids]
        data_train[safe_model] = {
            "questions": train_questions,
            "rewards": train_rewards,
        }

        ref_rewards = [results[qid]["success_rate"] for qid in shuffled[:ref_size]]
        data_ref[safe_model] = {
            "questions": ref_questions,
            "rewards": ref_rewards,
        }

    train_path = os.path.join(output_dir, "data_coding_train.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(data_train, f)
    print(f"Saved train pkl: {train_path}")

    ref_path = os.path.join(output_dir, "data_coding_ref.pkl")
    with open(ref_path, "wb") as f:
        pickle.dump(data_ref, f)
    print(f"Saved ref pkl: {ref_path}")

    _verify_pkl(train_path, ref_path)

    return data_train, data_ref


def _verify_pkl(train_path: str, ref_path: str):
    """Verify the generated pickle files match expected format."""
    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    print("\n=== Verification ===")
    print(f"Train models: {list(train.keys())}")
    print(f"Ref models: {list(ref.keys())}")

    ref_q_lists = [v["questions"] for v in ref.values()]
    for i in range(1, len(ref_q_lists)):
        assert ref_q_lists[0] == ref_q_lists[i], \
            "Ref questions must be identical across all models!"

    for model_name in train:
        n_train = len(train[model_name]["questions"])
        n_ref = len(ref[model_name]["questions"])
        rewards = train[model_name]["rewards"]
        avg_sr = sum(rewards) / len(rewards)
        print(f"  {model_name}: train={n_train}, ref={n_ref}, "
              f"train_avg_sr={avg_sr:.4f}")
    print("Verification passed.\n")


def build_questions_parquet(
    questions: list[dict],
    model_results: dict[str, dict],
    output_dir: str,
):
    """Build questions parquet (for embedding alignment in teacher_utils.py)."""
    common_ids = set(questions[0]["id"] for questions in [questions])
    for results in model_results.values():
        common_ids &= set(results.keys())

    q_texts = []
    for q in questions:
        if q["id"] in common_ids:
            q_texts.append(q["question"])

    df = pd.DataFrame({"problem": q_texts})
    path = os.path.join(output_dir, "questions_coding_10240.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved questions parquet: {path} ({len(df)} rows)")


def print_statistics(model_results: dict[str, dict]):
    """Print detailed statistics about the collected data."""
    print("\n=== Dataset Statistics ===")
    for model_name, results in model_results.items():
        srs = [v["success_rate"] for v in results.values()]
        bins = {"sr=0": 0, "0<sr<0.5": 0, "sr=0.5": 0, "0.5<sr<1": 0, "sr=1": 0}
        for sr in srs:
            if sr == 0:
                bins["sr=0"] += 1
            elif sr < 0.5:
                bins["0<sr<0.5"] += 1
            elif sr == 0.5:
                bins["sr=0.5"] += 1
            elif sr < 1:
                bins["0.5<sr<1"] += 1
            else:
                bins["sr=1"] += 1
        avg = sum(srs) / len(srs) if srs else 0
        print(f"\n  {model_name} (n={len(srs)}, avg_sr={avg:.4f}):")
        for k, v in bins.items():
            print(f"    {k}: {v} ({100*v/max(len(srs),1):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build teacher training data")
    parser.add_argument("--questions_path", type=str, required=True)
    parser.add_argument("--rollouts_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ref_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.questions_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    model_results = load_reward_files(args.rollouts_dir)
    if not model_results:
        print("ERROR: No reward files found. Run generate_rollouts.py first.")
        return

    print_statistics(model_results)

    build_teacher_pkl(
        questions, model_results,
        ref_size=args.ref_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    build_questions_parquet(questions, model_results, args.output_dir)

    print("[DONE] build_teacher_data complete.")


if __name__ == "__main__":
    main()
