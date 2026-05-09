"""
Step 1: Prepare coding question pool from TACO + APPS.

Loads both datasets from HuggingFace, samples a balanced subset by difficulty,
validates test cases, and saves intermediate JSON + RL-format parquet.

Usage:
    python prepare_dataset.py --output_dir ../datasets/coding --total 10240
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.set_int_max_str_digits(100000)

import pandas as pd


def load_taco(target_count: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet",
        split="train",
    )
    print(f"[TACO] loaded {len(ds)} raw problems")

    difficulty_levels = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]
    by_diff: dict[str, list[dict]] = {d: [] for d in difficulty_levels}

    for row in ds:
        diff = row.get("difficulty", "").strip()
        if diff not in by_diff:
            continue
        try:
            io = json.loads(row["input_output"]) if isinstance(row["input_output"], str) else row["input_output"]
        except (json.JSONDecodeError, TypeError):
            continue
        if not _valid_io(io):
            continue
        by_diff[diff].append(_normalize_row(row, io, source="taco"))

    for d in difficulty_levels:
        print(f"  [TACO] {d}: {len(by_diff[d])} valid problems")

    per_level = target_count // len(difficulty_levels)
    remainder = target_count - per_level * len(difficulty_levels)
    rng = random.Random(seed)

    selected: list[dict] = []
    shortfall = 0
    for i, d in enumerate(difficulty_levels):
        need = per_level + (1 if i < remainder else 0)
        pool = by_diff[d]
        rng.shuffle(pool)
        take = min(need, len(pool))
        selected.extend(pool[:take])
        shortfall += need - take

    if shortfall > 0:
        remaining = []
        selected_ids = {r["id"] for r in selected}
        for d in difficulty_levels:
            remaining.extend(r for r in by_diff[d] if r["id"] not in selected_ids)
        rng.shuffle(remaining)
        selected.extend(remaining[:shortfall])
        if shortfall > len(remaining):
            print(f"  [TACO] WARNING: could only get {len(selected)} / {target_count}")

    print(f"[TACO] selected {len(selected)} problems")
    return selected


def load_apps(target_count: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset

    results: list[dict] = []
    for split in ("train", "test"):
        ds = load_dataset(
            "json",
            data_files=f"hf://datasets/codeparrot/apps/{split}.jsonl",
            split="train",
        )
        print(f"[APPS] loaded {len(ds)} from split={split}")
        for row in ds:
            try:
                io_str = row.get("input_output", "")
                if not io_str or io_str == "":
                    continue
                io = json.loads(io_str) if isinstance(io_str, str) else io_str
            except (json.JSONDecodeError, TypeError):
                continue
            if not _valid_io(io):
                continue
            results.append(_normalize_row(row, io, source="apps"))

    difficulty_levels = ["introductory", "interview", "competition"]
    by_diff: dict[str, list[dict]] = {d: [] for d in difficulty_levels}
    for r in results:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    for d in difficulty_levels:
        print(f"  [APPS] {d}: {len(by_diff[d])} valid problems")

    per_level = target_count // len(difficulty_levels)
    remainder = target_count - per_level * len(difficulty_levels)
    rng = random.Random(seed)

    selected: list[dict] = []
    shortfall = 0
    for i, d in enumerate(difficulty_levels):
        need = per_level + (1 if i < remainder else 0)
        pool = by_diff[d]
        rng.shuffle(pool)
        take = min(need, len(pool))
        selected.extend(pool[:take])
        shortfall += need - take

    if shortfall > 0:
        remaining = []
        selected_ids = {r["id"] for r in selected}
        for d in difficulty_levels:
            remaining.extend(r for r in by_diff[d] if r["id"] not in selected_ids)
        rng.shuffle(remaining)
        selected.extend(remaining[:shortfall])

    print(f"[APPS] selected {len(selected)} problems")
    return selected


def _valid_io(io: dict) -> bool:
    """Check that input_output dict has at least one usable test case."""
    if "fn_name" in io and io["fn_name"]:
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        return len(inputs) > 0 and len(inputs) == len(outputs)

    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    if not inputs or not outputs or len(inputs) != len(outputs):
        return False
    for inp, out in zip(inputs, outputs):
        if not isinstance(inp, str) or not isinstance(out, str):
            return False
    return True


def _normalize_row(row: dict, io: dict, source: str) -> dict:
    """Normalize a dataset row into a unified format."""
    fn_name = io.get("fn_name", None) or row.get("fn_name", None)
    is_fn_call = bool(fn_name)

    if source == "taco":
        qid = f"taco_{row.get('name', '')}_{hash(row['question']) % 10**8}"
        difficulty = row.get("difficulty", "UNKNOWN")
    else:
        qid = f"apps_{row.get('problem_id', hash(row['question']) % 10**8)}"
        difficulty = row.get("difficulty", "unknown")

    return {
        "id": qid,
        "source": source,
        "question": row["question"],
        "difficulty": difficulty,
        "input_output": io,
        "starter_code": row.get("starter_code", "") or "",
        "fn_name": fn_name,
        "is_function_call": is_fn_call,
    }


def build_rl_parquet(questions: list[dict], output_path: str):
    """Build RL-training-format parquet (same schema as deepscaler.parquet)."""
    import numpy as np

    SYSTEM_PROMPT = (
        "You are a helpful assistant that solves programming problems. "
        "Think step by step, then provide your solution as a single Python code block. "
        "Your code should read from standard input and print to standard output."
    )

    rows = []
    for i, q in enumerate(questions):
        prompt_text = q["question"]
        if q["starter_code"]:
            prompt_text += f"\n\nStarter code:\n```python\n{q['starter_code']}\n```"

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        extra_info = {
            "index": i,
            "question": q["question"],
            "id": q["id"],
            "source": q["source"],
            "difficulty": q["difficulty"],
            "input_output": json.dumps(q["input_output"]),
            "starter_code": q["starter_code"],
            "fn_name": q["fn_name"] or "",
            "is_function_call": q["is_function_call"],
        }
        rows.append({
            "data_source": q["source"],
            "prompt": np.array(prompt, dtype=object),
            "ability": "coding",
            "reward_model": "code_execution",
            "extra_info": extra_info,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"[RL parquet] saved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare coding dataset from TACO + APPS")
    parser.add_argument("--output_dir", type=str, default="../datasets/coding")
    parser.add_argument("--total", type=int, default=10240)
    parser.add_argument("--taco_ratio", type=float, default=0.5,
                        help="Fraction of total from TACO (rest from APPS)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_rl_parquet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    oversample = 1.2
    n_taco = int(args.total * args.taco_ratio * oversample)
    n_apps = int(args.total * (1 - args.taco_ratio) * oversample)

    taco_questions = load_taco(n_taco, seed=args.seed)
    apps_questions = load_apps(n_apps, seed=args.seed + 1)

    all_questions = taco_questions + apps_questions
    random.Random(args.seed + 2).shuffle(all_questions)

    seen = set()
    deduped = []
    for q in all_questions:
        key = q["question"].strip()[:500]
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    if len(deduped) < len(all_questions):
        print(f"[DEDUP] removed {len(all_questions) - len(deduped)} duplicates")
        all_questions = deduped

    if len(all_questions) > args.total:
        all_questions = all_questions[:args.total]
        print(f"[TRIM] trimmed to {args.total} questions")
    elif len(all_questions) < args.total:
        print(f"[WARNING] only {len(all_questions)} unique questions available "
              f"(target was {args.total})")

    for i, q in enumerate(all_questions):
        q["global_index"] = i

    print(f"\n=== Final dataset: {len(all_questions)} questions ===")
    src_counts = Counter(q["source"] for q in all_questions)
    diff_counts = Counter(q["difficulty"] for q in all_questions)
    print(f"  By source: {dict(src_counts)}")
    print(f"  By difficulty: {dict(diff_counts)}")

    questions_path = os.path.join(args.output_dir, "coding_questions.json")
    with open(questions_path, "w") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    print(f"Saved questions to {questions_path}")

    parquet_path = os.path.join(args.output_dir, "questions_coding_10240.parquet")
    df = pd.DataFrame({"problem": [q["question"] for q in all_questions]})
    df.to_parquet(parquet_path, index=False)
    print(f"Saved questions parquet to {parquet_path}")

    if not args.skip_rl_parquet:
        rl_path = os.path.join(args.output_dir, "coding.parquet")
        build_rl_parquet(all_questions, rl_path)

    print("\n[DONE] prepare_dataset complete.")


if __name__ == "__main__":
    main()
