"""
Evaluation module for RL training.

Evaluates the current policy on MATH-500, AIME2024, AIME2025, GSM8K,
and AIMO-AMC (AMC12 validation) datasets by reusing the already-loaded
rollout workers, so no extra GPU memory or separate processes are needed.

Usage:
    Called from ray_trainer.py at the end of each epoch.
"""

import os
import csv
import random
import importlib.util

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import datasets

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# ---------------------------------------------------------------------------
# Math reward scorer (loaded lazily to avoid import-time side effects)
# ---------------------------------------------------------------------------
_COMPUTE_SCORE_FN = None


def _get_compute_score():
    global _COMPUTE_SCORE_FN
    if _COMPUTE_SCORE_FN is None:
        reward_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "verl", "verl", "utils", "reward_score", "math.py",
        )
        spec = importlib.util.spec_from_file_location("_math_reward", reward_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _COMPUTE_SCORE_FN = mod.compute_score
    return _COMPUTE_SCORE_FN


# ---------------------------------------------------------------------------
# Prompt template (must match the one used during RL training)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a mathematical problem solver. "
    "Reason step by step. "
    "At the end, output ONLY the final answer in the form \\boxed{...} "
    "and nothing else."
)

EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0

# ---------------------------------------------------------------------------
# AIME dataset configuration (referenced from test/budget_eval_aime.py)
# ---------------------------------------------------------------------------
AIME_DATASETS = {
    "AIME2024": {
        "name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_field": "problem",
        "answer_field": "answer",
    },
    "AIME2025-I": {
        "name": "opencompass/AIME2025",
        "config": "AIME2025-I",
        "split": "test",
        "problem_field": "question",
        "answer_field": "answer",
    },
    "AIME2025-II": {
        "name": "opencompass/AIME2025",
        "config": "AIME2025-II",
        "split": "test",
        "problem_field": "question",
        "answer_field": "answer",
    },
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
_MATH500_CACHE = None
_AIME2024_CACHE = None
_AIME2025_CACHE = None
_GSM8K_CACHE = None
_AIMO_AMC_CACHE = None


def load_math500():
    """Load MATH-500 dataset (cached after first call)."""
    global _MATH500_CACHE
    if _MATH500_CACHE is None:
        _MATH500_CACHE = datasets.load_dataset(
            "HuggingFaceH4/MATH-500", split="test"
        )
    return _MATH500_CACHE


def load_aime2024():
    """Load AIME 2024 dataset (cached after first call)."""
    global _AIME2024_CACHE
    if _AIME2024_CACHE is None:
        cfg = AIME_DATASETS["AIME2024"]
        ds = datasets.load_dataset(cfg["name"], split=cfg["split"])
        problems = [ds[i][cfg["problem_field"]] for i in range(len(ds))]
        answers = [str(ds[i][cfg["answer_field"]]) for i in range(len(ds))]
        _AIME2024_CACHE = (problems, answers)
    return _AIME2024_CACHE


def load_aime2025():
    """Load AIME 2025 dataset (AIME2025-I + AIME2025-II, cached)."""
    global _AIME2025_CACHE
    if _AIME2025_CACHE is None:
        problems, answers = [], []
        for key in ["AIME2025-I", "AIME2025-II"]:
            cfg = AIME_DATASETS[key]
            ds = datasets.load_dataset(
                cfg["name"], name=cfg["config"], split=cfg["split"]
            )
            for i in range(len(ds)):
                problems.append(ds[i][cfg["problem_field"]])
                answers.append(str(ds[i][cfg["answer_field"]]))
        _AIME2025_CACHE = (problems, answers)
    return _AIME2025_CACHE


def load_gsm8k():
    """Load GSM8K test set (cached after first call).

    Returns (problems, answers) where answers are the final numeric strings
    extracted after the ``####`` marker.
    """
    global _GSM8K_CACHE
    if _GSM8K_CACHE is None:
        import re
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        problems, answers = [], []
        for i in range(len(ds)):
            problems.append(ds[i]["question"])
            raw_answer = ds[i]["answer"]
            # Extract the number after "####"
            match = re.search(r"####\s*(.+?)\s*$", raw_answer)
            answers.append(match.group(1).strip() if match else raw_answer.strip())
        _GSM8K_CACHE = (problems, answers)
    return _GSM8K_CACHE


def load_aimo_amc():
    """Load AI-MO/aimo-validation-amc dataset (cached after first call).

    Returns (problems, answers) – 83 AMC12 problems with integer answers.
    """
    global _AIMO_AMC_CACHE
    if _AIMO_AMC_CACHE is None:
        ds = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")
        problems = [ds[i]["problem"] for i in range(len(ds))]
        # NOTE: answer field is float (e.g. 142.0); convert to int string
        # so that compute_score("142", "142") matches correctly.
        raw_answers = [ds[i]["answer"] for i in range(len(ds))]
        answers = []
        for a in raw_answers:
            if isinstance(a, float) and a == int(a):
                answers.append(str(int(a)))
            else:
                answers.append(str(a))
        _AIMO_AMC_CACHE = (problems, answers)
    return _AIMO_AMC_CACHE


def sample_math500_indices(ds, total_samples=200, seed=42):
    """Return indices into *ds* with proportional sampling per level.

    Each level gets a number of samples proportional to its share of the
    full dataset.  Rounding uses the largest-remainder method so the total
    is exactly *total_samples*.
    """
    rng = random.Random(seed)

    level_groups: dict[int, list[int]] = {}
    for i in range(len(ds)):
        lv = ds[i]["level"]
        level_groups.setdefault(lv, []).append(i)

    n_total = len(ds)
    sorted_levels = sorted(level_groups)

    exact = {lv: len(level_groups[lv]) / n_total * total_samples
             for lv in sorted_levels}
    floor_counts = {lv: int(exact[lv]) for lv in sorted_levels}
    remainders = {lv: exact[lv] - floor_counts[lv] for lv in sorted_levels}

    deficit = total_samples - sum(floor_counts.values())
    for lv in sorted(sorted_levels, key=lambda l: -remainders[l]):
        if deficit <= 0:
            break
        floor_counts[lv] += 1
        deficit -= 1

    sampled: list[int] = []
    for lv in sorted_levels:
        pool = level_groups[lv]
        k = min(floor_counts[lv], len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def sample_fixed_indices(total_size, sample_size, seed=42):
    """Return a fixed random subset of indices, or the full range."""
    if sample_size is None or sample_size <= 0 or sample_size >= total_size:
        return list(range(total_size))

    rng = random.Random(seed)
    return sorted(rng.sample(range(total_size), sample_size))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_problems(problems, tokenizer, max_prompt_length=1024):
    """Tokenize a list of math problems into left-padded tensors."""
    all_ids, all_masks = [], []

    for problem in problems:
        user_content = f"Problem:\n{problem}\n\nSolution:\n"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]       # (1, seq_len)
        mask = enc["attention_mask"]  # (1, seq_len)

        seq_len = ids.shape[1]
        if seq_len > max_prompt_length:
            ids = ids[:, -max_prompt_length:]
            mask = mask[:, -max_prompt_length:]
        elif seq_len < max_prompt_length:
            pad_len = max_prompt_length - seq_len
            ids = torch.cat(
                [torch.full((1, pad_len), tokenizer.pad_token_id, dtype=ids.dtype), ids],
                dim=1,
            )
            mask = torch.cat(
                [torch.zeros((1, pad_len), dtype=mask.dtype), mask], dim=1
            )

        all_ids.append(ids)
        all_masks.append(mask)

    input_ids = torch.cat(all_ids, dim=0)
    attention_mask = torch.cat(all_masks, dim=0)
    position_ids = torch.clip(torch.cumsum(attention_mask.long(), dim=-1) - 1, min=0)
    return input_ids, attention_mask, position_ids


# ---------------------------------------------------------------------------
# Generation + scoring (uses the already-loaded rollout workers)
# ---------------------------------------------------------------------------
def generate_and_score(
    actor_rollout_wg,
    tokenizer,
    problems,
    answers,
    max_prompt_length=1024,
    val_temperature=EVAL_TEMPERATURE,
    val_top_p=EVAL_TOP_P,
    batch_size=16,
    desc="[Eval]",
):
    """Generate one response per problem via rollout workers and score it.

    Returns
    -------
    n_correct : int    – number of correct answers
    accuracy : float   – overall accuracy
    """
    compute_score = _get_compute_score()
    world_size = actor_rollout_wg.world_size

    n_correct = 0
    n_total = len(problems)
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=desc, ncols=100):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)

        input_ids, attention_mask, position_ids = tokenize_problems(
            problems[start:end], tokenizer, max_prompt_length
        )

        gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
            "val_top_p": val_top_p,
        }

        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
        gen_padded.meta_info["val_temperature"] = val_temperature
        out_padded = actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        response_ids = out.batch["responses"]
        pred_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        for pred_text, answer in zip(pred_texts, answers[start:end]):
            score = compute_score(pred_text, answer)
            if score == 1.0:
                n_correct += 1

    accuracy = n_correct / n_total if n_total else 0.0
    return n_correct, accuracy


# ---------------------------------------------------------------------------
# CSV / PNG persistence
# ---------------------------------------------------------------------------
def _append_csv(csv_path, epoch, eval_type, overall_acc, math500_acc,
                aime2024_acc, aime2025_acc, gsm8k_acc, aimo_amc_acc):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "eval_type", "overall_acc",
                "math500_acc", "aime2024_acc", "aime2025_acc",
                "gsm8k_acc", "aimo_amc_acc",
            ])
        writer.writerow([
            epoch,
            eval_type,
            f"{overall_acc:.4f}",
            f"{math500_acc:.4f}",
            f"{aime2024_acc:.4f}",
            f"{aime2025_acc:.4f}",
            f"{gsm8k_acc:.4f}",
            f"{aimo_amc_acc:.4f}",
        ])


def plot_eval_results(csv_path, png_path):
    """Read the CSV and produce a 5x1 subplot training-progress plot."""
    epochs = []
    overall_accs = []
    math500_accs = []
    aime2024_accs = []
    aime2025_accs = []
    gsm8k_accs = []
    aimo_amc_accs = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            overall_accs.append(float(row["overall_acc"]))
            math500_accs.append(float(row["math500_acc"]))
            aime2024_accs.append(float(row["aime2024_acc"]))
            aime2025_accs.append(float(row["aime2025_acc"]))
            gsm8k_accs.append(float(row.get("gsm8k_acc") or 0))
            aimo_amc_accs.append(float(row.get("aimo_amc_acc") or 0))

    n_epochs = len(epochs)
    fig_width = max(10, n_epochs * 0.4)
    fig, axes = plt.subplots(5, 1, figsize=(fig_width, 16))

    # --- Helper for single-line subplots ---
    single_cfg = [
        (axes[0], overall_accs, "Overall", "royalblue", "o-"),
        (axes[1], math500_accs, "MATH-500", "#ff7f0e", "s-"),
    ]

    for ax, accs, title, color, marker in single_cfg:
        ax.plot(
            epochs, accs, marker,
            color=color, linewidth=2, markersize=6, zorder=5,
        )
        for i, (x, y) in enumerate(zip(epochs, accs)):
            offset = 10 if i % 2 == 0 else -16
            ax.annotate(
                f"{y:.1%}", (x, y),
                textcoords="offset points", xytext=(0, offset),
                ha="center", fontsize=7, fontweight="bold",
            )
        min_y, max_y = min(accs), max(accs)
        margin = max(0.03, (max_y - min_y) * 0.2)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)

    # --- AIME (2024 + 2025) subplot ---
    ax_aime = axes[2]
    ax_aime.plot(
        epochs, aime2024_accs, "^-",
        color="#2ca02c", linewidth=2, markersize=6, zorder=5, label="AIME2024",
    )
    ax_aime.plot(
        epochs, aime2025_accs, "D-",
        color="#d62728", linewidth=2, markersize=6, zorder=5, label="AIME2025",
    )
    for i, (x, y24, y25) in enumerate(zip(epochs, aime2024_accs, aime2025_accs)):
        off_24 = 10 if i % 2 == 0 else -16
        off_25 = -16 if i % 2 == 0 else 10
        ax_aime.annotate(
            f"{y24:.1%}", (x, y24),
            textcoords="offset points", xytext=(0, off_24),
            ha="center", fontsize=7, fontweight="bold", color="#2ca02c",
        )
        ax_aime.annotate(
            f"{y25:.1%}", (x, y25),
            textcoords="offset points", xytext=(0, off_25),
            ha="center", fontsize=7, fontweight="bold", color="#d62728",
        )
    all_aime = aime2024_accs + aime2025_accs
    min_y, max_y = min(all_aime), max(all_aime)
    margin = max(0.03, (max_y - min_y) * 0.2)
    ax_aime.set_ylim(min_y - margin, max_y + margin)
    ax_aime.set_xlabel("Epoch", fontsize=11)
    ax_aime.set_ylabel("Accuracy", fontsize=11)
    ax_aime.set_title("AIME (2024 + 2025)", fontsize=13, fontweight="bold")
    ax_aime.set_xticks(epochs)
    ax_aime.legend(fontsize=10, loc="best")
    ax_aime.grid(True, alpha=0.3)

    # --- GSM8K subplot ---
    ax_gsm = axes[3]
    ax_gsm.plot(
        epochs, gsm8k_accs, "p-",
        color="#9467bd", linewidth=2, markersize=6, zorder=5,
    )
    for i, (x, y) in enumerate(zip(epochs, gsm8k_accs)):
        offset = 10 if i % 2 == 0 else -16
        ax_gsm.annotate(
            f"{y:.1%}", (x, y),
            textcoords="offset points", xytext=(0, offset),
            ha="center", fontsize=7, fontweight="bold",
        )
    min_y, max_y = min(gsm8k_accs), max(gsm8k_accs)
    margin = max(0.03, (max_y - min_y) * 0.2)
    ax_gsm.set_ylim(min_y - margin, max_y + margin)
    ax_gsm.set_xlabel("Epoch", fontsize=11)
    ax_gsm.set_ylabel("Accuracy", fontsize=11)
    ax_gsm.set_title("GSM8K", fontsize=13, fontweight="bold")
    ax_gsm.set_xticks(epochs)
    ax_gsm.grid(True, alpha=0.3)

    # --- AIMO-AMC subplot ---
    ax_amc = axes[4]
    ax_amc.plot(
        epochs, aimo_amc_accs, "h-",
        color="#8c564b", linewidth=2, markersize=6, zorder=5,
    )
    for i, (x, y) in enumerate(zip(epochs, aimo_amc_accs)):
        offset = 10 if i % 2 == 0 else -16
        ax_amc.annotate(
            f"{y:.1%}", (x, y),
            textcoords="offset points", xytext=(0, offset),
            ha="center", fontsize=7, fontweight="bold",
        )
    min_y, max_y = min(aimo_amc_accs), max(aimo_amc_accs)
    margin = max(0.03, (max_y - min_y) * 0.2)
    ax_amc.set_ylim(min_y - margin, max_y + margin)
    ax_amc.set_xlabel("Epoch", fontsize=11)
    ax_amc.set_ylabel("Accuracy", fontsize=11)
    ax_amc.set_title("AIMO-AMC (AMC12 Validation)", fontsize=13, fontweight="bold")
    ax_amc.set_xticks(epochs)
    ax_amc.grid(True, alpha=0.3)

    fig.suptitle("Evaluation During Training", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[Eval] Plot saved to {png_path}")


# ---------------------------------------------------------------------------
# Main entry point (called from ray_trainer.py)
# ---------------------------------------------------------------------------
def evaluate_math500_epoch(
    actor_rollout_wg,
    tokenizer,
    epoch,
    total_epochs,
    output_dir,
    total_samples=200,
    max_prompt_length=1024,
    val_temperature=EVAL_TEMPERATURE,
    val_top_p=EVAL_TOP_P,
    batch_size=16,
    evaluate_datasets=None,
    gsm8k_sample_size=0,
):
    """Run evaluation for one training epoch on selected datasets.

    Parameters
    ----------
    actor_rollout_wg : WorkerGroup
        The already-initialised rollout workers (model is loaded on GPUs).
    tokenizer : PreTrainedTokenizer
    epoch : int           – current epoch index (0-based)
    total_epochs : int    – total number of training epochs
    output_dir : str      – directory for CSV / PNG outputs
    total_samples : int   – (unused, kept for API compatibility)
    max_prompt_length : int
    val_temperature : float
    val_top_p : float
    batch_size : int      – questions per generation call
    evaluate_datasets : list[str] or None
        Which datasets to evaluate. Supported names:
        "math500", "aime2024", "aime2025", "gsm8k", "aimo_amc".
        If None or empty, evaluate ALL datasets (backward compatible).
    gsm8k_sample_size : int
        Fixed-size GSM8K subset used for faster monitoring. Set to <= 0
        or >= dataset size to evaluate the full GSM8K test set.

    Returns
    -------
    overall_acc : float
    dataset_accs : dict[str, float]
        {"math500": ..., "aime2024": ..., "aime2025": ...,
         "gsm8k": ..., "aimo_amc": ...}
    """
    # Determine which datasets to evaluate
    ALL_DATASET_NAMES = {"math500", "aime2024", "aime2025", "gsm8k", "aimo_amc"}
    if evaluate_datasets is not None and len(evaluate_datasets) > 0:
        active_datasets = set(d.strip().lower() for d in evaluate_datasets)
        unknown = active_datasets - ALL_DATASET_NAMES
        if unknown:
            print(f"[Eval] WARNING: Unknown dataset names ignored: {unknown}")
        active_datasets = active_datasets & ALL_DATASET_NAMES
        if not active_datasets:
            print("[Eval] WARNING: No valid dataset names provided, evaluating ALL")
            active_datasets = ALL_DATASET_NAMES
    else:
        active_datasets = ALL_DATASET_NAMES

    print(f"[Eval] Active datasets for epoch {epoch}: {sorted(active_datasets)}")
    if val_temperature != EVAL_TEMPERATURE or val_top_p != EVAL_TOP_P:
        print(
            "[Eval] Override eval decoding params: "
            f"temperature={EVAL_TEMPERATURE}, top_p={EVAL_TOP_P} "
            f"(received temperature={val_temperature}, top_p={val_top_p})"
        )
    effective_temperature = EVAL_TEMPERATURE
    effective_top_p = EVAL_TOP_P
    eval_type = "full"

    # Initialize accumulators
    math500_correct, math500_acc = 0, 0.0
    aime2024_correct, aime2024_acc = 0, 0.0
    aime2025_correct, aime2025_acc = 0, 0.0
    gsm8k_correct, gsm8k_acc = 0, 0.0
    aimo_amc_correct, aimo_amc_acc = 0, 0.0
    math500_n = aime2024_n = aime2025_n = gsm8k_n = aimo_amc_n = 0

    # --- MATH-500 ---
    if "math500" in active_datasets:
        ds = load_math500()
        math500_problems = [ds[i]["problem"] for i in range(len(ds))]
        math500_answers = [ds[i]["answer"] for i in range(len(ds))]
        math500_n = len(math500_problems)

        print(
            f"\n{'='*60}\n"
            f"[Eval] Epoch {epoch}: MATH-500 ({math500_n}/{len(ds)})\n"
            f"{'='*60}"
        )
        math500_correct, math500_acc = generate_and_score(
            actor_rollout_wg=actor_rollout_wg,
            tokenizer=tokenizer,
            problems=math500_problems,
            answers=math500_answers,
            max_prompt_length=max_prompt_length,
            val_temperature=effective_temperature,
            val_top_p=effective_top_p,
            batch_size=batch_size,
            desc="[MATH-500 Eval]",
        )
        print(
            f"[Eval] MATH-500: "
            f"{math500_correct}/{math500_n} = {math500_acc:.4f}"
        )

    # --- AIME 2024 ---
    if "aime2024" in active_datasets:
        aime2024_problems, aime2024_answers = load_aime2024()
        aime2024_n = len(aime2024_problems)
        print(
            f"\n{'='*60}\n"
            f"[Eval] Epoch {epoch}: AIME2024 ({aime2024_n} questions)\n"
            f"{'='*60}"
        )
        aime2024_correct, aime2024_acc = generate_and_score(
            actor_rollout_wg=actor_rollout_wg,
            tokenizer=tokenizer,
            problems=aime2024_problems,
            answers=aime2024_answers,
            max_prompt_length=max_prompt_length,
            val_temperature=effective_temperature,
            val_top_p=effective_top_p,
            batch_size=batch_size,
            desc="[AIME2024 Eval]",
        )
        print(
            f"[Eval] AIME2024: "
            f"{aime2024_correct}/{aime2024_n} = {aime2024_acc:.4f}"
        )

    # --- AIME 2025 ---
    if "aime2025" in active_datasets:
        aime2025_problems, aime2025_answers = load_aime2025()
        aime2025_n = len(aime2025_problems)
        print(
            f"\n{'='*60}\n"
            f"[Eval] Epoch {epoch}: AIME2025 ({aime2025_n} questions)\n"
            f"{'='*60}"
        )
        aime2025_correct, aime2025_acc = generate_and_score(
            actor_rollout_wg=actor_rollout_wg,
            tokenizer=tokenizer,
            problems=aime2025_problems,
            answers=aime2025_answers,
            max_prompt_length=max_prompt_length,
            val_temperature=effective_temperature,
            val_top_p=effective_top_p,
            batch_size=batch_size,
            desc="[AIME2025 Eval]",
        )
        print(
            f"[Eval] AIME2025: "
            f"{aime2025_correct}/{aime2025_n} = {aime2025_acc:.4f}"
        )

    # --- GSM8K ---
    if "gsm8k" in active_datasets:
        gsm8k_all_problems, gsm8k_all_answers = load_gsm8k()
        gsm8k_indices = sample_fixed_indices(
            total_size=len(gsm8k_all_problems),
            sample_size=gsm8k_sample_size,
            seed=42,
        )
        gsm8k_problems = [gsm8k_all_problems[i] for i in gsm8k_indices]
        gsm8k_answers = [gsm8k_all_answers[i] for i in gsm8k_indices]
        gsm8k_n = len(gsm8k_problems)
        if 0 < gsm8k_n < len(gsm8k_all_problems):
            eval_type = "partial"
        print(
            f"\n{'='*60}\n"
            f"[Eval] Epoch {epoch}: GSM8K ({gsm8k_n}/{len(gsm8k_all_problems)})\n"
            f"{'='*60}"
        )
        gsm8k_correct, gsm8k_acc = generate_and_score(
            actor_rollout_wg=actor_rollout_wg,
            tokenizer=tokenizer,
            problems=gsm8k_problems,
            answers=gsm8k_answers,
            max_prompt_length=max_prompt_length,
            val_temperature=effective_temperature,
            val_top_p=effective_top_p,
            batch_size=batch_size,
            desc="[GSM8K Eval]",
        )
        print(
            f"[Eval] GSM8K: "
            f"{gsm8k_correct}/{gsm8k_n} = {gsm8k_acc:.4f}"
        )

    # --- AIMO-AMC ---
    if "aimo_amc" in active_datasets:
        aimo_amc_problems, aimo_amc_answers = load_aimo_amc()
        aimo_amc_n = len(aimo_amc_problems)
        print(
            f"\n{'='*60}\n"
            f"[Eval] Epoch {epoch}: AIMO-AMC ({aimo_amc_n} questions)\n"
            f"{'='*60}"
        )
        aimo_amc_correct, aimo_amc_acc = generate_and_score(
            actor_rollout_wg=actor_rollout_wg,
            tokenizer=tokenizer,
            problems=aimo_amc_problems,
            answers=aimo_amc_answers,
            max_prompt_length=max_prompt_length,
            val_temperature=effective_temperature,
            val_top_p=effective_top_p,
            batch_size=batch_size,
            desc="[AIMO-AMC Eval]",
        )
        print(
            f"[Eval] AIMO-AMC: "
            f"{aimo_amc_correct}/{aimo_amc_n} = {aimo_amc_acc:.4f}"
        )

    # --- Overall ---
    total_correct = (
        math500_correct + aime2024_correct + aime2025_correct
        + gsm8k_correct + aimo_amc_correct
    )
    total_problems = (
        math500_n + aime2024_n + aime2025_n
        + gsm8k_n + aimo_amc_n
    )
    overall_acc = total_correct / total_problems if total_problems else 0.0

    print(
        f"\n[Eval] Epoch {epoch} Overall: "
        f"{total_correct}/{total_problems} = {overall_acc:.4f}"
    )

    # Persist results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "eval_results.csv")
    png_path = os.path.join(output_dir, "eval_results.png")

    _append_csv(
        csv_path, epoch, eval_type,
        overall_acc, math500_acc, aime2024_acc, aime2025_acc,
        gsm8k_acc, aimo_amc_acc,
    )
    plot_eval_results(csv_path, png_path)

    dataset_accs = {
        "math500": math500_acc,
        "aime2024": aime2024_acc,
        "aime2025": aime2025_acc,
        "gsm8k": gsm8k_acc,
        "aimo_amc": aimo_amc_acc,
    }
    return overall_acc, dataset_accs
