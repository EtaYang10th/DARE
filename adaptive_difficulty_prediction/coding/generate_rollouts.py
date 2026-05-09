"""
Step 2: Generate code rollouts with vLLM and evaluate them.

For each model, generates N rollouts per question, executes them,
computes rewards, and saves results.

Usage:
    python generate_rollouts.py \
        --questions_path ../datasets/coding/coding_questions.json \
        --output_dir ../datasets/coding/rollouts \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --n_rollouts 8 \
        --tensor_parallel_size 1
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

sys.set_int_max_str_digits(100000)

from execute_code import evaluate_solution, extract_code


SYSTEM_PROMPT_STDIO = (
    "You are a helpful assistant that solves programming problems. "
    "Think step by step, then provide your solution as a single Python code block. "
    "Your code should read from standard input and print to standard output."
)

SYSTEM_PROMPT_FUNCTION = (
    "You are a helpful assistant that solves programming problems. "
    "Think step by step, then provide your solution as a single Python code block "
    "implementing the requested function."
)


def format_prompt(question: dict) -> list[dict]:
    """Format a question into chat messages for the model."""
    is_fn = question.get("is_function_call", False)
    system = SYSTEM_PROMPT_FUNCTION if is_fn else SYSTEM_PROMPT_STDIO

    user_content = question["question"]
    starter = question.get("starter_code", "")
    if starter and starter.strip():
        user_content += f"\n\nStarter code:\n```python\n{starter.strip()}\n```"
    if is_fn and question.get("fn_name"):
        user_content += f"\n\nImplement the function `{question['fn_name']}`."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def generate_with_vllm(
    model_name: str,
    questions: list[dict],
    n_rollouts: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, list[str]]:
    """Generate rollouts using vLLM. Returns {question_id: [code1, code2, ...]}."""
    from vllm import LLM, SamplingParams

    print(f"[vLLM] Loading model: {model_name}")
    max_model_len = max_tokens + 1024
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=n_rollouts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = []
    question_ids = []
    for q in questions:
        messages = format_prompt(q)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        question_ids.append(q["id"])

    print(f"[vLLM] Generating {len(prompts)} prompts x {n_rollouts} rollouts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"[vLLM] Generation done in {elapsed:.1f}s "
          f"({len(prompts) * n_rollouts / elapsed:.1f} samples/s)")

    results: dict[str, list[str]] = {}
    for qid, output in zip(question_ids, outputs):
        raw_texts = [o.text for o in output.outputs]
        results[qid] = raw_texts

    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def evaluate_all_rollouts(
    questions: list[dict],
    rollouts: dict[str, list[str]],
    timeout_per_case: int = 10,
    max_test_cases: int = 20,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 200,
) -> dict:
    """
    Evaluate all rollouts for all questions with tqdm + incremental checkpointing.

    Args:
        checkpoint_path: if set, saves partial results here every checkpoint_interval questions.
                         On restart, loads existing results and skips already-evaluated questions.
        checkpoint_interval: save checkpoint every N questions.
    """
    from tqdm import tqdm

    existing_results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            existing_data = json.load(f)
            existing_results = existing_data.get("results", {})
        print(f"[RESUME] Loaded {len(existing_results)} existing eval results from checkpoint")

    results = dict(existing_results)
    q_lookup = {q["id"]: q for q in questions}

    to_eval = [(qid, raw_outputs) for qid, raw_outputs in rollouts.items()
               if qid not in results]
    print(f"[Eval] {len(to_eval)} questions to evaluate "
          f"({len(results)} already done, {len(rollouts)} total)")

    pbar = tqdm(to_eval, desc="Evaluating rollouts", unit="question")
    newly_done = 0

    for qid, raw_outputs in pbar:
        q = q_lookup.get(qid)
        if q is None:
            continue

        io = q["input_output"]
        fn_name = q.get("fn_name") if q.get("is_function_call") else None

        rewards = []
        pass_rates = []
        for raw in raw_outputs:
            code = extract_code(raw)
            eval_result = evaluate_solution(
                code, io,
                fn_name=fn_name,
                timeout_per_case=timeout_per_case,
                max_test_cases=max_test_cases,
            )
            rewards.append(eval_result["reward"])
            pass_rates.append(eval_result["pass_rate"])

        success_rate = sum(rewards) / len(rewards) if rewards else 0.0
        results[qid] = {
            "rewards": rewards,
            "success_rate": success_rate,
            "pass_rates": pass_rates,
        }
        newly_done += 1

        done_total = len(results)
        avg_sr = sum(v["success_rate"] for v in results.values()) / max(done_total, 1)
        pbar.set_postfix(done=done_total, avg_sr=f"{avg_sr:.3f}")

        if checkpoint_path and newly_done % checkpoint_interval == 0:
            _save_eval_checkpoint(checkpoint_path, results)

    if checkpoint_path:
        _save_eval_checkpoint(checkpoint_path, results)

    n_with_signal = sum(1 for v in results.values() if 0 < v["success_rate"] < 1)
    avg_sr = sum(v["success_rate"] for v in results.values()) / max(len(results), 1)
    print(f"\n[Eval] {len(results)} questions evaluated")
    print(f"  Average success rate: {avg_sr:.4f}")
    print(f"  Questions with learning signal (0 < sr < 1): "
          f"{n_with_signal}/{len(results)} ({100*n_with_signal/max(len(results),1):.1f}%)")

    return results


def _save_eval_checkpoint(path: str, results: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results}, f)
    os.replace(tmp, path)
    print(f"  [CHECKPOINT] saved {len(results)} results to {path}")


def save_results(
    model_name: str,
    rollouts: dict[str, list[str]],
    eval_results: dict,
    output_dir: str,
):
    """Save rollouts and evaluation results to disk."""
    safe_name = model_name.replace("/", "_")

    rollouts_path = os.path.join(output_dir, f"rollouts_{safe_name}.json")
    with open(rollouts_path, "w") as f:
        json.dump(rollouts, f, ensure_ascii=False)
    print(f"Saved rollouts to {rollouts_path}")

    rewards_path = os.path.join(output_dir, f"rewards_{safe_name}.json")
    with open(rewards_path, "w") as f:
        json.dump({"model": model_name, "results": eval_results}, f,
                  indent=2, ensure_ascii=False)
    print(f"Saved rewards to {rewards_path}")


def load_existing_rollouts(model_name: str, output_dir: str) -> Optional[dict]:
    """Load previously generated rollouts for resumption."""
    safe_name = model_name.replace("/", "_")
    rollouts_path = os.path.join(output_dir, f"rollouts_{safe_name}.json")
    if os.path.exists(rollouts_path):
        print(f"[RESUME] Found existing rollouts at {rollouts_path}")
        with open(rollouts_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate code rollouts")
    parser.add_argument("--questions_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name/path")
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--timeout_per_case", type=int, default=10)
    parser.add_argument("--max_test_cases", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip generation, only evaluate existing rollouts")
    parser.add_argument("--generate_only", action="store_true",
                        help="Skip evaluation, only generate rollouts")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing rollouts if available")
    parser.add_argument("--batch_start", type=int, default=0,
                        help="Start index for question subset (for parallel runs)")
    parser.add_argument("--batch_end", type=int, default=-1,
                        help="End index for question subset (-1 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.questions_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    if args.batch_end > 0:
        questions = questions[args.batch_start:args.batch_end]
        print(f"Using subset [{args.batch_start}:{args.batch_end}], "
              f"{len(questions)} questions")

    rollouts = None
    if args.eval_only or args.resume:
        rollouts = load_existing_rollouts(args.model, args.output_dir)

    if rollouts is None and not args.eval_only:
        rollouts = generate_with_vllm(
            model_name=args.model,
            questions=questions,
            n_rollouts=args.n_rollouts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        safe_name = args.model.replace("/", "_")
        rollouts_path = os.path.join(args.output_dir, f"rollouts_{safe_name}.json")
        with open(rollouts_path, "w") as f:
            json.dump(rollouts, f, ensure_ascii=False)
        print(f"[SAVE] Rollouts saved to {rollouts_path}")

    if rollouts is None:
        print("ERROR: No rollouts available. Run generation first.")
        sys.exit(1)

    if not args.generate_only:
        safe_name = args.model.replace("/", "_")
        ckpt_path = os.path.join(args.output_dir, f"eval_checkpoint_{safe_name}.json")
        eval_results = evaluate_all_rollouts(
            questions, rollouts,
            timeout_per_case=args.timeout_per_case,
            max_test_cases=args.max_test_cases,
            checkpoint_path=ckpt_path,
            checkpoint_interval=100,
        )
        save_results(args.model, rollouts, eval_results, args.output_dir)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    print("[DONE] generate_rollouts complete.")


if __name__ == "__main__":
    main()
