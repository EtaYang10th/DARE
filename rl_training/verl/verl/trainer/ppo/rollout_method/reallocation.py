"""Reallocation rollouts: reallocate rollout budget per prompt based on predicted difficulty.

Pre-rollout: filter prompts outside keep_range and compute per-prompt allocation.
Grouped generation: generate each budget group with exact n (no trim/expand waste).
"""

import ast
import uuid
from collections import Counter

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.rollout_method import dataprotoitem_to_dataproto, COLOR_RED, COLOR_RESET


def pre_rollout_difficulty_filter(batch, predicted_labels, n, keep_range_str=None):
    """Filter prompts before rollout based on predicted accuracy and compute allocations.

    predicted_labels contains estimated accuracy (success rate) for each question.
    Prompts with accuracy outside keep_range are removed.
    Allocation per prompt: round((1 - accuracy) * n) * 2, i.e. proportional to
    expected incorrect count, giving harder questions more rollout budget.

    Args:
        batch: DataProto with prompts (before pop/generation), one item per prompt.
        predicted_labels: Tensor of predicted accuracy for every item in the training dataset.
        n: Base number of rollouts per prompt.
        keep_range_str: Raw config string for rollout_keep_range, e.g. "[0.1,0.9]".

    Returns:
        filtered_batch: DataProto with kept prompts.
        allocations: dict {position_in_filtered_batch: allocated_rollout_count}.
        n_skipped: Number of skipped prompts.
    """
    keep_range = None
    if keep_range_str is not None:
        rkr_clean = str(keep_range_str).strip().strip("'\"")
        if rkr_clean:
            keep_range = ast.literal_eval(rkr_clean)

    batch_indices = batch.non_tensor_batch['index']
    n_prompts = len(batch)

    keep_mask = np.ones(n_prompts, dtype=bool)
    raw_allocations = []
    skip_by_range = 0
    skip_by_zero_alloc = 0
    alloc_counter = Counter()

    for i in range(n_prompts):
        idx = int(batch_indices[i])
        accuracy = float(predicted_labels[idx])

        if keep_range is not None:
            keep_lo, keep_hi = keep_range
            if accuracy < keep_lo or accuracy > keep_hi:
                keep_mask[i] = False
                raw_allocations.append(0)
                skip_by_range += 1
                continue

        estimated_n_incorrect = round((1.0 - accuracy) * n)
        allocated = estimated_n_incorrect * 2
        if allocated == 0:
            keep_mask[i] = False
            raw_allocations.append(0)
            skip_by_zero_alloc += 1
            continue

        raw_allocations.append(allocated)
        alloc_counter[allocated] += 1

    n_kept = int(keep_mask.sum())
    print(f"{COLOR_RED}[pre_reallocation] ===== 预测难度预过滤 ====={COLOR_RESET}")
    print(f"{COLOR_RED}[pre_reallocation] 总prompts: {n_prompts}, "
          f"保留: {n_kept}, "
          f"跳过(范围外): {skip_by_range}, "
          f"跳过(分配为0): {skip_by_zero_alloc}{COLOR_RESET}")
    for alloc_val, count in sorted(alloc_counter.items()):
        print(f"{COLOR_RED}[pre_reallocation] 分配{alloc_val}个rollout: "
              f"{count}个prompt{COLOR_RESET}")

    if not keep_mask.all():
        filtered_batch = dataprotoitem_to_dataproto(batch[keep_mask.tolist()])
    else:
        filtered_batch = batch

    allocations = {}
    kept_idx = 0
    for i in range(n_prompts):
        if keep_mask[i]:
            allocations[kept_idx] = raw_allocations[i]
            kept_idx += 1

    n_skipped = n_prompts - n_kept
    return filtered_batch, allocations, n_skipped


def grouped_generate(actor_rollout_wg, batch, gen_batch, allocations, metrics):
    """Generate rollouts with exact per-prompt budget — no trim/expand waste.

    Groups prompts by their allocated rollout count, calls generate_sequences
    once per group with the exact ``n``, then concatenates results.

    Args:
        actor_rollout_wg: Worker group with generate_sequences().
        batch: DataProto (prompts after pop, has non_tensor fields only).
        gen_batch: DataProto (input_ids / attention_mask / position_ids).
        allocations: dict {prompt_position: allocated_rollout_count}.
        metrics: dict for logging.

    Returns:
        combined_batch: DataProto with all prompts, each repeated by its
            allocation count and unioned with generated responses.
        gen_batch_out: The gen_batch (unchanged, for downstream compatibility).
    """
    world_size = actor_rollout_wg.world_size

    groups = {}
    for idx, alloc_n in allocations.items():
        groups.setdefault(alloc_n, []).append(idx)

    print(f"{COLOR_RED}[grouped_generate] ===== 按预算分组生成 ====={COLOR_RESET}")
    for alloc_n in sorted(groups):
        print(f"{COLOR_RED}[grouped_generate] n={alloc_n}: "
              f"{len(groups[alloc_n])} 个prompt{COLOR_RESET}")

    all_parts = []
    total_rollouts = 0

    for alloc_n, prompt_indices in sorted(groups.items()):
        grp_mask = np.zeros(len(gen_batch), dtype=bool)
        for pi in prompt_indices:
            grp_mask[pi] = True

        grp_gen = dataprotoitem_to_dataproto(gen_batch[grp_mask.tolist()])
        grp_batch = dataprotoitem_to_dataproto(batch[grp_mask.tolist()])

        grp_batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(grp_batch))], dtype=object)

        grp_gen.meta_info['override_n'] = alloc_n

        grp_gen_padded, pad_sz = pad_dataproto_to_divisor(grp_gen, world_size)
        grp_output_padded = actor_rollout_wg.generate_sequences(grp_gen_padded)

        n_real = len(prompt_indices) * alloc_n
        if pad_sz > 0:
            grp_output = dataprotoitem_to_dataproto(grp_output_padded[:n_real])
        else:
            grp_output = grp_output_padded

        grp_batch = grp_batch.repeat(repeat_times=alloc_n, interleave=True)
        grp_batch = grp_batch.union(grp_output)

        all_parts.append(grp_batch)
        total_rollouts += n_real
        print(f"{COLOR_RED}[grouped_generate] n={alloc_n}: "
              f"生成 {n_real} 条rollout 完成{COLOR_RESET}")

    combined = DataProto.concat(all_parts) if len(all_parts) > 1 else all_parts[0]

    metrics['grouped_generate/n_groups'] = len(groups)
    metrics['grouped_generate/total_rollouts'] = total_rollouts
    print(f"{COLOR_RED}[grouped_generate] 总计 {len(allocations)} 个prompt, "
          f"{total_rollouts} 条rollout, "
          f"{len(groups)} 个预算组{COLOR_RESET}")

    return combined, gen_batch
