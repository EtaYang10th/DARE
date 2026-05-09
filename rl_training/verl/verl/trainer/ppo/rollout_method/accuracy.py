"""Accuracy-related helpers for rollout and replay batches."""

import numpy as np
import torch


def assign_original_accuracy_by_uid(batch, reward_tensor):
    """Compute prompt-level original accuracy grouped by uid and write it into batch."""
    uids = batch.non_tensor_batch['uid']
    orig_acc = np.empty(len(batch), dtype=object)
    seen = {}
    for uid in uids:
        if uid not in seen:
            seen[uid] = len(seen)
    for uid in seen:
        uid_mask = uids == uid
        uid_rewards = reward_tensor[uid_mask].sum(-1)
        acc_val = int((uid_rewards > 0).sum()) / int(uid_mask.sum())
        for idx in np.where(uid_mask)[0]:
            orig_acc[idx] = acc_val
    batch.non_tensor_batch['original_accuracy'] = orig_acc
    return orig_acc


def backfill_original_accuracy_by_contiguous_groups(buffer, contiguous_groups_fn):
    """Backfill original_accuracy for replay buffer using contiguous index groups."""
    buf_len = len(buffer)
    buf_indices = np.array(buffer.non_tensor_batch['index'], dtype=int)
    buf_groups = contiguous_groups_fn(buf_indices)
    buf_scores = buffer.batch['token_level_scores']
    buf_orig_acc = np.empty(buf_len, dtype=object)
    for _gid, positions in buf_groups:
        group_rewards = buf_scores[positions].sum(dim=-1)
        acc = float((group_rewards > 0).float().mean())
        for pos in positions:
            buf_orig_acc[pos] = acc
    buffer.non_tensor_batch['original_accuracy'] = buf_orig_acc
    return buf_orig_acc


def compute_solve_none_all(batch, reward_tensor):
    """Count prompts with all failures or all successes grouped by uid."""
    uids = batch.non_tensor_batch['uid']
    unique_uids = np.unique(uids)
    solve_none = 0
    solve_all = 0

    for uid in unique_uids:
        uid_mask = uids == uid
        uid_rewards = reward_tensor[uid_mask].sum(-1)
        if (uid_rewards == 0).all():
            solve_none += 1
        elif (uid_rewards == 1).all():
            solve_all += 1

    return {
        'batch/solve_none': solve_none,
        'batch/solve_all': solve_all,
    }


def compute_ref_solve_none_all(batch, reward_tensor):
    """Reference-rollout variant returning tuple for backward-compatible callers."""
    stats = compute_solve_none_all(batch, reward_tensor)
    return stats['batch/solve_none'], stats['batch/solve_all']
