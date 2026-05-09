"""Post-rollout filtering helpers."""

import ast

import numpy as np

from verl.trainer.ppo.rollout_method import dataprotoitem_to_dataproto


def post_rollout_keep_range_filter(batch, reward_tensor, rollout_keep_range_str):
    """Filter prompts whose realized accuracy falls outside rollout_keep_range."""
    if rollout_keep_range_str is None:
        return batch, reward_tensor, None

    rkr_clean = str(rollout_keep_range_str).strip().strip("'\"")
    if not rkr_clean:
        return batch, reward_tensor, None

    keep_lo, keep_hi = ast.literal_eval(rkr_clean)
    uids = batch.non_tensor_batch['uid']
    seen_uids = {}
    for uid in uids:
        if uid not in seen_uids:
            seen_uids[uid] = len(seen_uids)
    unique_uids = list(seen_uids.keys())

    keep_mask = np.ones(len(batch), dtype=bool)
    n_filtered = 0
    for uid in unique_uids:
        uid_mask = uids == uid
        uid_rewards = reward_tensor[uid_mask].sum(-1)
        n_correct = int((uid_rewards > 0).sum())
        n_total = int(uid_mask.sum())
        accuracy = n_correct / n_total
        if accuracy < keep_lo or accuracy > keep_hi:
            keep_mask[uid_mask] = False
            n_filtered += 1

    before_count = len(batch)
    if n_filtered > 0:
        batch = dataprotoitem_to_dataproto(batch[keep_mask.tolist()])
        reward_tensor = reward_tensor[keep_mask]
        batch.batch['token_level_scores'] = reward_tensor

    stats = {
        'n_filtered': n_filtered,
        'n_kept': len(unique_uids) - n_filtered,
        'n_prompts_before': len(unique_uids),
        'response_count_before': before_count,
        'response_count_after': len(batch),
        'keep_lo': keep_lo,
        'keep_hi': keep_hi,
    }
    return batch, reward_tensor, stats
