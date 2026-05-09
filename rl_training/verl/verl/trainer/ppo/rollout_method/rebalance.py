"""Rebalance rollouts: generate extra rollouts to balance correct/incorrect ratio toward 50% per prompt."""

import sys
import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.rollout_method import dataprotoitem_to_dataproto


def rebalance_rollouts(actor_rollout_wg, reward_fn, batch, reward_tensor, gen_batch, n, max_n, metrics):
    """Generate extra rollouts to balance correct/incorrect ratio toward 50% per prompt.

    For prompts where the correct/incorrect split after n initial rollouts is not 50/50,
    generate up to (max_n - n) additional responses.  If the new response belongs to the
    minority class it replaces a random majority-class sample; otherwise it is discarded.
    The total number of responses per prompt stays at n.
    """
    extra_budget = max_n - n
    if extra_budget <= 0:
        return batch, reward_tensor

    uids = batch.non_tensor_batch['uid']
    seen = {}
    for uid in uids:
        if uid not in seen:
            seen[uid] = len(seen)
    unique_uids_ordered = list(seen.keys())

    unbalanced_orig_indices = []
    for i, uid in enumerate(unique_uids_ordered):
        uid_mask = uids == uid
        uid_scores = reward_tensor[uid_mask].sum(-1)
        n_correct = int((uid_scores > 0).sum())
        n_incorrect = int((uid_scores == 0).sum())
        if n_correct != n_incorrect:
            unbalanced_orig_indices.append(i)

    if not unbalanced_orig_indices:
        metrics['rebalance/n_unbalanced'] = 0
        return batch, reward_tensor

    n_unbalanced = len(unbalanced_orig_indices)
    metrics['rebalance/n_unbalanced_before'] = n_unbalanced
    print(f"[rebalance] {n_unbalanced}/{len(unique_uids_ordered)} prompts need rebalancing")

    extra_mask = [False] * len(gen_batch)
    for idx in unbalanced_orig_indices:
        extra_mask[idx] = True
    extra_gen_batch = dataprotoitem_to_dataproto(gen_batch[extra_mask])

    world_size = actor_rollout_wg.world_size
    extra_gen_batch_padded, pad_size = pad_dataproto_to_divisor(extra_gen_batch, world_size)

    print(f"[rebalance] Generating extra rollouts for {n_unbalanced} prompts ...")
    extra_gen_output_padded = actor_rollout_wg.generate_sequences(extra_gen_batch_padded)

    if pad_size > 0:
        keep = n_unbalanced * n
        extra_gen_output = dataprotoitem_to_dataproto(extra_gen_output_padded[:keep])
    else:
        extra_gen_output = extra_gen_output_padded

    extra_non_tensor_indices = []
    for idx in unbalanced_orig_indices:
        extra_non_tensor_indices.extend([idx * n] * n)
    extra_non_tensor = {
        key: val[extra_non_tensor_indices]
        for key, val in batch.non_tensor_batch.items()
    }

    extra_batch_for_reward = DataProto(
        batch=extra_gen_output.batch,
        non_tensor_batch=extra_non_tensor,
        meta_info=batch.meta_info,
    )
    extra_reward_tensor = reward_fn(extra_batch_for_reward)

    effective_per_prompt = min(extra_budget, n)
    n_rebalanced = 0
    col_w = len(str(n_unbalanced))

    for j, orig_idx in enumerate(unbalanced_orig_indices):
        uid = unique_uids_ordered[orig_idx]
        uid_mask = uids == uid
        uid_indices = np.where(uid_mask)[0]

        uid_scores = reward_tensor[uid_mask].sum(-1)
        init_c = int((uid_scores > 0).sum())
        init_i = int((uid_scores == 0).sum())
        status = f"{init_c}:{init_i}"

        for k in range(effective_per_prompt):
            uid_scores = reward_tensor[uid_mask].sum(-1)
            n_correct = int((uid_scores > 0).sum())
            n_incorrect = int((uid_scores == 0).sum())

            if n_correct == n_incorrect:
                status = f"{init_c}:{init_i}->{n_correct}:{n_incorrect} BALANCED"
                break

            extra_idx = j * n + k
            extra_is_correct = extra_reward_tensor[extra_idx].sum(-1).item() > 0

            need_correct = n_correct < n_incorrect
            replaced = False
            if (need_correct and extra_is_correct) or (not need_correct and not extra_is_correct):
                if need_correct:
                    majority_mask = (uid_scores == 0).numpy()
                else:
                    majority_mask = (uid_scores > 0).numpy()
                majority_global = uid_indices[majority_mask]
                replace_idx = int(np.random.choice(majority_global))

                for key in extra_gen_output.batch.keys():
                    batch.batch[key][replace_idx] = extra_gen_output.batch[key][extra_idx]
                reward_tensor[replace_idx] = extra_reward_tensor[extra_idx]
                replaced = True

            new_scores = reward_tensor[uid_mask].sum(-1)
            nc = int((new_scores > 0).sum())
            ni = int((new_scores == 0).sum())
            tag = "replace" if replaced else "discard"
            mark = "+" if extra_is_correct else "-"
            status = f"{init_c}:{init_i} try {k+1}/{effective_per_prompt} [{mark}{tag}] -> {nc}:{ni}"
            line = (f"\r[rebalance] [{j+1:>{col_w}}/{n_unbalanced}] {status}"
                    "                    ")
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            uid_scores = reward_tensor[uid_mask].sum(-1)
            nc = int((uid_scores > 0).sum())
            ni = int((uid_scores == 0).sum())
            if nc == ni:
                status = f"{init_c}:{init_i}->{nc}:{ni} BALANCED"
            else:
                status = f"{init_c}:{init_i}->{nc}:{ni} (not balanced)"

        balanced = (int((reward_tensor[uid_mask].sum(-1) > 0).sum())
                    == int((reward_tensor[uid_mask].sum(-1) == 0).sum()))
        if balanced:
            n_rebalanced += 1
        tag_final = "OK" if balanced else "FAIL"
        line = (f"\r[rebalance] [{j+1:>{col_w}}/{n_unbalanced}] "
                f"{status} [{tag_final}]"
                "                    ")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    n_still_unbalanced = n_unbalanced - n_rebalanced
    metrics['rebalance/n_rebalanced'] = n_rebalanced
    metrics['rebalance/n_still_unbalanced'] = n_still_unbalanced
    print(f"[rebalance] Done: {n_rebalanced}/{n_unbalanced} balanced, "
          f"{n_still_unbalanced} still unbalanced")

    return batch, reward_tensor
