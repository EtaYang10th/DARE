"""Rollout strategy modules extracted from RayPPOTrainer.

Provides reusable functions for post-rollout generation strategies
(rebalance, inspiration, reallocation) and sample attenuation tracking.
"""

import numpy as np
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import DataProtoItem

COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object.

    Always extracts raw tensors from the TensorDict to build a fresh one,
    guaranteeing that TensorDict.batch_size matches actual tensor shapes.
    TensorDict views from boolean/slice indexing can carry stale batch_size
    through serialization, causing downstream shape mismatches.
    """
    batch = item.batch
    if isinstance(batch, TensorDict):
        keys = list(batch.keys())
        if keys:
            tensors = {k: batch[k].contiguous() for k in keys}
            return DataProto.from_dict(
                tensors=tensors,
                non_tensors=item.non_tensor_batch,
                meta_info=item.meta_info,
            )
        else:
            non_tensors = {}
            for key, val in item.non_tensor_batch.items():
                non_tensors[key] = np.array(val, dtype=object)
            inferred_size = next(iter(non_tensors.values())).shape[0] if non_tensors else 0
            batch = TensorDict(source={}, batch_size=[inferred_size])
            return DataProto(batch=batch, non_tensor_batch=non_tensors, meta_info=item.meta_info)

    return DataProto.from_dict(
        tensors=batch,
        non_tensors=item.non_tensor_batch,
        meta_info=item.meta_info,
    )


from verl.trainer.ppo.rollout_method.rebalance import rebalance_rollouts
from verl.trainer.ppo.rollout_method.inspiration import (
    inspiration_for_hard,
    inspiration_for_hard_memory,
    apply_hard_length_reward_shaping,
)
# from verl.trainer.ppo.rollout_method.inspiration import inspiration_for_easy
# Legacy easy-prompt regeneration is intentionally disabled to avoid
# confusion with the current length-penalty-based implementation.
from verl.trainer.ppo.rollout_method.reallocation import (
    pre_rollout_difficulty_filter,
    grouped_generate,
)
from verl.trainer.ppo.rollout_method.attenuation import attenuation_update, print_attenuation_stats
from verl.trainer.ppo.rollout_method.accuracy import (
    assign_original_accuracy_by_uid,
    backfill_original_accuracy_by_contiguous_groups,
    compute_solve_none_all,
    compute_ref_solve_none_all,
)
from verl.trainer.ppo.rollout_method.filtering import post_rollout_keep_range_filter
