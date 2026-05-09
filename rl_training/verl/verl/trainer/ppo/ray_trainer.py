# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# teacher_replay
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import json
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type

import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.coverage_tracking import SampleTrainingTracker
from verl.trainer.ppo.epoch_timing import _timer, _accumulate_time
from verl.trainer.ppo.metrics import reduce_metrics, compute_step_metrics, compute_timing_metrics, compute_epoch_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import shutil 
import time
from collections import defaultdict
import random
import math

from verl.trainer.ppo.rollout_method import (
    dataprotoitem_to_dataproto,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_RESET,
    rebalance_rollouts,
    inspiration_for_hard,
    inspiration_for_hard_memory,
    apply_hard_length_reward_shaping,
    # inspiration_for_easy,  # Legacy easy regeneration disabled; controlled by easy_length_penalty_coeff.
    pre_rollout_difficulty_filter,
    grouped_generate,
    attenuation_update,
    print_attenuation_stats,
    assign_original_accuracy_by_uid,
    backfill_original_accuracy_by_contiguous_groups,
    compute_solve_none_all,
    compute_ref_solve_none_all,
    post_rollout_keep_range_filter,
)
from verl.trainer.ppo.easy_penalty_diagnostics import (
    report_easy_penalty_stats,
    write_easy_penalty_diagnostics,
)
from verl.trainer.ppo.is_diagnostics import log_step_diagnostics


WorkerType = Type[Worker]

def _contiguous_groups(idx_array):
    """Identify contiguous runs of the same index.

    Returns list of (index_value, [positions]).
    Supports variable-size groups produced by reallocation.
    """
    groups = []
    if len(idx_array) == 0:
        return groups
    cur = idx_array[0]
    pos = [0]
    for i in range(1, len(idx_array)):
        if idx_array[i] == cur:
            pos.append(i)
        else:
            groups.append((int(cur), pos))
            cur = idx_array[i]
            pos = [i]
    groups.append((int(cur), pos))
    return groups


def apply_easy_length_reward_shaping(batch: DataProto,
                                     easy_threshold: float,
                                     length_penalty_coeff: float):
    """Apply easy-prompt length penalty directly to token-level rewards."""
    stats = {
        'enabled': False,
        'status': 'disabled',
        'easy_threshold': float(easy_threshold),
        'length_penalty_coeff': float(length_penalty_coeff),
        'total_responses': int(len(batch)),
        'total_prompts': 0,
        'easy_responses': 0,
        'easy_prompts': 0,
        'applied_responses': 0,
        'applied_prompts': 0,
        'avg_easy_weight': 0.0,
        'avg_applied_length_norm': 0.0,
        'avg_applied_response_len': 0.0,
        'avg_penalty': 0.0,
        'max_penalty': 0.0,
        'total_penalty': 0.0,
        'reward_mean_before': 0.0,
        'reward_mean_after': 0.0,
        'reward_delta_mean': 0.0,
        'applied_clip_ratio': 0.0,
    }

    token_level_scores = batch.batch['token_level_scores']
    batch.batch['token_level_rewards'] = token_level_scores.clone()
    batch.meta_info['easy_penalty_stats'] = stats

    if length_penalty_coeff <= 0 or len(batch) == 0:
        return stats

    orig_acc_arr = batch.non_tensor_batch.get('original_accuracy', None)
    if orig_acc_arr is None:
        stats['status'] = 'missing_original_accuracy'
        batch.meta_info['easy_penalty_stats'] = stats
        return stats

    responses = batch.batch['responses']
    response_length = int(responses.shape[-1])
    if response_length <= 0:
        stats['status'] = 'empty_response'
        batch.meta_info['easy_penalty_stats'] = stats
        return stats

    device = token_level_scores.device
    work_dtype = torch.float32
    response_mask = batch.batch['attention_mask'][:, -response_length:].to(work_dtype)
    response_len = response_mask.sum(dim=-1)
    length_norm = response_len / max(float(response_length), 1.0)
    raw_sequence_reward = token_level_scores.sum(dim=-1).to(work_dtype)

    denom = max(1.0 - float(easy_threshold), 1e-8)
    acc_tensor = torch.tensor(
        [float(a) for a in orig_acc_arr],
        dtype=work_dtype,
        device=device)
    easy_weight = ((acc_tensor - float(easy_threshold)) / denom).clamp(0.0, 1.0)
    correct_mask = (raw_sequence_reward > 0).to(work_dtype)
    applied_weight = easy_weight * correct_mask
    penalty = float(length_penalty_coeff) * applied_weight * length_norm

    token_level_rewards = token_level_scores.clone()
    last_token_idx = response_len.long().clamp(min=1) - 1
    row_idx = torch.arange(len(batch), device=device)
    token_level_rewards[row_idx, last_token_idx] -= penalty.to(token_level_rewards.dtype)
    batch.batch['token_level_rewards'] = token_level_rewards

    easy_mask_cpu = (easy_weight > 0).detach().cpu().numpy()
    applied_mask_cpu = (applied_weight > 0).detach().cpu().numpy()
    length_norm_cpu = length_norm.detach().cpu().numpy()
    response_len_cpu = response_len.detach().cpu().numpy()
    penalty_cpu = penalty.detach().cpu().numpy()
    raw_reward_cpu = raw_sequence_reward.detach().cpu().numpy()
    shaped_reward_cpu = token_level_rewards.sum(dim=-1).detach().cpu().numpy()

    stats['enabled'] = True
    stats['status'] = 'applied' if bool(applied_mask_cpu.any()) else 'no_matching_samples'
    stats['easy_responses'] = int(easy_mask_cpu.sum())
    stats['applied_responses'] = int(applied_mask_cpu.sum())
    stats['reward_mean_before'] = float(raw_reward_cpu.mean()) if len(raw_reward_cpu) > 0 else 0.0
    stats['reward_mean_after'] = float(shaped_reward_cpu.mean()) if len(shaped_reward_cpu) > 0 else 0.0
    stats['reward_delta_mean'] = stats['reward_mean_before'] - stats['reward_mean_after']

    if stats['easy_responses'] > 0:
        stats['avg_easy_weight'] = float(easy_weight[easy_weight > 0].mean().detach().cpu().item())

    if stats['applied_responses'] > 0:
        applied_penalty = penalty_cpu[applied_mask_cpu]
        applied_length_norm = length_norm_cpu[applied_mask_cpu]
        applied_response_len = response_len_cpu[applied_mask_cpu]
        stats['avg_applied_length_norm'] = float(applied_length_norm.mean())
        stats['avg_applied_response_len'] = float(applied_response_len.mean())
        stats['avg_penalty'] = float(applied_penalty.mean())
        stats['max_penalty'] = float(applied_penalty.max())
        stats['total_penalty'] = float(applied_penalty.sum())
        stats['applied_clip_ratio'] = float(np.mean(applied_length_norm >= 0.999))

    uids_arr = batch.non_tensor_batch.get('uid', None)
    if uids_arr is not None:
        prompt_easy = {}
        prompt_applied = {}
        for i, uid in enumerate(uids_arr):
            if uid not in prompt_easy:
                prompt_easy[uid] = bool(easy_mask_cpu[i])
            prompt_applied[uid] = prompt_applied.get(uid, False) or bool(applied_mask_cpu[i])
        stats['total_prompts'] = len(prompt_easy)
        stats['easy_prompts'] = int(sum(prompt_easy.values()))
        stats['applied_prompts'] = int(sum(prompt_applied.values()))
    else:
        stats['total_prompts'] = int(len(batch))
        stats['easy_prompts'] = stats['easy_responses']
        stats['applied_prompts'] = stats['applied_responses']

    batch.meta_info['easy_penalty_stats'] = stats
    return stats


def select_buffer(buffer, teacher_scores, config):
    
    import numpy as np
    import torch
    import random

    indices = np.array(buffer.non_tensor_batch['index'], dtype=int)
    raw_groups = _contiguous_groups(indices)
    n_raw = len(raw_groups)
    print(f"Before deduplication, buffer size: {n_raw}")

    for i, (gid, positions) in enumerate(raw_groups):
        assert all(indices[p] == gid for p in positions), f"Index mismatch at group {i}"

    # deduplication: keep last occurrence of each unique prompt index
    last_occurrence = {}
    for i, (gid, _) in enumerate(raw_groups):
        last_occurrence[gid] = i

    dedup_mask = np.zeros(len(indices), dtype=bool)
    for gi in sorted(last_occurrence.values()):
        for p in raw_groups[gi][1]:
            dedup_mask[p] = True
    buffer = dataprotoitem_to_dataproto(buffer[dedup_mask.tolist()])

    indices = np.array(buffer.non_tensor_batch['index'], dtype=int)
    groups = _contiguous_groups(indices)
    n_dedup = len(groups)
    
    # remove all 0 or all 1 (skip when using IS-based selection to preserve
    # extreme samples for more accurate difficulty estimation)
    selection_method = config.data.get('selection_method', 'teacher')
    if selection_method != 'is':
        token_level_scores = buffer.batch['token_level_scores']
        keep_gi = []
        for i, (gid, positions) in enumerate(groups):
            group_rewards = token_level_scores[positions].sum(dim=-1)
            avg_reward = group_rewards.mean().item()
            if avg_reward != 0.0 and avg_reward != 1.0:
                keep_gi.append(i)
        if len(keep_gi) < len(groups):
            keep_mask = np.zeros(len(buffer), dtype=bool)
            for gi in keep_gi:
                for p in groups[gi][1]:
                    keep_mask[p] = True
            buffer = dataprotoitem_to_dataproto(buffer[keep_mask.tolist()])
            indices = np.array(buffer.non_tensor_batch['index'], dtype=int)
            groups = _contiguous_groups(indices)
        n = len(groups)
        print(f"After deduplication and filtering, buffer size: {n}")
    else:
        n = n_dedup
        print(f"After deduplication (IS mode, kept all), buffer size: {n}")

    for i, (gid, positions) in enumerate(groups):
        assert all(indices[p] == gid for p in positions), f"Index mismatch at group {i}"
    
    if config.data.replay_strategy == "teacher":
        raise NotImplementedError
    elif config.data.replay_strategy == "random":
        n_needed = int((1 - config.data.sigma) * config.data.train_batch_size)

        # --- Parse rollout_priority_range ---
        rollout_priority_range_str = config.data.get('rollout_priority_range', None)
        use_priority = False
        if rollout_priority_range_str is not None:
            import ast
            rpr_clean = str(rollout_priority_range_str).strip().strip("'\"")
            if rpr_clean:
                priority_lo, priority_hi = ast.literal_eval(rpr_clean)
                if 'original_accuracy' in buffer.non_tensor_batch:
                    use_priority = True
                else:
                    print("[rollout_priority] WARNING: original_accuracy not in buffer, "
                          "priority disabled for this call")

        group_acc = None
        if use_priority:
            orig_acc = np.array(buffer.non_tensor_batch['original_accuracy'], dtype=float)
            group_acc = np.array([orig_acc[positions[0]] for _, positions in groups])

        all_groups = list(range(n))
        if use_priority:
            in_range = [g for g in all_groups if priority_lo <= group_acc[g] <= priority_hi]
            out_range = [g for g in all_groups if not (priority_lo <= group_acc[g] <= priority_hi)]
            random.shuffle(in_range)
            random.shuffle(out_range)
            ordered_groups = in_range + out_range
        else:
            random.shuffle(all_groups)
            ordered_groups = all_groups

        sel_groups = ordered_groups[:min(n_needed, len(ordered_groups))]

        if use_priority:
            n_priority_sel = sum(1 for g in sel_groups
                                 if priority_lo <= group_acc[g] <= priority_hi)
            n_total_in_range = sum(1 for g in range(n)
                                   if priority_lo <= group_acc[g] <= priority_hi)
            print(f"{COLOR_RED}[rollout_priority] 共 {n} 组中有 {n_total_in_range} 组在 "
                  f"[{priority_lo}, {priority_hi}] 范围内，已选择 {n_priority_sel} 个优先组 + "
                  f"{len(sel_groups) - n_priority_sel} 个非优先组，共需 {n_needed} 个{COLOR_RESET}")
    
    selected_mask = np.zeros(len(buffer), dtype=bool)
    for gi in sel_groups:
        for p in groups[gi][1]:
            selected_mask[p] = True
    selected_data = dataprotoitem_to_dataproto(buffer[selected_mask.tolist()])
    
    remaining_budget = config.data.buffer_size - int(config.data.sigma * config.data.train_batch_size)  
    if n > remaining_budget:
        if remaining_budget == 0:
            buffer = None
            return buffer, selected_data
        else:
            keep_groups = groups[-remaining_budget:]
            keep_mask = np.zeros(len(buffer), dtype=bool)
            for _, positions in keep_groups:
                for p in positions:
                    keep_mask[p] = True
            buffer = dataprotoitem_to_dataproto(buffer[keep_mask.tolist()])
        indices = np.array(buffer.non_tensor_batch['index'], dtype=int)
        groups = _contiguous_groups(indices)
        n = len(groups)
        for i, (gid, positions) in enumerate(groups):
            assert all(indices[p] == gid for p in positions), f"Index mismatch at group {i}"
        
    return buffer, selected_data

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Teacher = 7

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_teacher = Role.Teacher in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        # from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        # self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
        #                                  tokenizer=self.tokenizer,
        #                                  prompt_key=self.config.data.prompt_key,
        #                                  max_prompt_length=self.config.data.max_prompt_length,
        #                                  filter_prompts=True,
        #                                  return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                  truncation='error',
        #                                  format_reward=self.config.data.get('format_reward', False))
        # train_batch_size = self.config.data.train_batch_size
        # if self.config.trainer.rejection_sample:
        #     train_batch_size *= self.config.trainer.rejection_sample_multiplier
        #     train_batch_size = int(train_batch_size)
        # self.train_dataloader = DataLoader(dataset=self.train_dataset,
        #                                    batch_size=train_batch_size,
        #                                    shuffle=True, 
        #                                    drop_last=True,
        #                                    collate_fn=collate_fn)

        # self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
        #                                tokenizer=self.tokenizer,
        #                                prompt_key=self.config.data.prompt_key,
        #                                max_prompt_length=self.config.data.max_prompt_length,
        #                                filter_prompts=True,
        #                                return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                truncation='error',
        #                                  format_reward=self.config.data.get('format_reward', False))
        # self.val_dataloader = DataLoader(dataset=self.val_dataset,
        #                                  batch_size=len(self.val_dataset),
        #                                  shuffle=True,
        #                                  drop_last=True,
        #                                  collate_fn=collate_fn)

        # assert len(self.train_dataloader) >= 1
        # assert len(self.val_dataloader) >= 1

        # print(f'Size of train dataloader: {len(self.train_dataloader)}')
        # print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # set total steps and save freq
        self.total_training_steps = self.config.data.mu * self.config.trainer.total_epochs
        if self.config.trainer.save_freq < 0:
            self.config.trainer.save_freq = self.config.data.mu

        # inject total_training_steps to actor/critic optim_config. This is hacky.

        if self.config.trainer.total_training_steps is not None:
            raise NotImplementedError

        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
            self.config.critic.optim.total_training_steps = self.total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('Validation: Generation end.')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        self.prefix_name = "verl_controller"
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # init teacher model
        if self.use_teacher:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Teacher)
            print(f"Teacher resource pool: {resource_pool}")
            teacher_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Teacher],
                                                config=self.config.teacher_model,
                                            )
            self.resource_pool_to_cls[resource_pool]['teacher'] = teacher_cls
        
        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print("[Worker Group] Creating worker group for other roles")
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, name_prefix=self.prefix_name)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
                
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # initialize teacher
        if self.use_teacher:
            self.teacher_wg = all_wg['teacher']
            self.teacher_wg.init_model()
        
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _cleanup_old_step_checkpoints(self, parent_dir, max_keep=5):
        """Remove old global_step_* checkpoint directories, keeping only the latest `max_keep`."""
        if not os.path.isdir(parent_dir):
            return
        step_dirs = []
        for name in os.listdir(parent_dir):
            if name.startswith('global_step_') and os.path.isdir(os.path.join(parent_dir, name)):
                try:
                    step_num = int(name.split('_')[-1])
                    step_dirs.append((step_num, name))
                except ValueError:
                    continue
        step_dirs.sort(key=lambda x: x[0])
        dirs_to_remove = step_dirs[:-max_keep] if len(step_dirs) > max_keep else []
        for _, name in dirs_to_remove:
            path = os.path.join(parent_dir, name)
            print(f'Removing old checkpoint: {path}')
            shutil.rmtree(path)

    def _expected_checkpoint_world_size(self):
        return int(self.config.trainer.nnodes) * int(self.config.trainer.n_gpus_per_node)

    def _checkpoint_dir_complete(self, path, world_size):
        if path is None or not os.path.isdir(path):
            return False

        required_files = []
        for rank in range(world_size):
            required_files.append(os.path.join(path, f'model_world_size_{world_size}_rank_{rank}.pt'))
            required_files.append(os.path.join(path, f'extra_state_world_size_{world_size}_rank_{rank}.pt'))

        missing_files = [file for file in required_files if not os.path.exists(file)]
        if missing_files:
            print(f'Checkpoint directory incomplete: {path}')
            for file in missing_files[:8]:
                print(f'  missing: {file}')
            if len(missing_files) > 8:
                print(f'  ... and {len(missing_files) - 8} more missing files')
            return False
        return True

    def _save_checkpoint(self, save_checkpoints=False, save_optimizer=True):
        # Always write latest_checkpointed_iteration.txt so resume can
        # discover the most recent step regardless of checkpoint format.
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir,
            'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        if save_checkpoints:
            actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        'final_checkpoints')
        else:
            actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            save_checkpoints=save_checkpoints,
            save_optimizer=save_optimizer,
        )

        if self.use_critic:
            if save_checkpoints:
                critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                        'final_checkpoints')
            else:
                critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                save_checkpoints=save_checkpoints,
                save_optimizer=save_optimizer,
            )

        if not save_checkpoints:
            max_keep = self.config.trainer.get('max_ckpt_num', 5)
            actor_parent = os.path.join(self.config.trainer.default_local_dir, 'actor')
            self._cleanup_old_step_checkpoints(actor_parent, max_keep=max_keep)
            if self.use_critic:
                critic_parent = os.path.join(self.config.trainer.default_local_dir, 'critic')
                self._cleanup_old_step_checkpoints(critic_parent, max_keep=max_keep)

    def _find_latest_global_step_dir(self, parent_dir):
        """Find the latest global_step_* HF checkpoint directory."""
        if not os.path.isdir(parent_dir):
            return None, None
        step_dirs = []
        for name in os.listdir(parent_dir):
            if name.startswith('global_step_') and os.path.isdir(os.path.join(parent_dir, name)):
                try:
                    step_num = int(name.split('_')[-1])
                    step_dirs.append((step_num, name))
                except ValueError:
                    continue
        if not step_dirs:
            return None, None
        step_dirs.sort(key=lambda x: x[0])
        latest_step, latest_name = step_dirs[-1]
        return latest_step, os.path.join(parent_dir, latest_name)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        if self.config.trainer.resume_mode == 'auto':
            if self.config.trainer.resume_from_path is None and self.config.trainer.get('checkpoint_path', None) is None:
                print('Training from scratch')
                return 0

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir,
            'latest_checkpointed_iteration.txt')
        if os.path.exists(local_latest_checkpointed_iteration):
            with open(local_latest_checkpointed_iteration, 'r') as f:
                self.global_steps = int(f.read().strip())
        else:
            # Fallback: infer from latest global_step_* directory
            actor_parent = os.path.join(self.config.trainer.default_local_dir, 'actor')
            inferred_step, _ = self._find_latest_global_step_dir(actor_parent)
            if inferred_step is not None:
                self.global_steps = inferred_step
                print(f'No latest_checkpointed_iteration.txt, '
                      f'inferred global_steps={inferred_step} from global_step_* dirs')
            else:
                print('No latest checkpoint found, training from scratch')
                return 0

        resume_name = self.config.trainer.resume_from_path or self.config.trainer.get('checkpoint_path', None)
        if resume_name is None:
            print('Training from scratch')
            return 0

        actor_path = os.path.join(self.config.trainer.default_local_dir, 'actor', resume_name)
        critic_path = os.path.join(self.config.trainer.default_local_dir, 'critic', resume_name)
        world_size = self._expected_checkpoint_world_size()

        use_fallback_checkpoint = False
        use_hf_checkpoint = False
        actor_resume_complete = self._checkpoint_dir_complete(actor_path, world_size)
        critic_resume_complete = (not self.use_critic) or self._checkpoint_dir_complete(critic_path, world_size)
        if not (actor_resume_complete and critic_resume_complete):
            # Try config.checkpoint_path as sharded fallback
            fallback_name = self.config.trainer.get('checkpoint_path', None)
            if fallback_name:
                fallback_actor_path = os.path.join(self.config.trainer.default_local_dir, 'actor', fallback_name)
                fallback_critic_path = os.path.join(self.config.trainer.default_local_dir, 'critic', fallback_name)
                fallback_actor_complete = self._checkpoint_dir_complete(fallback_actor_path, world_size)
                fallback_critic_complete = (not self.use_critic) or self._checkpoint_dir_complete(fallback_critic_path, world_size)
                if fallback_actor_complete and fallback_critic_complete:
                    print(f'Resume checkpoint unavailable or incomplete at {actor_path}, fallback to {fallback_actor_path}')
                    actor_path = fallback_actor_path
                    critic_path = fallback_critic_path
                    resume_name = fallback_name
                    use_fallback_checkpoint = True

            # If still no sharded checkpoint, try latest global_step_* HF checkpoint
            if not use_fallback_checkpoint:
                actor_parent = os.path.join(self.config.trainer.default_local_dir, 'actor')
                hf_step, hf_actor_path = self._find_latest_global_step_dir(actor_parent)
                if hf_actor_path is not None:
                    print(f'No sharded checkpoint available, '
                          f'falling back to HF checkpoint: {hf_actor_path}')
                    actor_path = hf_actor_path
                    self.global_steps = hf_step
                    use_hf_checkpoint = True
                else:
                    print(f'Resume checkpoint unavailable or incomplete: {actor_path}')
                    print('Training from scratch')
                    return 0

        if use_fallback_checkpoint and isinstance(resume_name, str) and resume_name.startswith('global_step_'):
            try:
                self.global_steps = int(resume_name.split('_')[-1])
            except ValueError:
                pass

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {actor_path}')

        if use_hf_checkpoint:
            # HF checkpoint: reload model weights via from_pretrained on each worker,
            # optimizer stays freshly initialized, LR scheduler will be fast-forwarded.
            self.actor_rollout_wg.load_hf_checkpoint(actor_path)
            self._resumed_from_hf = True
            print(f'[Resume] Loaded HF checkpoint from {actor_path}')
            print(f'[Resume] Optimizer reset to initial state; '
                  f'LR scheduler will fast-forward {self.global_steps} steps')
        else:
            # Sharded checkpoint: load model + optimizer + LR scheduler
            self.actor_rollout_wg.load_checkpoint(actor_path,
                                                  del_local_after_load=False)
            if self.use_critic:
                self.critic_wg.load_checkpoint(critic_path,
                                               del_local_after_load=False)
        
    def _rebalance_rollouts(self, batch, reward_tensor, gen_batch, n, max_n, metrics):
        return rebalance_rollouts(
            self.actor_rollout_wg, self.reward_fn,
            batch, reward_tensor, gen_batch, n, max_n, metrics)

    def _inspiration_for_hard(self, batch, reward_tensor, gen_batch, n, max_n, metrics,
                              hard_threshold=0.2):
        return inspiration_for_hard(
            self.actor_rollout_wg, self.reward_fn,
            batch, reward_tensor, gen_batch, n, max_n, metrics,
            hard_threshold=hard_threshold,
            tokenizer=self.tokenizer,
            max_prompt_length=int(self.config.data.max_prompt_length))

    def _inspiration_for_hard_memory(self, batch, gen_batch, predicted_labels, n, metrics,
                                     epoch=None, batch_step=None):
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return inspiration_for_hard_memory(
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer,
            replay_buffer=self.replay_buffer,
            batch=batch,
            gen_batch=gen_batch,
            predicted_labels=predicted_labels,
            n=n,
            metrics=metrics,
            hard_threshold=float(rollout_cfg.get('hard_memory_threshold', 0.2)),
            mix_ratio=float(rollout_cfg.get('hard_memory_mix_ratio', 0.5)),
            max_prompt_length=int(self.config.data.max_prompt_length),
            max_snippets=int(rollout_cfg.get('hard_memory_max_snippets', 3)),
            max_chars_per_snippet=int(rollout_cfg.get('hard_memory_max_chars_per_snippet', 160)),
            output_dir=self.config.trainer.default_local_dir,
            epoch=epoch,
            batch_step=batch_step,
            global_step=self.global_steps,
        )

    # def _inspiration_for_easy(self, batch, reward_tensor, gen_batch, n, metrics):
    #     # Legacy implementation path intentionally disabled.
    #     # Easy-prompt handling now controlled by easy_length_penalty_coeff.
    #     return batch, reward_tensor

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _print_attenuation_stats(self):
        print_attenuation_stats(self.attenuation_counts, self.sample_attenuation,
                                self.attenuation_decay_factor)

    def _attenuation_update(self, sample_key, accuracy):
        return attenuation_update(
            self.attenuation_counts, self.sample_attenuation,
            sample_key, accuracy)

    def _predict_difficulty_with_llm(self, all_questions, batch_size=256):
        """Use the current (RL-updated) policy model to predict difficulty.

        Constructs a difficulty-estimation prompt for every question and asks
        the policy model to output a success probability between 0 and 1.
        No reference rollout is required.

        All prompts are packed into a single ``generate_sequences`` call so
        that the vLLM engine only wakes up / sleeps once, avoiding the huge
        overhead of repeated sleep/wake cycles.
        """
        import gc
        from verl.utils.model import compute_position_id_with_mask
        import verl.utils.torch_functional as verl_F

        predicted_labels = torch.full((len(all_questions),), 0.5)
        target_indices = list(range(len(all_questions)))

        difficulty_system = (
            "You are an expert math problem difficulty estimator. "
            "Given a math problem, estimate the probability that it can be "
            "correctly solved. Output ONLY a single decimal number between "
            "0.0 and 1.0, where 0.0 means impossible and 1.0 means trivial. "
            "Do not include any other text."
        )

        world_size = self.actor_rollout_wg.world_size
        max_prompt_length = self.config.data.max_prompt_length

        # ---- tokenize ALL prompts at once ----
        ids_list, mask_list, pos_list = [], [], []
        for idx in target_indices:
            chat = [
                {"role": "system", "content": difficulty_system},
                {"role": "user", "content": all_questions[idx]},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False)
            ids, mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
                tokenizer=self.tokenizer,
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='left')
            pos = compute_position_id_with_mask(mask)
            ids_list.append(ids[0])
            mask_list.append(mask[0])
            pos_list.append(pos[0])

        gen_batch = DataProto.from_single_dict({
            'input_ids': torch.stack(ids_list),
            'attention_mask': torch.stack(mask_list),
            'position_ids': torch.stack(pos_list),
        })
        gen_batch.meta_info['override_n'] = 1
        gen_batch.meta_info['val_temperature'] = 0.3
        gen_batch.meta_info['val_max_tokens'] = 32

        # ---- single generate call (one wake/sleep cycle) ----
        print(f"[LLM_predict] Generating for all {len(target_indices)} "
              f"questions in one call (n=1, max_tokens=32) ...")
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, world_size)
        output_padded = self.actor_rollout_wg.generate_sequences(
            gen_batch_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        # ---- parse results ----
        response_ids = output.batch['responses']
        n_parsed = 0
        for j, idx in enumerate(target_indices):
            resp = response_ids[j]
            resp_tokens = resp[resp != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(
                resp_tokens, skip_special_tokens=True).strip()
            score = self._parse_difficulty_score(text)
            predicted_labels[idx] = score
            if score != 0.5:
                n_parsed += 1

        print(f"[LLM_predict] Done: {len(target_indices)} questions, "
              f"{n_parsed} parsed ok")

        del ids_list, mask_list, pos_list, gen_batch
        gc.collect()
        return predicted_labels

    def _predict_entropy_scores(self, batch_size=256):
        """Estimate raw entropy scores for all prompts via short-prefix generation.

        For each prompt, generate a single short prefix with the current policy,
        then recompute token-level entropy on that generated prefix. The prompt's
        score is the mean entropy across valid generated tokens.
        """
        import gc
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import collate_fn

        dataset_size = len(self.train_dataset)
        entropy_scores = torch.zeros(dataset_size, dtype=torch.float32)
        entropy_prefix_len = int(self.config.data.get('entropy_prefix_len', 64))
        entropy_batch_size = int(self.config.data.get('entropy_batch_size', batch_size))
        entropy_temperature = float(self.config.data.get('entropy_temperature', 0.6))
        entropy_clip_percentile_raw = self.config.data.get('entropy_clip_percentile', None)
        entropy_clip_percentile = None
        if entropy_clip_percentile_raw is not None:
            _clip_str = str(entropy_clip_percentile_raw).strip().strip("'\"")
            if _clip_str:
                entropy_clip_percentile = float(_clip_str)

        entropy_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=entropy_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        world_size = self.actor_rollout_wg.world_size
        total_batches = len(entropy_loader)
        write_offset = 0

        for batch_idx, batch_dict in enumerate(entropy_loader):
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info['override_n'] = 1
            gen_batch.meta_info['val_temperature'] = entropy_temperature
            gen_batch.meta_info['val_max_tokens'] = entropy_prefix_len

            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
            output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            entropy_out = self.actor_rollout_wg.compute_entropy(output)
            token_entropy = entropy_out.batch['old_entropy']
            response_ids = output.batch['responses']
            response_mask = response_ids.ne(self.tokenizer.pad_token_id).float()
            valid_token_count = response_mask.sum(dim=-1).clamp(min=1.0)
            mean_entropy = (token_entropy * response_mask).sum(dim=-1) / valid_token_count

            cur_batch_size = mean_entropy.size(0)
            entropy_scores[write_offset:write_offset + cur_batch_size] = mean_entropy.cpu().float()
            write_offset += cur_batch_size

            print(f"[entropy] Batch {batch_idx + 1}/{total_batches}: "
                  f"{cur_batch_size} prompts scored")

            del batch, gen_batch, gen_batch_padded, output_padded, output, entropy_out
            gc.collect()

        assert write_offset == dataset_size, (
            f"[entropy] scored {write_offset} prompts, expected {dataset_size}")

        if entropy_clip_percentile is not None:
            clip_q = float(np.clip(entropy_clip_percentile, 0.0, 100.0))
            clip_val = float(torch.quantile(entropy_scores, clip_q / 100.0).item())
            entropy_scores = entropy_scores.clamp(max=clip_val)
            print(f"[entropy] Clipped raw scores at p{clip_q:.1f} = {clip_val:.6f}")

        print(f"[entropy] Score stats: min={float(entropy_scores.min()):.6f}, "
              f"max={float(entropy_scores.max()):.6f}, "
              f"mean={float(entropy_scores.mean()):.6f}, "
              f"std={float(entropy_scores.std()):.6f}")
        return entropy_scores

    @staticmethod
    def _parse_difficulty_score(text):
        """Extract a 0-1 float from LLM-generated text."""
        import re
        for m in re.findall(r'\d+\.?\d*', text):
            try:
                val = float(m)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
        return 0.5

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        import traceback
        import sys
        import time
        
        print("Start to train!")

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.enable_coverage_tracking = bool(self.config.trainer.get("enable_coverage_tracking", False))
        self.sample_training_tracker = None
        if self.enable_coverage_tracking:
            self.sample_training_tracker = SampleTrainingTracker(self.config.trainer.default_local_dir)

        _sa_raw = self.config.data.get('sample_attenuation', None)
        self.sample_attenuation = None
        if _sa_raw is not None:
            import ast as _ast
            _sa_str = str(_sa_raw).strip().strip("'\"")
            if _sa_str:
                _sa_parsed = _ast.literal_eval(_sa_str)
                if isinstance(_sa_parsed, (list, tuple)) and len(_sa_parsed) == 2:
                    self.sample_attenuation = (float(_sa_parsed[0]), float(_sa_parsed[1]))
                else:
                    raise ValueError(
                        f"[Attenuation] sample_attenuation must be a [lo, hi] interval, "
                        f"got: {_sa_str}")
        self.attenuation_counts = {}
        self.attenuation_decay_factor = float(self.config.data.get('attenuation_decay_factor', 0.5))
        if self.sample_attenuation is not None:
            _att_path = os.path.join(self.config.trainer.default_local_dir, "attenuation_counts.json")
            if os.path.exists(_att_path):
                with open(_att_path, "r", encoding="utf-8") as f:
                    self.attenuation_counts = {k: int(v) for k, v in json.load(f).items()}
                print(f"[Attenuation] Loaded {len(self.attenuation_counts)} tracked samples")
            self._print_attenuation_stats()

        self._selection_method = self.config.data.get('selection_method', 'teacher')
        self._use_raw_entropy_selection = (self._selection_method == 'entropy')
        if self._selection_method == 'is':
            from verl.trainer.ppo.is_data_selector import ISDataSelector
            is_save_dir = os.path.join(
                self.config.trainer.default_local_dir, 'is_diagnostics')
            _is_def_raw = str(self.config.data.get('is_default_label', '')).strip().strip("'\"")
            _is_default_label = float(_is_def_raw) if _is_def_raw else 0.5
            self.is_selector = ISDataSelector(
                clip_range=float(self.config.data.get('is_clip_range', 5.0)),
                ess_threshold=float(self.config.data.get('is_ess_threshold', 2.0)),
                default_label=_is_default_label,
                save_dir=is_save_dir,
            )
            _is_mode_desc = "teacher fallback" if self.use_teacher else f"fixed={_is_default_label}"
            print(f"[IS Selector] Initialized: clip_range={self.is_selector.clip_range}, "
                  f"ess_threshold={self.is_selector.ess_threshold}, "
                  f"default_label={self.is_selector.default_label} ({_is_mode_desc})")
        elif self._selection_method == 'bayesian':
            from verl.trainer.ppo.bayesian_data_selector import BayesianDataSelector
            _bay_save_dir = os.path.join(
                self.config.trainer.default_local_dir, 'bayesian_diagnostics')
            _bay_alpha0 = float(self.config.data.get('bayesian_alpha0', 1.0))
            _bay_beta0 = float(self.config.data.get('bayesian_beta0', 1.0))
            _bay_decay = float(self.config.data.get('bayesian_decay', 0.5))
            _bay_target = float(self.config.data.get('bayesian_target_gamma', 0.5))
            _bay_def_raw = str(self.config.data.get('bayesian_default_label', '')).strip().strip("'\"")
            _bay_default = float(_bay_def_raw) if _bay_def_raw else 0.5
            self.bayesian_selector = BayesianDataSelector(
                alpha0=_bay_alpha0,
                beta0=_bay_beta0,
                decay=_bay_decay,
                target_gamma=_bay_target,
                default_label=_bay_default,
                save_dir=_bay_save_dir,
            )
            # Try to restore posteriors from a previous run
            _bay_state_path = os.path.join(
                self.config.trainer.default_local_dir, 'bayesian_posteriors.json')
            self.bayesian_selector.load_state(_bay_state_path)
            print(f"[Bayesian Selector] Initialized: "
                  f"α₀={_bay_alpha0}, β₀={_bay_beta0}, "
                  f"λ={_bay_decay}, γ*={_bay_target}, "
                  f"default={_bay_default}")
        elif self._selection_method == 'LLM_predict':
            print(f"[LLM_predict] 使用当前policy model预测难度, "
                  f"不依赖teacher model")
        elif self._selection_method == 'entropy':
            print(f"[entropy] 使用当前policy model的短前缀raw entropy进行全数据选样")
            if self.sample_attenuation is not None:
                print(f"{COLOR_RED}[entropy] sample_attenuation 依赖accuracy语义，"
                      f"raw entropy 选样下将自动跳过其相关逻辑{COLOR_RESET}")
            _rkr_raw = str(self.config.data.get('rollout_keep_range', '')).strip().strip("'\"")
            if _rkr_raw:
                print(f"{COLOR_RED}[entropy] rollout_keep_range 依赖accuracy语义，"
                      f"raw entropy 选样下将自动跳过其相关逻辑{COLOR_RESET}")
        elif self._selection_method == '':
            print(f"{COLOR_RED}[Random Baseline] SELECTION_METHOD为空, "
                  f"跳过所有难度预测, 均匀随机选取 (原始GRPO){COLOR_RESET}")
        elif self._selection_method != 'teacher':
            raise ValueError(
                f"Unknown data.selection_method='{self._selection_method}'. "
                f"Expected 'teacher', 'is', 'bayesian', 'LLM_predict', 'entropy', or '' (random).")

        _isrr_raw = str(self.config.data.get('is_rerollout_ratio', '')).strip().strip("'\"")
        self.is_rerollout_ratio = float(_isrr_raw) if _isrr_raw else None
        if self.is_rerollout_ratio is not None:
            if self._selection_method != 'is':
                print(f"{COLOR_RED}[IS重用] is_rerollout_ratio={self.is_rerollout_ratio} "
                      f"已设置，但 selection_method='{self._selection_method}' 不是 'is'，"
                      f"该参数不会生效{COLOR_RESET}")
                self.is_rerollout_ratio = None
            else:
                print(f"{COLOR_RED}[IS重用] IS选中的prompt中，"
                      f"{self.is_rerollout_ratio:.0%} 将重新rollout，"
                      f"{1 - self.is_rerollout_ratio:.0%} 复用buffer中的rollout{COLOR_RESET}")

        _prt_raw = str(self.config.trainer.get('perf_regression_threshold', '')).strip().strip("'\"")
        self.perf_regression_threshold = float(_prt_raw) if _prt_raw else None
        if self.perf_regression_threshold is not None:
            print(f"{COLOR_RED}#### [性能回退检测] 已启用，阈值: {self.perf_regression_threshold} ####{COLOR_RESET}")
            print(f"{COLOR_RED}#### 每epoch结束后评估，若整体准确率比上一轮低超过 "
                  f"{self.perf_regression_threshold}，将回退模型权重并重训 ####{COLOR_RESET}")
        self._prev_eval_acc = None
        self._epoch_regression_retries = 0
        self._MAX_REGRESSION_RETRIES = 3

        # --- Epoch timing tracker (实时写入JSON) ---
        self._timing_json_path = os.path.join(
            self.config.trainer.default_local_dir, "epoch_timing.json")
        if os.path.exists(self._timing_json_path):
            with open(self._timing_json_path, "r", encoding="utf-8") as f:
                self._epoch_timings = json.load(f)
        else:
            self._epoch_timings = {}

        self._current_epoch_timing_breakdown = None

        self.global_steps = 0
        self._resumed_from_hf = False
        
        self._load_checkpoint()

        # If resumed from HF checkpoint (no optimizer/LR state), fast-forward
        # the LR scheduler so cosine schedule is at the correct position.
        if self._resumed_from_hf and self.global_steps > 0:
            print(f'[Resume] Fast-forwarding LR scheduler by {self.global_steps} steps')
            self.actor_rollout_wg.fast_forward_lr_scheduler(self.global_steps)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        effective_total_steps = self.total_training_steps - self.global_steps
        trained_epochs = self.global_steps // self.config.data.mu
        
        # we start from step 1
        self.global_steps += 1
        
        # Maximum number of retry attempts
        max_retries = self.config.trainer.get('max_retries', 3)
        retry_count = 0
        retry_delay = self.config.trainer.get('retry_delay', 60)  # seconds

        # Load replay buffer from disk if training breaks
        if self.global_steps == 1:
            self.replay_buffer = None
        else:
            replay_buffer_dir = os.path.join(self.config.trainer.default_local_dir, 'replay_buffer.pkl')
            local_latest_buffer_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                   'latest_buffer_iteration.txt')
            if os.path.exists(local_latest_buffer_iteration) and os.path.exists(replay_buffer_dir):
                with open(local_latest_buffer_iteration, 'r') as f:
                    buf_step = int(f.read().strip())
                if self.global_steps == buf_step + 1:
                    self.replay_buffer = DataProto.load_from_disk(replay_buffer_dir)
                    if 'original_accuracy' not in self.replay_buffer.non_tensor_batch:
                        print("[replay_buffer] Backfilling original_accuracy from token_level_scores")
                        backfill_original_accuracy_by_contiguous_groups(
                            self.replay_buffer, _contiguous_groups)
                else:
                    print(f'[Resume] Buffer iteration mismatch '
                          f'(buffer={buf_step}, expected={self.global_steps - 1}), '
                          f'starting with empty replay buffer')
                    self.replay_buffer = None
            else:
                print(f'[Resume] No replay buffer found, starting with empty replay buffer')
                self.replay_buffer = None
        
        while retry_count <= max_retries:
            try:
                with tqdm(total=effective_total_steps, desc="Training Progress") as pbar:
                    epoch = trained_epochs
                    while epoch < self.config.trainer.total_epochs:
                        epoch_raw = {}
                        epoch_metrics = {}
                        # --- Per-epoch timing accumulators ---
                        _epoch_rollout_time = 0.0
                        _epoch_training_time = 0.0
                        _epoch_eval_time = 0.0
                        _epoch_timing_breakdown = defaultdict(float)
                        self._current_epoch_timing_breakdown = _epoch_timing_breakdown

                        if self.perf_regression_threshold is not None:
                            _pre_epoch_global_steps = self.global_steps
                            _rollback_dir = os.path.join(
                                self.config.trainer.default_local_dir, 'rollback')
                            os.makedirs(_rollback_dir, exist_ok=True)
                            _rollback_actor_path = os.path.join(_rollback_dir, 'actor')
                            self.actor_rollout_wg.save_checkpoint(
                                _rollback_actor_path, None, save_checkpoints=True)
                            with open(os.path.join(_rollback_dir, 'global_steps.txt'), 'w') as f:
                                f.write(str(self.global_steps))
                            if self.replay_buffer is not None:
                                self.replay_buffer.save_to_disk(
                                    os.path.join(_rollback_dir, 'replay_buffer.pkl'))
                            if self.sample_attenuation is not None and self.attenuation_counts:
                                with open(os.path.join(_rollback_dir, 'attenuation_counts.json'),
                                          'w', encoding='utf-8') as f:
                                    json.dump(self.attenuation_counts, f, ensure_ascii=False)
                            print(f"{COLOR_RED}#### [回退点] epoch {epoch} 开始前已保存回退检查点 "
                                  f"(global_steps={self.global_steps}) ####{COLOR_RESET}")

                        with _timer('epoch', epoch_raw):
                            with _timer('data_selection', epoch_raw):
                                from torch.utils.data import DataLoader
                                from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
                                # Release previous epoch's dataset to free memory
                                if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                                    del self.train_dataset
                                self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error',
                                        format_reward=self.config.data.get('format_reward', False)
                                        )
                                if self.enable_coverage_tracking:
                                    self.sample_training_tracker.initialize_from_dataframe(
                                        dataframe=self.train_dataset.dataframe,
                                        prompt_key=self.config.data.prompt_key,
                                    )

                                if self._selection_method == '':
                                    # ---- Random baseline (原始GRPO): 跳过所有难度预测 ----
                                    predicted_labels = torch.full((len(self.train_dataset),), 0.5)
                                    ref_indices = None
                                    print(f"{COLOR_RED}[Random Baseline] 均匀随机选取, "
                                          f"无难度预测, 无reference rollout{COLOR_RESET}")

                                elif not self.config.data.random_selection:
                                    print("Not random!")

                                    if self._selection_method == 'LLM_predict':
                                        # ---- LLM_predict: skip reference rollout ----
                                        all_questions = [item['extra_info']['question']
                                                         for item in self.train_dataset]
                                        with _accumulate_time(_epoch_timing_breakdown, 'selection_generation_seconds'):
                                            predicted_labels = self._predict_difficulty_with_llm(
                                                all_questions)
                                        print("LLM Predict Success!")
                                        ref_indices = None
                                        del all_questions
                                        import gc; gc.collect()

                                    elif self._selection_method == 'entropy':
                                        with _accumulate_time(_epoch_timing_breakdown, 'selection_generation_seconds'):
                                            predicted_labels = self._predict_entropy_scores()
                                        ref_indices = None
                                        print("Entropy Prediction Success!")

                                    elif self._selection_method == 'bayesian':
                                        # ---- Bayesian (MoPPS): Thompson Sampling, no reference rollout needed ----
                                        teacher_fallback = None
                                        if self.use_teacher:
                                            all_questions = [item['extra_info']['question'] for item in self.train_dataset]
                                            teacher_fallback = self.teacher_wg.predict(
                                                all_questions, [], [], [],
                                                batch_size=self.config.teacher_model.batch_size)
                                            print(f"[Bayesian+Teacher] Teacher fallback 预测完成, "
                                                  f"共 {len(teacher_fallback)} 题")
                                            del all_questions

                                        predicted_labels = self.bayesian_selector.estimate_difficulty(
                                            dataset_size=len(self.train_dataset),
                                            ref_indices=None,
                                            ref_labels=None,
                                            epoch=epoch,
                                            fallback_labels=teacher_fallback,
                                        )
                                        ref_indices = None
                                        if teacher_fallback is not None:
                                            del teacher_fallback
                                        import gc; gc.collect()
                                        print("Bayesian Prediction Success!")

                                    else:
                                        # ---- teacher / is: reference rollout ----
                                        # When IS mode and epoch > 0, build reference set
                                        # from replay buffer instead of doing a fresh rollout.
                                        _is_buffer_ref = (
                                            self._selection_method == 'is'
                                            and epoch > 0
                                            and self.replay_buffer is not None
                                            and len(self.replay_buffer) > 0
                                        )

                                        if _is_buffer_ref:
                                            # --- IS buffer-sourced reference set ---
                                            _buf_indices = np.array(
                                                self.replay_buffer.non_tensor_batch['index'], dtype=int)
                                            _buf_acc = np.array(
                                                self.replay_buffer.non_tensor_batch['original_accuracy'],
                                                dtype=float)
                                            # 去重，保留每个 prompt 最后出现的位置（最新）
                                            _seen = {}
                                            for _i, _idx in enumerate(_buf_indices):
                                                _seen[int(_idx)] = _i
                                            # 按位置倒序 = 最新优先
                                            _unique_items = sorted(
                                                _seen.items(), key=lambda x: x[1], reverse=True)
                                            _ref_size = min(
                                                self.config.data.ref_size, len(_unique_items))

                                            ref_indices = []
                                            ref_labels = []
                                            ref_questions = []
                                            for _idx, _pos in _unique_items[:_ref_size]:
                                                ref_indices.append(_idx)
                                                ref_labels.append(float(_buf_acc[_pos]))
                                                ref_questions.append(
                                                    self.train_dataset[_idx]['extra_info']['question'])

                                            print(
                                                f"{COLOR_RED}[IS BufferRef] epoch {epoch}: "
                                                f"从 buffer 取 {len(ref_indices)} 个 prompt 作为 "
                                                f"reference set (跳过 rollout), "
                                                f"mean_acc={np.mean(ref_labels):.3f}{COLOR_RESET}")

                                            # 保存 buffer-sourced ref_data 到磁盘
                                            ref_data = {}
                                            for _ri, _rl in zip(ref_indices, ref_labels):
                                                ref_data[_ri] = {
                                                    'question': self.train_dataset[_ri]['extra_info']['question'],
                                                    'rewards': [_rl],
                                                    'source': 'buffer',
                                                }
                                            import pickle
                                            ref_data_save_path = os.path.join(
                                                self.config.trainer.default_local_dir, 'saved_ref_data')
                                            if not os.path.exists(ref_data_save_path):
                                                os.makedirs(ref_data_save_path)
                                            with open(os.path.join(
                                                    ref_data_save_path,
                                                    f'ref_data_epoch_{epoch}.pkl'), 'wb') as f:
                                                pickle.dump(ref_data, f)
                                            print(f"[Ref] Saved {len(ref_data)} buffer-sourced items "
                                                  f"to {ref_data_save_path}")

                                        else:
                                            # --- Normal reference rollout (epoch 0, or teacher mode) ---
                                            ref_indices = random.sample(range(len(self.train_dataset)), self.config.data.ref_size)
                                            ref_dataset = torch.utils.data.Subset(self.train_dataset, indices=ref_indices)  

                                            _ref_batch_size = min(self.config.data.train_batch_size, len(ref_indices))
                                            assert len(ref_dataset) % _ref_batch_size == 0
                                            ref_dataloader = DataLoader(dataset=ref_dataset,
                                                    batch_size=_ref_batch_size,
                                                    shuffle=False, 
                                                    drop_last=False,
                                                    collate_fn=collate_fn)
                                            
                                            # Collect reference samples, assign labels, and store them for later use
                                            ref_batches = []
                                            ref_solve_none = 0
                                            ref_solve_all = 0
                                            with _accumulate_time(_epoch_timing_breakdown, 'reference_rollout_seconds'):
                                                for _, batch_dict in enumerate(tqdm(ref_dataloader, desc="Reference Rollout")):
                                                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                                                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                                            dtype=object)
                                                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                                    batch = batch.union(gen_batch_output)
                                                    if self.use_critic:
                                                        values = self.critic_wg.compute_values(batch)
                                                        batch = batch.union(values)
                                                    if self.use_rm:
                                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                                        batch = batch.union(reward_tensor)
                                                    reward_tensor = self.reward_fn(batch)
                                                    batch.batch['token_level_scores'] = reward_tensor
                                                    _ref_solve_stats = compute_ref_solve_none_all(batch, reward_tensor)
                                                    ref_solve_none += _ref_solve_stats[0]
                                                    ref_solve_all += _ref_solve_stats[1]
                                                    ref_batches.append(batch)
                                                
                                            epoch_metrics['epoch/ref_solve_none'] = ref_solve_none
                                            epoch_metrics['epoch/ref_solve_all'] = ref_solve_all
                                            
                                            ref_batches = DataProto.concat(ref_batches)
                                            ref_data = defaultdict(dict)
                                            for _, batch in enumerate(ref_batches):
                                                key = batch.non_tensor_batch['index']
                                                ref_data[key]['question'] = batch.non_tensor_batch['extra_info']['question']
                                                ref_data[key]['rewards'] = ref_data[key].get('rewards', []) + [batch.batch['token_level_scores'].sum(-1).item()]
                                            
                                            ref_questions = []
                                            ref_labels = []
                                            ref_indices = []
                                            for key in ref_data:
                                                ref_questions.append(ref_data[key]['question'])
                                                ref_labels.append(np.mean(ref_data[key]['rewards']))
                                                ref_indices.append(key)
                                            
                                            if (not self._use_raw_entropy_selection
                                                    and self.sample_attenuation is not None):
                                                _newly_att_ref = 0
                                                for _rk, _rl in zip(ref_indices, ref_labels):
                                                    _rk_str = str(_rk)
                                                    if self._attenuation_update(_rk_str, float(_rl)):
                                                        _newly_att_ref += 1
                                                if _newly_att_ref > 0:
                                                    print(f"[Attenuation] {_newly_att_ref} updated from reference rollout")
                                                epoch_metrics['attenuation/newly_attenuated_ref'] = _newly_att_ref

                                            import pickle
                                            ref_data_save_path = os.path.join(self.config.trainer.default_local_dir,'saved_ref_data')
                                            if not os.path.exists(ref_data_save_path):
                                                os.makedirs(ref_data_save_path)
                                            with open(os.path.join(ref_data_save_path,
                                                            f'ref_data_epoch_{epoch}.pkl'), 'wb') as f:
                                                pickle.dump(ref_data, f)
                                            print(f"[Ref] Saved {len(ref_data)} items to {ref_data_save_path}")

                                        # Predict adaptive difficulty / success rate
                                        if self._selection_method == 'teacher':
                                            all_questions = [item['extra_info']['question'] for item in self.train_dataset]
                                            predicted_labels = self.teacher_wg.predict(all_questions, ref_questions, ref_labels, ref_indices, batch_size=self.config.teacher_model.batch_size)
                                            print("Teacher Prediction Success!")
                                            del ref_batches, ref_data, all_questions
                                            import gc; gc.collect()
                                        elif self._selection_method == 'is':
                                            teacher_fallback = None
                                            if self.use_teacher:
                                                all_questions = [item['extra_info']['question'] for item in self.train_dataset]
                                                teacher_fallback = self.teacher_wg.predict(
                                                    all_questions, ref_questions, ref_labels, ref_indices,
                                                    batch_size=self.config.teacher_model.batch_size)
                                                print(f"[IS+Teacher] Teacher fallback 预测完成, "
                                                      f"共 {len(teacher_fallback)} 题")
                                                del all_questions

                                            predicted_labels = self.is_selector.estimate_difficulty(
                                                replay_buffer=self.replay_buffer,
                                                compute_log_prob_fn=self.actor_rollout_wg.compute_log_prob,
                                                dataset_size=len(self.train_dataset),
                                                ref_indices=ref_indices,
                                                ref_labels=ref_labels,
                                                epoch=epoch,
                                                fallback_labels=teacher_fallback,
                                            )
                                            print("IS Prediction Success!")
                                            if not _is_buffer_ref:
                                                del ref_batches
                                            del ref_data
                                            if teacher_fallback is not None:
                                                del teacher_fallback
                                            import gc; gc.collect()
                                else:
                                    predicted_labels = torch.rand(len(self.train_dataset))
                                    print("Random!")
                                    
                                predicted_label_save_path = os.path.join(self.config.trainer.default_local_dir, 'saved_predicted_labels')
                                if not os.path.exists(predicted_label_save_path):
                                    os.makedirs(predicted_label_save_path)
                                torch.save(predicted_labels, os.path.join(predicted_label_save_path, f'predicted_labels_epoch_{epoch}.pt'))
                                print(f"[Predicted Labels] Saved {len(predicted_labels)} labels to {predicted_label_save_path}")

                                if epoch != 0:
                                    selection_budget = int(self.config.data.mu * self.config.data.train_batch_size * self.config.data.sigma)
                                else:
                                    selection_budget = int(self.config.data.mu * self.config.data.train_batch_size)

                                if self._selection_method == '':
                                    # ---- Random baseline: 均匀随机选取，不使用任何难度分布 ----
                                    selected_indices = torch.randperm(len(self.train_dataset))[:selection_budget]
                                else:
                                    # Get selected subset
                                    # --- 处理 Bayesian NaN 标记: 未覆盖的题分配探索概率 ---
                                    _nan_mask = torch.isnan(predicted_labels)
                                    _n_nan = int(_nan_mask.sum())
                                    if _n_nan > 0:
                                        # 用 default_label 填充 NaN，后续会给这些题降权
                                        _bay_fill = float(self.config.data.get('bayesian_default_label', '').strip().strip("'\"") or '0.5')
                                        predicted_labels[_nan_mask] = _bay_fill
                                        print(f"[Bayesian] {_n_nan}/{len(predicted_labels)} 题无后验, "
                                              f"填充={_bay_fill}, 将在logit中降权")

                                    if self._use_raw_entropy_selection:
                                        _entropy_tau = float(self.config.data.get('entropy_tau', 1.0))
                                        dataset_sampling_logits = predicted_labels / max(_entropy_tau, 1e-8)
                                        if epoch == 0:
                                            print(f"{COLOR_RED}[SampleDist] Raw entropy softmax: "
                                                  f"tau={_entropy_tau}{COLOR_RESET}")
                                    else:
                                        _dist_type = self.config.data.get('sample_dist_type', 'laplace')
                                        if _dist_type == 'beta':
                                            _beta_peak = float(self.config.data.get('beta_peak', 0.5))
                                            _beta_kappa = float(self.config.data.get('beta_kappa', 20.0))
                                            _a_param = _beta_peak * _beta_kappa + 1.0
                                            _b_param = (1.0 - _beta_peak) * _beta_kappa + 1.0
                                            _x_clamped = predicted_labels.clamp(1e-8, 1.0 - 1.0e-8)
                                            dataset_sampling_logits = (
                                                (_a_param - 1.0) * torch.log(_x_clamped)
                                                + (_b_param - 1.0) * torch.log(1.0 - _x_clamped)
                                            )
                                            if epoch == 0:
                                                print(f"{COLOR_RED}[SampleDist] Beta distribution: "
                                                      f"peak={_beta_peak}, kappa={_beta_kappa}, "
                                                      f"a={_a_param:.2f}, b={_b_param:.2f}{COLOR_RESET}")
                                        else:
                                            dataset_sampling_scores = -torch.abs(predicted_labels - self.config.data.alpha)
                                            dataset_sampling_logits = dataset_sampling_scores / self.config.data.tau
                                            if epoch == 0:
                                                print(f"{COLOR_RED}[SampleDist] Laplace distribution: "
                                                      f"alpha={self.config.data.alpha}, "
                                                      f"tau={self.config.data.tau}{COLOR_RESET}")

                                    # --- Bayesian: 对无后验的题在 logit 中降权 ---
                                    if _n_nan > 0 and not self._use_raw_entropy_selection:
                                        _covered_logits = dataset_sampling_logits[~_nan_mask]
                                        if len(_covered_logits) > 0:
                                            # 用可配置的探索比例控制未覆盖题的采样概率
                                            # explore_ratio=0.3 表示未覆盖题的总采样概率质量
                                            # 占有后验题总概率质量的 30%
                                            _explore_ratio = float(self.config.data.get(
                                                'bayesian_explore_ratio', 0.3))
                                            import torch.nn.functional as _F
                                            _covered_probs = _F.softmax(_covered_logits, dim=0)
                                            _covered_total_mass = float(_covered_probs.sum())
                                            # 目标: nan题均匀分享 explore_ratio 的概率质量
                                            # p_each_nan = (explore_ratio * covered_mass) / n_nan
                                            # logit_nan = log(p_each_nan) + logsumexp(covered_logits)
                                            import math
                                            _target_nan_mass = _explore_ratio * _covered_total_mass
                                            _p_each_nan = _target_nan_mass / _n_nan
                                            _logsumexp_covered = float(torch.logsumexp(_covered_logits, dim=0))
                                            _explore_logit = math.log(max(_p_each_nan, 1e-30)) + _logsumexp_covered
                                            dataset_sampling_logits[_nan_mask] = _explore_logit
                                            # 验证实际比例
                                            _all_probs = _F.softmax(dataset_sampling_logits, dim=0)
                                            _actual_nan_ratio = float(_all_probs[_nan_mask].sum())
                                            print(f"[Bayesian] 探索配置: explore_ratio={_explore_ratio:.0%}, "
                                                  f"实际未覆盖题概率质量={_actual_nan_ratio:.1%} "
                                                  f"({_n_nan} 题), "
                                                  f"有后验题概率质量={1-_actual_nan_ratio:.1%} "
                                                  f"({int((~_nan_mask).sum())} 题)")
                                        else:
                                            print(f"[Bayesian] Epoch {epoch}: 全部 {_n_nan} 题无后验, "
                                                  f"保持均匀采样 (logit 不变)")

                                    if (not self._use_raw_entropy_selection
                                            and self.sample_attenuation is not None
                                            and self.attenuation_counts):
                                        _n_att = 0
                                        for _i in range(len(self.train_dataset)):
                                            _skey = str(self.train_dataset[_i]['extra_info']['index'])
                                            _cnt = self.attenuation_counts.get(_skey, 0)
                                            if _cnt > 0:
                                                dataset_sampling_logits[_i] += _cnt * math.log(self.attenuation_decay_factor)
                                                _n_att += 1
                                        if _n_att > 0:
                                            print(f"[Attenuation] Applied multiplicative decay to "
                                                  f"{_n_att}/{len(self.train_dataset)} samples")

                                    if self.use_teacher and self._selection_method == 'is':
                                        _is_set = getattr(self.is_selector, 'last_is_indices', set())
                                        _ref_set = set(int(ri) for ri in ref_indices) if ref_indices is not None else set()
                                        _teacher_mask = torch.ones(len(self.train_dataset), dtype=torch.bool)
                                        for _idx in _is_set:
                                            _teacher_mask[_idx] = False
                                        for _idx in _ref_set:
                                            _teacher_mask[_idx] = False
                                        _non_teacher_mask = ~_teacher_mask
                                        _n_teacher_candidates = int(_teacher_mask.sum())
                                        _n_non_teacher_candidates = int(_non_teacher_mask.sum())

                                        epoch_metrics['teacher_balance/is_candidates'] = len(_is_set)
                                        epoch_metrics['teacher_balance/ref_candidates'] = len(_ref_set)
                                        epoch_metrics['teacher_balance/teacher_candidates'] = _n_teacher_candidates
                                        epoch_metrics['teacher_balance/non_teacher_candidates'] = _n_non_teacher_candidates

                                        if _n_teacher_candidates > 0 and _n_non_teacher_candidates > 0:
                                            _selection_budget_safe = max(int(selection_budget), 1)
                                            _coverage_ratio = _n_non_teacher_candidates / _selection_budget_safe
                                            _dyn_power = float(self.config.data.get(
                                                'teacher_dynamic_power', 2.0))
                                            _target_teacher_share = 1.0 / (
                                                1.0 + (_coverage_ratio ** _dyn_power))
                                            _target_teacher_share = float(np.clip(
                                                _target_teacher_share, 1e-4, 1.0 - 1e-4))
                                            _max_abs_shift = float(self.config.data.get(
                                                'teacher_dynamic_max_logit_shift', 12.0))

                                            _log_mass_teacher = torch.logsumexp(
                                                dataset_sampling_logits[_teacher_mask], dim=0)
                                            _log_mass_non_teacher = torch.logsumexp(
                                                dataset_sampling_logits[_non_teacher_mask], dim=0)
                                            _current_teacher_share = float(torch.exp(
                                                _log_mass_teacher - torch.logaddexp(
                                                    _log_mass_teacher, _log_mass_non_teacher)).item())
                                            _current_log_odds = float((
                                                _log_mass_teacher - _log_mass_non_teacher).item())
                                            _target_log_odds = float(
                                                np.log(_target_teacher_share)
                                                - np.log(1.0 - _target_teacher_share))
                                            _teacher_logit_shift = float(np.clip(
                                                _target_log_odds - _current_log_odds,
                                                -_max_abs_shift, _max_abs_shift))

                                            dataset_sampling_logits[_teacher_mask] += _teacher_logit_shift

                                            _post_log_mass_teacher = torch.logsumexp(
                                                dataset_sampling_logits[_teacher_mask], dim=0)
                                            _post_log_mass_non_teacher = torch.logsumexp(
                                                dataset_sampling_logits[_non_teacher_mask], dim=0)
                                            _post_teacher_share = float(torch.exp(
                                                _post_log_mass_teacher - torch.logaddexp(
                                                    _post_log_mass_teacher,
                                                    _post_log_mass_non_teacher)).item())

                                            epoch_metrics['teacher_balance/non_teacher_per_budget'] = _coverage_ratio
                                            epoch_metrics['teacher_balance/dynamic_power'] = _dyn_power
                                            epoch_metrics['teacher_balance/current_teacher_mass_share'] = _current_teacher_share
                                            epoch_metrics['teacher_balance/target_teacher_mass_share'] = _target_teacher_share
                                            epoch_metrics['teacher_balance/post_teacher_mass_share'] = _post_teacher_share
                                            epoch_metrics['teacher_balance/teacher_logit_shift'] = _teacher_logit_shift

                                            print(f"{COLOR_RED}[Teacher动态平衡] "
                                                  f"IS={len(_is_set)}, Ref={len(_ref_set)}, Teacher={_n_teacher_candidates}; "
                                                  f"非Teacher候选/预算={_coverage_ratio:.3f}, "
                                                  f"Teacher质量占比 {_current_teacher_share:.3f} -> {_post_teacher_share:.3f} "
                                                  f"(target={_target_teacher_share:.3f}, shift={_teacher_logit_shift:+.3f})"
                                                  f"{COLOR_RESET}")
                                        else:
                                            _selection_budget_safe = max(int(selection_budget), 1)
                                            _coverage_ratio = _n_non_teacher_candidates / _selection_budget_safe
                                            epoch_metrics['teacher_balance/non_teacher_per_budget'] = _coverage_ratio
                                            epoch_metrics['teacher_balance/dynamic_power'] = float(
                                                self.config.data.get('teacher_dynamic_power', 2.0))
                                            epoch_metrics['teacher_balance/current_teacher_mass_share'] = (
                                                1.0 if _n_teacher_candidates > 0 else 0.0)
                                            epoch_metrics['teacher_balance/target_teacher_mass_share'] = (
                                                1.0 if _n_teacher_candidates > 0 else 0.0)
                                            epoch_metrics['teacher_balance/post_teacher_mass_share'] = (
                                                1.0 if _n_teacher_candidates > 0 else 0.0)
                                            epoch_metrics['teacher_balance/teacher_logit_shift'] = 0.0
                                            print(f"{COLOR_RED}[Teacher动态平衡] 跳过缩放: "
                                                  f"IS={len(_is_set)}, Ref={len(_ref_set)}, Teacher={_n_teacher_candidates}"
                                                  f"{COLOR_RESET}")

                                    dataset_sampling_logits -= dataset_sampling_logits.max()
                                    dataset_sampling_probabilities = torch.softmax(dataset_sampling_logits, dim=0)
                                    selected_indices = torch.multinomial(
                                        dataset_sampling_probabilities, selection_budget, replacement=False)

                                selected_indices = selected_indices[torch.randperm(len(selected_indices))]

                                selected_indices_save_path = os.path.join(self.config.trainer.default_local_dir, 'saved_selected_indices')
                                if not os.path.exists(selected_indices_save_path):
                                    os.makedirs(selected_indices_save_path)
                                torch.save(selected_indices, os.path.join(selected_indices_save_path, f'selected_indices_epoch_{epoch}.pt'))
                                print(f"[Selected Indices] Saved {len(selected_indices)} indices to {selected_indices_save_path}")

                                use_dataset = torch.utils.data.Subset(self.train_dataset, indices=selected_indices.tolist())  
                                use_train_dataloader = DataLoader(dataset=use_dataset,
                                                            batch_size=len(selected_indices) // self.config.data.mu, # replay note
                                                            shuffle=False, 
                                                            drop_last=False,
                                                            collate_fn=collate_fn)
                                assert len(use_train_dataloader) == self.config.data.mu

                            with _timer('Rollout_update', epoch_raw):
                                for batch_step, batch_dict in enumerate(tqdm(use_train_dataloader, desc="Rollout_update")):
                                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                                    metrics = {}

                                    n_gen = self.config.actor_rollout_ref.rollout.n
                                    generation_type_raw = self.config.actor_rollout_ref.rollout.get(
                                        'generation_type', '')
                                    # Support comma-separated multi-select, e.g. "inspiration_for_hard,rebalance"
                                    _gt_str = str(generation_type_raw).strip().strip("'\"")
                                    generation_types = set(
                                        t.strip() for t in _gt_str.split(',') if t.strip()
                                    )
                                    if self._use_raw_entropy_selection and 'reallocation' in generation_types:
                                        generation_types.remove('reallocation')
                                        if not getattr(self, '_entropy_reallocation_warned', False):
                                            print(f"{COLOR_RED}[entropy] generation_type='reallocation' 依赖accuracy语义，"
                                                  f"在 raw entropy 选样下已自动禁用{COLOR_RESET}")
                                            self._entropy_reallocation_warned = True
                                    if self._use_raw_entropy_selection and 'inspiration_for_hard_memory' in generation_types:
                                        generation_types.remove('inspiration_for_hard_memory')
                                        if not getattr(self, '_entropy_hard_memory_warned', False):
                                            print(f"{COLOR_RED}[entropy] generation_type='inspiration_for_hard_memory' 依赖accuracy语义，"
                                                  f"在 raw entropy 选样下已自动禁用{COLOR_RESET}")
                                            self._entropy_hard_memory_warned = True

                                    pre_alloc = None
                                    is_buffer_plan = {}
                                    if 'reallocation' in generation_types:
                                        batch, pre_alloc, n_pre_skipped = pre_rollout_difficulty_filter(
                                            batch, predicted_labels, n_gen,
                                            keep_range_str=self.config.data.get('rollout_keep_range', None))
                                        metrics['pre_reallocation/n_skipped'] = n_pre_skipped
                                        metrics['pre_reallocation/n_kept'] = len(batch)

                                        if len(batch) == 0:
                                            print(f"{COLOR_RED}[pre_reallocation] "
                                                  f"所有prompt均被过滤，跳过此batch{COLOR_RESET}")
                                            continue

                                        world_size = self.actor_rollout_wg.world_size
                                        remainder = len(batch) % world_size
                                        if remainder != 0:
                                            trim_size = len(batch) - remainder
                                            batch = dataprotoitem_to_dataproto(batch[:trim_size])
                                            pre_alloc = {k: v for k, v in pre_alloc.items()
                                                         if k < trim_size}
                                            print(f"[pre_reallocation] Trimmed "
                                                  f"{trim_size + remainder} -> {trim_size} "
                                                  f"for divisibility (world_size={world_size})")

                                        if (pre_alloc
                                                and self.is_rerollout_ratio is not None
                                                and epoch > 0
                                                and self.replay_buffer is not None
                                                and len(self.replay_buffer) > 0):
                                            _is_obj = getattr(self, 'is_selector', None)
                                            _is_idx_set = (
                                                getattr(_is_obj, 'last_is_indices', set())
                                                if _is_obj else set())
                                            if _is_idx_set:
                                                _buf_idx = np.array(
                                                    self.replay_buffer.non_tensor_batch['index'],
                                                    dtype=int)
                                                _batch_indices = batch.non_tensor_batch['index']
                                                _total_saved = 0
                                                for _pp in list(pre_alloc.keys()):
                                                    _didx = int(_batch_indices[_pp])
                                                    if _didx not in _is_idx_set:
                                                        continue
                                                    _bm = (_buf_idx == _didx)
                                                    _nba = int(_bm.sum())
                                                    if _nba == 0:
                                                        continue
                                                    _alloc = pre_alloc[_pp]
                                                    _desired_fresh = max(1, round(
                                                        self.is_rerollout_ratio * _alloc))
                                                    _from_buf = min(
                                                        _alloc - _desired_fresh, _nba)
                                                    _actual_fresh = _alloc - _from_buf
                                                    if _from_buf > 0:
                                                        _buf_pos = np.where(_bm)[0]
                                                        _sel = random.sample(
                                                            list(_buf_pos), _from_buf)
                                                        pre_alloc[_pp] = _actual_fresh
                                                        is_buffer_plan[_pp] = {
                                                            'total_budget': _alloc,
                                                            'actual_fresh': _actual_fresh,
                                                            'buffer_positions': _sel,
                                                            'dataset_idx': _didx,
                                                        }
                                                        _total_saved += _alloc - _actual_fresh
                                                if is_buffer_plan:
                                                    print(
                                                        f"{COLOR_RED}[IS预分配] "
                                                        f"{len(is_buffer_plan)} 个IS选中prompt"
                                                        f"从buffer复用, "
                                                        f"节省 {_total_saved} 次rollout生成"
                                                        f"{COLOR_RESET}")
                                                    metrics['is_prealloc/n_prompts'] = (
                                                        len(is_buffer_plan))
                                                    metrics['is_prealloc/n_saved'] = _total_saved

                                    # pop those keys for generation
                                    start_time = time.time()
                                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                                    if 'reallocation' in generation_types and pre_alloc is not None:
                                        batch, gen_batch = grouped_generate(
                                            self.actor_rollout_wg, batch, gen_batch,
                                            pre_alloc, metrics)
                                    elif 'inspiration_for_hard_memory' in generation_types:
                                        batch, gen_batch = self._inspiration_for_hard_memory(
                                            batch=batch,
                                            gen_batch=gen_batch,
                                            predicted_labels=predicted_labels,
                                            n=n_gen,
                                            metrics=metrics,
                                            epoch=epoch,
                                            batch_step=batch_step,
                                        )
                                    else:
                                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                        batch.non_tensor_batch['uid'] = np.array(
                                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                            dtype=object)
                                        batch = batch.repeat(
                                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                                            interleave=True)
                                        batch = batch.union(gen_batch_output)

                                    elapsed_time = time.time() - start_time
                                    metrics['gen_rollout'] = elapsed_time
                                    _epoch_rollout_time += elapsed_time
                                    _epoch_timing_breakdown['main_rollout_seconds'] += elapsed_time
                                    print(f"[gen_rollout] Time elapsed: {elapsed_time:.3f} seconds")

                                    # compute values
                                    if self.use_critic:
                                        values = self.critic_wg.compute_values(batch)
                                        batch = batch.union(values)

                                    # compute scores using reward model and/or reward function
                                    if self.use_rm:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                        batch = batch.union(reward_tensor)

                                    reward_tensor = self.reward_fn(batch)
                                    batch.batch['token_level_scores'] = reward_tensor

                                    # Store original accuracy (before rebalancing) for
                                    # priority-based replay buffer selection.
                                    _orig_acc = assign_original_accuracy_by_uid(batch, reward_tensor)
                                    _uids_pre = batch.non_tensor_batch['uid']
                                    _seen_pre = {}
                                    for _u in _uids_pre:
                                        if _u not in _seen_pre:
                                            _seen_pre[_u] = len(_seen_pre)

                                    log_step_diagnostics(
                                        batch=batch, orig_acc=_orig_acc,
                                        group_size=self.config.actor_rollout_ref.rollout.n,
                                        batch_step=batch_step, epoch=epoch,
                                        selection_method=self._selection_method,
                                        is_random_selection=self.config.data.random_selection or self._selection_method == '',
                                        use_teacher=self.use_teacher,
                                        output_dir=self.config.trainer.default_local_dir,
                                        ref_indices=ref_indices if (not self.config.data.random_selection and self._selection_method != '') else None,
                                        predicted_labels=predicted_labels,
                                        is_selector=getattr(self, 'is_selector', None),
                                        is_default_label_str=str(self.config.data.get('is_default_label', '')),
                                    )

                                    # --- Bayesian posterior update with rollout feedback ---
                                    if self._selection_method == 'bayesian':
                                        _bay_batch_indices = batch.non_tensor_batch['index']
                                        _bay_uids = batch.non_tensor_batch['uid']
                                        _bay_rewards_per_prompt = defaultdict(list)
                                        _bay_token_scores = batch.batch['token_level_scores']
                                        for _bi in range(len(batch)):
                                            _didx = int(_bay_batch_indices[_bi])
                                            _r = float(_bay_token_scores[_bi].sum().item())
                                            _bay_rewards_per_prompt[_didx].append(_r)

                                        # --- 诊断: 预测精度 vs 实际精度 ---
                                        _bay_pred_vals = []
                                        _bay_actual_vals = []
                                        _bay_abs_errors = []
                                        for _didx, _rewards in _bay_rewards_per_prompt.items():
                                            _actual_acc = sum(1 for r in _rewards if r > 0) / max(len(_rewards), 1)
                                            _bay_actual_vals.append(_actual_acc)
                                            if predicted_labels is not None and _didx < len(predicted_labels):
                                                _pred_acc = float(predicted_labels[_didx])
                                                _bay_pred_vals.append(_pred_acc)
                                                _bay_abs_errors.append(abs(_pred_acc - _actual_acc))

                                        if _bay_abs_errors:
                                            import numpy as _np
                                            _mae = float(_np.mean(_bay_abs_errors))
                                            _rmse = float(_np.sqrt(_np.mean(_np.array(_bay_abs_errors) ** 2)))
                                            _corr = float(_np.corrcoef(_bay_pred_vals, _bay_actual_vals)[0, 1]) if len(_bay_pred_vals) > 1 else 0.0
                                            _pred_mean = float(_np.mean(_bay_pred_vals))
                                            _actual_mean = float(_np.mean(_bay_actual_vals))
                                            print(f"\033[92m[Bayesian 诊断] 预测vs实际: "
                                                  f"MAE={_mae:.4f}, RMSE={_rmse:.4f}, "
                                                  f"Corr={_corr:.4f} | "
                                                  f"pred_mean={_pred_mean:.4f}, "
                                                  f"actual_mean={_actual_mean:.4f} | "
                                                  f"n={len(_bay_abs_errors)}\033[0m")
                                            epoch_metrics['bayesian/mae'] = _mae
                                            epoch_metrics['bayesian/rmse'] = _rmse
                                            epoch_metrics['bayesian/correlation'] = _corr
                                            epoch_metrics['bayesian/pred_mean'] = _pred_mean
                                            epoch_metrics['bayesian/actual_mean'] = _actual_mean

                                            # 按难度区间统计预测精度
                                            from verl.trainer.ppo.bayesian_data_selector import _ascii_histogram
                                            _ascii_histogram(
                                                _bay_abs_errors,
                                                f"Bayesian 预测误差分布 (|pred-actual|)",
                                                n_bins=20, width=55, color="\033[92m")
                                            _ascii_histogram(
                                                _bay_actual_vals,
                                                f"Rollout 实际正确率分布",
                                                bins=[0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625,
                                                      0.6875, 0.8125, 0.9375, 1.001],
                                                bin_labels=["0/8+1/16", "1/8±1/16", "2/8±1/16",
                                                            "3/8±1/16", "4/8±1/16", "5/8±1/16",
                                                            "6/8±1/16", "7/8±1/16", "8/8-1/16"],
                                                width=55, color="\033[96m")

                                        self.bayesian_selector.update_posteriors(
                                            rollout_indices=list(_bay_rewards_per_prompt.keys()),
                                            rollout_rewards_per_prompt=_bay_rewards_per_prompt,
                                            group_size=self.config.actor_rollout_ref.rollout.n,
                                        )
                                        # Persist posteriors for resume
                                        _bay_state_path = os.path.join(
                                            self.config.trainer.default_local_dir,
                                            'bayesian_posteriors.json')
                                        self.bayesian_selector.save_state(_bay_state_path)

                                        # 保存诊断到 JSON
                                        if _bay_abs_errors:
                                            _bay_diag_dir = os.path.join(
                                                self.config.trainer.default_local_dir,
                                                'bayesian_diagnostics')
                                            os.makedirs(_bay_diag_dir, exist_ok=True)
                                            _bay_rollout_diag = {
                                                'epoch': epoch,
                                                'n_prompts': len(_bay_rewards_per_prompt),
                                                'mae': _mae, 'rmse': _rmse,
                                                'correlation': _corr,
                                                'pred_mean': _pred_mean,
                                                'actual_mean': _actual_mean,
                                                'n_posteriors': len(self.bayesian_selector.posteriors),
                                            }
                                            _bay_diag_path = os.path.join(
                                                _bay_diag_dir,
                                                f'rollout_diagnostics_epoch_{epoch}.json')
                                            with open(_bay_diag_path, 'w') as f:
                                                json.dump(_bay_rollout_diag, f, indent=2)

                                    if (not self._use_raw_entropy_selection
                                            and self.sample_attenuation is not None):
                                        _batch_indices = batch.non_tensor_batch['index']
                                        _newly_att_roll = 0
                                        for _u in _seen_pre:
                                            _m = _uids_pre == _u
                                            _first = np.where(_m)[0][0]
                                            _sidx = str(_batch_indices[_first])
                                            _sacc = float(_orig_acc[_first])
                                            if self._attenuation_update(_sidx, _sacc):
                                                _newly_att_roll += 1
                                        if _newly_att_roll > 0:
                                            print(f"[Attenuation] {_newly_att_roll} updated from rollout")
                                        self._print_attenuation_stats()
                                        metrics['attenuation/newly_attenuated_rollout'] = _newly_att_roll
                                        metrics['attenuation/n_tracked'] = len(self.attenuation_counts)
                                        metrics['attenuation/n_available'] = (
                                            len(self.train_dataset) - len(self.attenuation_counts))

                                    # Post-rollout generation strategies
                                    max_n_gen = self.config.actor_rollout_ref.rollout.get(
                                        'max_n', n_gen)

                                    if 'inspiration_for_hard_memory' in generation_types:
                                        max_n_gen = n_gen

                                    if 'rebalance' in generation_types and max_n_gen > n_gen:
                                        rb_start = time.time()
                                        batch, reward_tensor = self._rebalance_rollouts(
                                            batch, reward_tensor, gen_batch,
                                            n_gen, max_n_gen, metrics)
                                        rb_elapsed = time.time() - rb_start
                                        metrics['rebalance_time'] = rb_elapsed
                                        _epoch_timing_breakdown['aux_rollout_seconds'] += rb_elapsed
                                        print(f"[rebalance_rollouts] Time elapsed: "
                                              f"{rb_elapsed:.3f} seconds")
                                    if 'inspiration_for_hard' in generation_types \
                                            and max_n_gen > n_gen:
                                        rb_start = time.time()
                                        batch, reward_tensor = self._inspiration_for_hard(
                                            batch, reward_tensor, gen_batch,
                                            n_gen, max_n_gen, metrics)
                                        rb_elapsed = time.time() - rb_start
                                        metrics['inspiration_hard_time'] = rb_elapsed
                                        _epoch_timing_breakdown['aux_rollout_seconds'] += rb_elapsed
                                        print(f"[inspiration_for_hard] Time elapsed: "
                                              f"{rb_elapsed:.3f} seconds")
                                    # reallocation: 已在 grouped_generate 中直接按预算生成，无需后处理

                                    # Filter prompts whose actual rollout accuracy is outside
                                    # rollout_keep_range (applies to ALL generation types).
                                    rollout_keep_range_str = self.config.data.get(
                                        'rollout_keep_range', None)
                                    if self._use_raw_entropy_selection:
                                        rollout_keep_range_str = None
                                    _keep_range_stats = None
                                    if rollout_keep_range_str is not None:
                                        batch, reward_tensor, _keep_range_stats = post_rollout_keep_range_filter(
                                            batch, reward_tensor, rollout_keep_range_str)
                                    if _keep_range_stats is not None:
                                        if _keep_range_stats['n_filtered'] > 0:
                                            print(
                                                f"[rollout_keep_range] Filtered {_keep_range_stats['n_filtered']}/"
                                                f"{_keep_range_stats['n_prompts_before']} prompts "
                                                f"(accuracy outside [{_keep_range_stats['keep_lo']}, {_keep_range_stats['keep_hi']}]), "
                                                f"batch: {_keep_range_stats['response_count_before']} -> {_keep_range_stats['response_count_after']} responses")
                                        else:
                                            print(
                                                f"[rollout_keep_range] All "
                                                f"{_keep_range_stats['n_prompts_before']} prompts within "
                                                f"[{_keep_range_stats['keep_lo']}, {_keep_range_stats['keep_hi']}], no filtering needed")
                                        metrics['filter/n_filtered_by_keep_range'] = _keep_range_stats['n_filtered']
                                        metrics['filter/n_kept_by_keep_range'] = _keep_range_stats['n_kept']

                                    world_size = self.actor_rollout_wg.world_size
                                    remainder = len(batch) % world_size
                                    if remainder != 0:
                                        trim_size = len(batch) - remainder
                                        batch = dataprotoitem_to_dataproto(batch[:trim_size])
                                        reward_tensor = reward_tensor[:trim_size]
                                        batch.batch['token_level_scores'] = reward_tensor
                                        print(f"[divisibility_trim] Trimmed batch "
                                              f"{trim_size + remainder} -> {trim_size} "
                                              f"(world_size={world_size})")

                                    metrics.update(compute_solve_none_all(batch, reward_tensor))
                                    
                                    start_time = time.time()
                                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                    batch = batch.union(old_log_prob)
                                    elapsed_time = time.time() - start_time
                                    metrics['compute_log_prob'] = elapsed_time
                                    print(f"[compute_log_prob] Time elapsed: {elapsed_time:.3f} seconds")

                                    # ---- IS pre-alloc merge: combine buffer entries ----
                                    # Fresh batch now has old_log_probs (current policy).
                                    # Buffer entries keep their stored behavior-policy
                                    # old_log_probs.  Assign the same uid so GRPO groups
                                    # them together, and sort by index for contiguity so
                                    # buffer dedup keeps the combined group as one unit.
                                    if is_buffer_plan:
                                        _fresh_idx = batch.non_tensor_batch['index']
                                        _fresh_uids = batch.non_tensor_batch['uid']
                                        _idx2uid = {}
                                        for _i in range(len(batch)):
                                            _di = int(_fresh_idx[_i])
                                            if _di not in _idx2uid:
                                                _idx2uid[_di] = _fresh_uids[_i]

                                        _buf_parts = []
                                        _n_combined = 0
                                        for _pp, _plan in is_buffer_plan.items():
                                            _didx = _plan['dataset_idx']
                                            _uid = _idx2uid.get(_didx)
                                            if _uid is None:
                                                continue
                                            _be = dataprotoitem_to_dataproto(
                                                self.replay_buffer[_plan['buffer_positions']])
                                            _be.non_tensor_batch['uid'] = np.array(
                                                [_uid] * len(_be), dtype=object)
                                            _buf_parts.append(_be)
                                            _n_combined += len(_be)

                                        if _buf_parts:
                                            _before = len(batch)
                                            for _bp in _buf_parts:
                                                batch = DataProto.concat([batch, _bp])
                                            _all_idx = np.array(
                                                batch.non_tensor_batch['index'], dtype=int)
                                            _sort_order = np.argsort(
                                                _all_idx, kind='stable')
                                            batch = dataprotoitem_to_dataproto(
                                                batch[_sort_order.tolist()])
                                            reward_tensor = batch.batch['token_level_scores']
                                            print(
                                                f"{COLOR_RED}[IS合并] "
                                                f"合并 {_n_combined} 条buffer数据到 "
                                                f"{len(is_buffer_plan)} 个prompt, "
                                                f"batch: {_before} -> {len(batch)}"
                                                f"{COLOR_RESET}")
                                            metrics['is_combine/n_entries'] = _n_combined
                                    # ---- end IS pre-alloc merge ----

                                    # Replay
                                    if epoch == 0:
                                        if self.replay_buffer is None:
                                            self.replay_buffer = dataprotoitem_to_dataproto(batch[:])
                                        else:
                                            self.replay_buffer, _ = select_buffer(
                                                self.replay_buffer, predicted_labels, self.config)
                                            if self.replay_buffer is None:
                                                self.replay_buffer = dataprotoitem_to_dataproto(batch[:])
                                            else:
                                                self.replay_buffer = DataProto.concat([self.replay_buffer, batch])
                                    else:
                                        print("Replaying...")
                                        
                                        start_time = time.time()
                                        if self.replay_buffer is None:
                                            # Buffer lost after HF-checkpoint resume;
                                            # treat as epoch-0: seed buffer, no replay
                                            print("[Resume] replay_buffer is None, "
                                                  "seeding buffer (no replay this step)")
                                            self.replay_buffer = dataprotoitem_to_dataproto(batch[:])
                                            replay_batch = None
                                        else:
                                            self.replay_buffer, replay_batch = select_buffer(
                                                self.replay_buffer, predicted_labels, self.config) 

                                            # Now add current batch to buffer
                                            if self.replay_buffer is None:
                                                print("Buffer Cleared!")
                                                self.replay_buffer = dataprotoitem_to_dataproto(batch[:])
                                            else:
                                                self.replay_buffer = DataProto.concat([self.replay_buffer, batch])
                                        elapsed_time = time.time() - start_time
                                        metrics['select_buffer'] = elapsed_time
                                        print(f"[select_buffer] Time elapsed: {elapsed_time:.3f} seconds")

                                        # Compute overlap before concat so it reflects
                                        # fresh-vs-replay overlap (not IS-sourced data)
                                        if replay_batch is not None:
                                            num_overlap_replay = len(np.intersect1d(batch.non_tensor_batch['index'], replay_batch.non_tensor_batch['index']))
                                            metrics['replay_batch/num_overlap'] = num_overlap_replay
                                            batch = DataProto.concat([batch, replay_batch])
                                        else:
                                            metrics['replay_batch/num_overlap'] = 0

                                    # --- Diagnostic: full training batch accuracy ---
                                    if epoch > 0 and 'original_accuracy' in batch.non_tensor_batch:
                                        _diag_idx = batch.non_tensor_batch['index']
                                        _diag_seen = {}
                                        _diag_accs_list = []
                                        for _di, _didx in enumerate(_diag_idx):
                                            if _didx not in _diag_seen:
                                                _diag_seen[_didx] = True
                                                _diag_accs_list.append(float(
                                                    batch.non_tensor_batch['original_accuracy'][_di]))
                                        _diag_full_accs = np.array(_diag_accs_list)
                                        _diag_full_n = len(_diag_full_accs)
                                        print(
                                            f"{COLOR_RED}[Step {batch_step} 训练Batch] "
                                            f"{_diag_full_n} 题 (fresh+replay): "
                                            f"mean={_diag_full_accs.mean():.3f}, "
                                            f"全对={int((_diag_full_accs == 1.0).sum())}, "
                                            f"全错={int((_diag_full_accs == 0.0).sum())}, "
                                            f"有效={int(((_diag_full_accs > 0) & (_diag_full_accs < 1.0)).sum())}{COLOR_RESET}")
                                    
                                    # ensure batch is divisible by world_size after replay concat
                                    world_size = self.actor_rollout_wg.world_size
                                    remainder = len(batch) % world_size
                                    if remainder != 0:
                                        trim_size = len(batch) - remainder
                                        batch = dataprotoitem_to_dataproto(batch[:trim_size])
                                        print(f"[divisibility_trim] Trimmed batch "
                                              f"{trim_size + remainder} -> {trim_size} "
                                              f"(world_size={world_size})")

                                    # compute behavior log prob
                                    if self.config.actor_rollout_ref.actor.use_temp_log_prob:
                                        temp_log_prob = self.actor_rollout_wg.compute_temp_log_prob(batch)
                                        batch = batch.union(temp_log_prob) 

                                    if self.enable_coverage_tracking:
                                        self.sample_training_tracker.increment_from_batch_indices(batch.non_tensor_batch['index'])
                                        self.sample_training_tracker.save_state()
                                    
                                    # Save rollout data to disk (batch decode for efficiency)
                                    import pickle
                                    data_save_path = os.path.join(self.config.trainer.default_local_dir,'saved_data')
                                    os.makedirs(data_save_path, exist_ok=True)
                                    
                                    _all_indices = batch.non_tensor_batch['index']
                                    _all_responses = self.tokenizer.batch_decode(
                                        batch.batch['responses'], skip_special_tokens=True)
                                    _all_scores = batch.batch['token_level_scores'].sum(-1).tolist()
                                    _all_shaped_rewards = batch.batch['token_level_rewards'].sum(-1).tolist() if 'token_level_rewards' in batch.batch else list(_all_scores)
                                    
                                    saved_data = defaultdict(dict)
                                    for _i in range(len(batch)):
                                        key = _all_indices[_i]
                                        saved_data[key]['question'] = batch.non_tensor_batch['extra_info'][_i]['question']
                                        saved_data[key].setdefault('responses', []).append(_all_responses[_i])
                                        saved_data[key].setdefault('rewards', []).append(_all_scores[_i])
                                        saved_data[key].setdefault('shaped_rewards', []).append(_all_shaped_rewards[_i])
                                    del _all_responses, _all_scores, _all_shaped_rewards, _all_indices
                                    
                                    with open(os.path.join(data_save_path,
                                                    f'rollout_data_step_{self.global_steps}.pkl'), 'wb') as f:
                                        pickle.dump(saved_data, f)
                                    del saved_data

                                    if self.use_reference_policy:
                                        # compute reference log_prob
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)

                                    # compute rewards with KL penalty if needed

                                    # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                                    # where it is subtracted directly from the policy loss

                                    # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                    #     batch, kl_metrics = apply_kl_penalty(batch,
                                    #                                        kl_ctrl=self.kl_ctrl,
                                    #                                        kl_penalty=self.config.algorithm.kl_penalty)
                                    #     metrics.update(kl_metrics)
                                    # else:
                                    #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores'].clone()

                                    # --- Easy-prompt length penalty (controlled by coeff, no generation_type gate) ---
                                    _easy_threshold = float(self.config.actor_rollout_ref.rollout.get(
                                        'easy_threshold', 0.9))
                                    _easy_lp_coeff = float(self.config.actor_rollout_ref.actor.get(
                                        'easy_length_penalty_coeff', 0.0))
                                    batch.meta_info['easy_length_penalty_coeff'] = _easy_lp_coeff

                                    _easy_penalty_stats = apply_easy_length_reward_shaping(
                                        batch=batch,
                                        easy_threshold=_easy_threshold,
                                        length_penalty_coeff=_easy_lp_coeff)

                                    metrics['easy_penalty/easy_threshold'] = _easy_threshold
                                    metrics['easy_penalty/coeff'] = _easy_lp_coeff
                                    metrics['easy_penalty/enabled'] = float(_easy_penalty_stats['enabled'])
                                    metrics['easy_penalty/easy_prompts'] = _easy_penalty_stats['easy_prompts']
                                    metrics['easy_penalty/applied_prompts'] = _easy_penalty_stats['applied_prompts']
                                    metrics['easy_penalty/easy_responses'] = _easy_penalty_stats['easy_responses']
                                    metrics['easy_penalty/applied_responses'] = _easy_penalty_stats['applied_responses']
                                    metrics['easy_penalty/avg_easy_weight'] = _easy_penalty_stats['avg_easy_weight']
                                    metrics['easy_penalty/avg_applied_length_norm'] = _easy_penalty_stats['avg_applied_length_norm']
                                    metrics['easy_penalty/avg_applied_response_len'] = _easy_penalty_stats['avg_applied_response_len']
                                    metrics['easy_penalty/avg_penalty'] = _easy_penalty_stats['avg_penalty']
                                    metrics['easy_penalty/max_penalty'] = _easy_penalty_stats['max_penalty']
                                    metrics['easy_penalty/total_penalty'] = _easy_penalty_stats['total_penalty']
                                    metrics['easy_penalty/reward_delta_mean'] = _easy_penalty_stats['reward_delta_mean']
                                    metrics['easy_penalty/applied_clip_ratio'] = _easy_penalty_stats['applied_clip_ratio']

                                    if _easy_lp_coeff > 0:
                                        report_easy_penalty_stats(
                                            _easy_penalty_stats,
                                            easy_threshold=_easy_threshold,
                                            length_penalty_coeff=_easy_lp_coeff,
                                        )
                                        write_easy_penalty_diagnostics(
                                            batch=batch,
                                            stats=_easy_penalty_stats,
                                            output_dir=self.config.trainer.default_local_dir,
                                            epoch=epoch,
                                            batch_step=batch_step,
                                            global_step=self.global_steps,
                                            generation_types=generation_types,
                                            easy_threshold=_easy_threshold,
                                        )

                                    # --- Hard-prompt length bonus (controlled by coeff, no generation_type gate) ---
                                    _hard_len_threshold = float(self.config.actor_rollout_ref.rollout.get(
                                        'hard_length_threshold', 0.2))
                                    _hard_lb_coeff = float(self.config.actor_rollout_ref.actor.get(
                                        'hard_length_bonus_coeff', 0.0))

                                    _hard_len_stats = apply_hard_length_reward_shaping(
                                        batch=batch,
                                        hard_length_threshold=_hard_len_threshold,
                                        length_bonus_coeff=_hard_lb_coeff)

                                    metrics['hard_length_bonus/hard_length_threshold'] = _hard_len_threshold
                                    metrics['hard_length_bonus/coeff'] = _hard_lb_coeff
                                    metrics['hard_length_bonus/enabled'] = float(_hard_len_stats['enabled'])
                                    metrics['hard_length_bonus/hard_prompts'] = _hard_len_stats['hard_prompts']
                                    metrics['hard_length_bonus/applied_prompts'] = _hard_len_stats['applied_prompts']
                                    metrics['hard_length_bonus/hard_responses'] = _hard_len_stats['hard_responses']
                                    metrics['hard_length_bonus/applied_responses'] = _hard_len_stats['applied_responses']
                                    metrics['hard_length_bonus/avg_hard_weight'] = _hard_len_stats['avg_hard_weight']
                                    metrics['hard_length_bonus/avg_applied_length_norm'] = _hard_len_stats['avg_applied_length_norm']
                                    metrics['hard_length_bonus/avg_applied_response_len'] = _hard_len_stats['avg_applied_response_len']
                                    metrics['hard_length_bonus/avg_bonus'] = _hard_len_stats['avg_bonus']
                                    metrics['hard_length_bonus/max_bonus'] = _hard_len_stats['max_bonus']
                                    metrics['hard_length_bonus/total_bonus'] = _hard_len_stats['total_bonus']
                                    metrics['hard_length_bonus/reward_delta_mean'] = _hard_len_stats['reward_delta_mean']

                                    if _hard_lb_coeff > 0:
                                        print(
                                            f"{COLOR_RED}[难题长度奖励] status={_hard_len_stats['status']}, "
                                            f"hard prompts={_hard_len_stats['hard_prompts']}/{_hard_len_stats['total_prompts']}, "
                                            f"applied prompts={_hard_len_stats['applied_prompts']}/{_hard_len_stats['total_prompts']}, "
                                            f"applied responses={_hard_len_stats['applied_responses']}/{_hard_len_stats['total_responses']}, "
                                            f"avg bonus={_hard_len_stats['avg_bonus']:.4f}, "
                                            f"max bonus={_hard_len_stats['max_bonus']:.4f}, "
                                            f"avg len={_hard_len_stats['avg_applied_response_len']:.2f}, "
                                            f"reward Δmean={_hard_len_stats['reward_delta_mean']:.4f}, "
                                            f"HARD_LEN_THRESHOLD={_hard_len_threshold}, "
                                            f"HARD_LENGTH_BONUS_COEFF={_hard_lb_coeff}{COLOR_RESET}"
                                        )

                                    # compute advantages, executed on the driver process
                                    batch = compute_advantage(batch,
                                                            adv_estimator=self.config.algorithm.adv_estimator,
                                                            gamma=self.config.algorithm.gamma,
                                                            lam=self.config.algorithm.lam,
                                                            num_repeat=self.config.actor_rollout_ref.rollout.n)

                                    if _easy_lp_coeff > 0:
                                        _valid_mask = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:].bool()
                                        _valid_adv = torch.masked_select(batch.batch['advantages'], _valid_mask)
                                        if _valid_adv.numel() > 0:
                                            metrics['easy_penalty/advantages_mean'] = float(_valid_adv.mean().detach().item())
                                            metrics['easy_penalty/advantages_min'] = float(_valid_adv.min().detach().item())
                                            metrics['easy_penalty/advantages_max'] = float(_valid_adv.max().detach().item())

                                    if _hard_lb_coeff > 0:
                                        _valid_mask_hl = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:].bool()
                                        _valid_adv_hl = torch.masked_select(batch.batch['advantages'], _valid_mask_hl)
                                        if _valid_adv_hl.numel() > 0:
                                            metrics['hard_length_bonus/advantages_mean'] = float(_valid_adv_hl.mean().detach().item())
                                            metrics['hard_length_bonus/advantages_min'] = float(_valid_adv_hl.min().detach().item())
                                            metrics['hard_length_bonus/advantages_max'] = float(_valid_adv_hl.max().detach().item())

                                    # balance the number of valid tokens on each dp rank.
                                    # Note that this breaks the order of data inside the batch.
                                    # Please take care when you implement group based adv computation such as GRPO and rloo
                                    self._balance_batch(batch, metrics=metrics)

                                    # compute global_valid tokens
                                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                                    start_time = time.time()
                                    # update critic
                                    if self.use_critic:
                                        critic_output = self.critic_wg.update_critic(batch)
                                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                        metrics.update(critic_output_metrics)

                                    # implement critic warmup
                                    if self.config.trainer.critic_warmup <= self.global_steps:
                                        # update actor
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                        metrics.update(actor_output_metrics)

                                    elapsed_time = time.time() - start_time
                                    metrics['actor_update'] = elapsed_time
                                    _epoch_training_time += elapsed_time
                                    print(f"[actor_update] Time elapsed: {elapsed_time:.3f} seconds")


                                    # validate
                                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                        self.global_steps % self.config.trainer.test_freq == 0:
                                        val_metrics: dict = self._validate()
                                        metrics.update(val_metrics)

                                    if self.config.trainer.save_freq > 0 and \
                                            self.global_steps % self.config.trainer.save_freq == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=False)
                                        # Also persist replay buffer so HF-only resume works
                                        if self.replay_buffer is not None:
                                            replay_buffer_dir = os.path.join(self.config.trainer.default_local_dir, 'replay_buffer.pkl')
                                            self.replay_buffer.save_to_disk(replay_buffer_dir)
                                            latest_buffer_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                       'latest_buffer_iteration.txt')
                                            with open(latest_buffer_iteration, 'w') as f:
                                                f.write(str(self.global_steps))
                                    
                                    final_ckpt_freq = self.config.trainer.get('save_final_checkpoint_freq', -1)
                                    if final_ckpt_freq > 0 and \
                                            self.global_steps % final_ckpt_freq == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=True)

                                        replay_buffer_dir = os.path.join(self.config.trainer.default_local_dir, 'replay_buffer.pkl')
                                        self.replay_buffer.save_to_disk(replay_buffer_dir)
                                        
                                        latest_buffer_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                   'latest_buffer_iteration.txt')
                                        with open(latest_buffer_iteration, 'w') as f:
                                            f.write(str(self.global_steps))
                                        print(f"[Replay] Saved {len(self.replay_buffer)} items to {replay_buffer_dir}")

                                    # collect metrics
                                    metrics.update(compute_step_metrics(batch=batch, use_critic=self.use_critic))

                                    # --- new code: decode prompts and responses ---
                                    # Only decode the small sample we actually log to avoid
                                    # allocating thousands of large decoded strings
                                    max_examples_to_log = 30
                                    n_to_decode = min(len(batch), max_examples_to_log)
                                    rewards = batch.batch['token_level_rewards'].max(-1).values

                                    logged_prompts = self.tokenizer.batch_decode(
                                        batch.batch['prompts'][:n_to_decode], skip_special_tokens=True)
                                    logged_responses = self.tokenizer.batch_decode(
                                        batch.batch['responses'][:n_to_decode], skip_special_tokens=True)
                                    logged_rewards = [rewards[i].item() for i in range(n_to_decode)]

                                    if 'wandb' in logger.logger.keys():
                                        import pandas as pd
                                        import wandb
                                        table = {
                                                    "step": [str(self.global_steps)] * len(logged_prompts),
                                                    "prompt": logged_prompts,
                                                    "completion": logged_responses,
                                                    "reward": logged_rewards,
                                                }
                                        df = pd.DataFrame(table)
                                        logger.log(data={"completions": wandb.Table(dataframe=df)}, step=self.global_steps, backend='wandb')

                                    logger.log(data=metrics, step=self.global_steps)

                                    # Free batch and step temporaries
                                    del batch, metrics
                                    import gc
                                    gc.collect()

                                    self.global_steps += 1
                                    pbar.update(1)

                            epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                            if self.enable_coverage_tracking:
                                self.sample_training_tracker.save_state()
                                histogram_path = self.sample_training_tracker.plot_epoch_histogram(epoch=epoch)
                                print(f"[Coverage] Saved epoch histogram to {histogram_path}")
                            if self.sample_attenuation is not None:
                                _att_save_path = os.path.join(
                                    self.config.trainer.default_local_dir,
                                    "attenuation_counts.json")
                                with open(_att_save_path, "w", encoding="utf-8") as f:
                                    json.dump(self.attenuation_counts, f, ensure_ascii=False)
                                epoch_metrics['attenuation/total_tracked'] = len(self.attenuation_counts)
                                print(f"[Attenuation] Saved state (epoch {epoch}):")
                                self._print_attenuation_stats()

                            # Plot rollout success-rate distribution every 10 epochs and on the last epoch.
                            _should_plot_rollout = ((epoch + 1) % 10 == 0) or (epoch == self.config.trainer.total_epochs - 1)
                            if _should_plot_rollout:
                                try:
                                    import sys as _sys
                                    _script_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__), "..", "..", "..", ".."))
                                    if _script_dir not in _sys.path:
                                        _sys.path.insert(0, _script_dir)
                                    from plot_rollout_success_rate import plot_rollout_success_rate_distribution

                                    plot_rollout_success_rate_distribution(
                                        output_dir=self.config.trainer.default_local_dir,
                                        interval=10,
                                    )
                                except Exception as plot_err:
                                    print(f"[Plot] Rollout success-rate plotting failed: {plot_err}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"[Plot] Skipped epoch {epoch} (run every 10 epochs; last epoch always runs)")

                            # Evaluation every 3 epochs and always on the last epoch
                            _eval_success = False
                            _eval_overall_acc = None
                            _should_run_eval = ((epoch + 1) % 3 == 0) or (epoch == self.config.trainer.total_epochs - 1)
                            _epoch_eval_time = 0.0
                            if _should_run_eval:
                                _eval_start_time = time.time()
                                try:
                                    import sys as _sys
                                    _eval_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__), "..", "..", "..", ".."))
                                    if _eval_dir not in _sys.path:
                                        _sys.path.insert(0, _eval_dir)
                                    from eval_math500 import evaluate_math500_epoch
                                    _eval_ds_raw = str(self.config.trainer.get(
                                        'evaluate_dataset', '')).strip().strip("'\"")
                                    _eval_ds_list = (
                                        [d.strip() for d in _eval_ds_raw.split(',') if d.strip()]
                                        if _eval_ds_raw else None
                                    )
                                    overall_acc, dataset_accs = evaluate_math500_epoch(
                                        actor_rollout_wg=self.actor_rollout_wg,
                                        tokenizer=self.tokenizer,
                                        epoch=epoch,
                                        total_epochs=self.config.trainer.total_epochs,
                                        output_dir=os.path.join(
                                            self.config.trainer.default_local_dir, "plots"
                                        ),
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        val_temperature=self.config.actor_rollout_ref.rollout.val_temperature,
                                        batch_size=64,
                                        evaluate_datasets=_eval_ds_list,
                                        gsm8k_sample_size=400,
                                    )
                                    eval_metrics = {"eval/overall_acc": overall_acc}
                                    for ds_name, ds_acc in dataset_accs.items():
                                        eval_metrics[f"eval/{ds_name}_acc"] = ds_acc
                                    _eval_success = True
                                    _eval_overall_acc = overall_acc
                                except Exception as eval_err:
                                    print(f"[Eval] Evaluation failed: {eval_err}")
                                    import traceback
                                    traceback.print_exc()
                                _epoch_eval_time = time.time() - _eval_start_time
                            else:
                                print(f"[Eval] Skipped epoch {epoch} (run every 3 epochs; last epoch always runs)")

                            # --- 实时写入 epoch timing JSON ---
                            _epoch_key = str(epoch)
                            _data_selection_time = float(epoch_raw.get('data_selection', 0.0))
                            _selection_generation_time = float(_epoch_timing_breakdown.get('selection_generation_seconds', 0.0))
                            _reference_rollout_time = float(_epoch_timing_breakdown.get('reference_rollout_seconds', 0.0))
                            _main_rollout_time = float(_epoch_timing_breakdown.get('main_rollout_seconds', 0.0))
                            _aux_rollout_time = float(_epoch_timing_breakdown.get('aux_rollout_seconds', 0.0))
                            _rollout_total_time = (
                                _selection_generation_time
                                + _reference_rollout_time
                                + _main_rollout_time
                                + _aux_rollout_time
                            )
                            _epoch_total_time = float(epoch_raw.get('epoch', 0.0))
                            self._epoch_timings[_epoch_key] = {
                                "rollout_seconds": round(_rollout_total_time, 2),
                                "training_seconds": round(_epoch_training_time, 2),
                                "evaluation_seconds": round(_epoch_eval_time, 2),
                                "data_selection_seconds": round(_data_selection_time, 2),
                                "selection_generation_seconds": round(_selection_generation_time, 2),
                                "reference_rollout_seconds": round(_reference_rollout_time, 2),
                                "main_rollout_seconds": round(_main_rollout_time, 2),
                                "aux_rollout_seconds": round(_aux_rollout_time, 2),
                                "epoch_total_seconds": round(_epoch_total_time, 2),
                                "non_rollout_selection_seconds": round(max(_data_selection_time - _selection_generation_time - _reference_rollout_time, 0.0), 2),
                                "selection_method": str(self._selection_method),
                            }
                            # --- 显存 & 内存记录 (所有方法通用) ---
                            # GPU VRAM: 从 worker 端收集 (driver 无 CUDA 上下文)
                            try:
                                _gpu_stats_list = self.actor_rollout_wg.collect_gpu_memory_stats()
                                _valid = [s for s in _gpu_stats_list if isinstance(s, dict) and s.get("valid")]
                                if _valid:
                                    _peak_vram = round(max(s["peak_gb"] for s in _valid), 2)
                                    _avg_vram = round(sum(s["avg_gb"] for s in _valid) / len(_valid), 2)
                                    self._epoch_timings[_epoch_key]["peak_gpu_memory_gb"] = _peak_vram
                                    self._epoch_timings[_epoch_key]["avg_gpu_memory_gb"] = _avg_vram
                                    print(f"[Memory] Epoch {epoch} VRAM: "
                                          f"peak={_peak_vram:.2f} GB, "
                                          f"avg={_avg_vram:.2f} GB "
                                          f"({len(_valid)} GPUs)")
                                else:
                                    print("[Memory] 无有效 GPU 显存数据")
                            except Exception as _mem_err:
                                print(f"[Memory] 显存统计失败: {_mem_err}")
                            try:
                                import resource, psutil
                                _peak_ram_gb = round(
                                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 2)
                                _cur_ram_gb = round(
                                    psutil.Process().memory_info().rss / (1024 ** 3), 2)
                                self._epoch_timings[_epoch_key]["peak_ram_gb"] = _peak_ram_gb
                                self._epoch_timings[_epoch_key]["avg_ram_gb"] = _cur_ram_gb
                                print(f"[Memory] Epoch {epoch} RAM: "
                                      f"peak={_peak_ram_gb:.2f} GB, "
                                      f"current={_cur_ram_gb:.2f} GB")
                            except Exception as _ram_err:
                                print(f"[Memory] RAM 统计失败: {_ram_err}")
                            with open(self._timing_json_path, "w", encoding="utf-8") as f:
                                json.dump(self._epoch_timings, f, indent=2, ensure_ascii=False)
                            print(f"[Timing] Epoch {epoch}: "
                                  f"rollout_total={_rollout_total_time:.1f}s "
                                  f"(selection_gen={_selection_generation_time:.1f}s, "
                                  f"ref={_reference_rollout_time:.1f}s, "
                                  f"main={_main_rollout_time:.1f}s, "
                                  f"aux={_aux_rollout_time:.1f}s), "
                                  f"data_selection={_data_selection_time:.1f}s, "
                                  f"training={_epoch_training_time:.1f}s, "
                                  f"evaluation={_epoch_eval_time:.1f}s, "
                                  f"epoch_total={_epoch_total_time:.1f}s")

                            self._current_epoch_timing_breakdown = None

                            # --- 性能回退检测 ---
                            _regression_detected = False
                            if (self.perf_regression_threshold is not None
                                    and _eval_success
                                    and self._prev_eval_acc is not None):
                                _drop = self._prev_eval_acc - _eval_overall_acc
                                if _drop > self.perf_regression_threshold:
                                    _regression_detected = True
                                    self._epoch_regression_retries += 1
                                    print(f"{COLOR_RED}#### [性能回退检测] epoch {epoch}: "
                                          f"当前准确率 {_eval_overall_acc:.4f} 比上一轮 "
                                          f"{self._prev_eval_acc:.4f} 低 {_drop:.4f} > "
                                          f"阈值 {self.perf_regression_threshold} ####{COLOR_RESET}")

                                    if self._epoch_regression_retries <= self._MAX_REGRESSION_RETRIES:
                                        print(f"{COLOR_RED}#### [性能回退] 正在回退模型权重至 epoch {epoch} "
                                              f"开始前状态 (重试 {self._epoch_regression_retries}/"
                                              f"{self._MAX_REGRESSION_RETRIES}) ####{COLOR_RESET}")
                                        _rollback_dir = os.path.join(
                                            self.config.trainer.default_local_dir, 'rollback')
                                        _rollback_actor_path = os.path.join(
                                            _rollback_dir, 'actor')
                                        self.actor_rollout_wg.load_checkpoint(
                                            _rollback_actor_path, del_local_after_load=False)
                                        with open(os.path.join(
                                                _rollback_dir, 'global_steps.txt'), 'r') as f:
                                            self.global_steps = int(f.read().strip())
                                        _rollback_buf_path = os.path.join(
                                            _rollback_dir, 'replay_buffer.pkl')
                                        if os.path.exists(_rollback_buf_path):
                                            self.replay_buffer = DataProto.load_from_disk(
                                                _rollback_buf_path)
                                        if self.sample_attenuation is not None:
                                            _rollback_att_path = os.path.join(
                                                _rollback_dir, 'attenuation_counts.json')
                                            if os.path.exists(_rollback_att_path):
                                                with open(_rollback_att_path, 'r',
                                                          encoding='utf-8') as f:
                                                    self.attenuation_counts = {
                                                        k: int(v) for k, v
                                                        in json.load(f).items()}
                                        _ckpt_iter_path = os.path.join(
                                            self.config.trainer.default_local_dir,
                                            'latest_checkpointed_iteration.txt')
                                        with open(_ckpt_iter_path, 'w') as f:
                                            f.write(str(self.global_steps))
                                        _buf_iter_path = os.path.join(
                                            self.config.trainer.default_local_dir,
                                            'latest_buffer_iteration.txt')
                                        if os.path.exists(_buf_iter_path):
                                            with open(_buf_iter_path, 'w') as f:
                                                f.write(str(self.global_steps))
                                        print(f"{COLOR_RED}#### [性能回退] 已恢复至 "
                                              f"global_steps={self.global_steps}，"
                                              f"即将重新训练 epoch {epoch} ####{COLOR_RESET}")
                                        continue
                                    else:
                                        print(f"{COLOR_RED}#### [性能回退] epoch {epoch} 已达最大重试次数 "
                                              f"{self._MAX_REGRESSION_RETRIES}，跳过回退继续训练 "
                                              f"####{COLOR_RESET}")

                            if not _regression_detected:
                                self._epoch_regression_retries = 0

                            logger.log(data=epoch_metrics, step=self.global_steps-1)
                            if _eval_success:
                                logger.log(data=eval_metrics, step=self.global_steps - 1)
                                self._prev_eval_acc = _eval_overall_acc

                            if self.global_steps > self.total_training_steps:
                                self.global_steps -= 1
                                with _timer('save_final_checkpoint', epoch_raw):
                                    self._save_checkpoint(save_checkpoints=True)
                                print("Training completed successfully!")
                                return

                        epoch += 1

                # If we reach here, training completed successfully
                self.global_steps -= 1
                with _timer('save_final_checkpoint', epoch_raw):
                    self._save_checkpoint(save_checkpoints=True)
                print("Training completed successfully!")

                epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                logger.log(data=epoch_metrics, step=self.global_steps)
                return
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                logger.log(data={"error": error_msg}, step=self.global_steps)
                
                # Save the current state before exiting
                print(f"Saving emergency checkpoint at step {self.global_steps}")
                try:
                    self._save_checkpoint(save_checkpoints=True, save_optimizer=False)
                except Exception as save_err:
                    print(f"WARNING: Emergency checkpoint save failed: {save_err}")
                    print("Continuing without saving (FSDP state may be inconsistent after OOM).")
                
                if retry_count <= max_retries:
                    print(f"Attempting to resume training in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    
                    # Reload the checkpoint to resume from where we left off
                    try:
                        self._load_checkpoint()
                        print(f"Resumed training from step {self.global_steps}")
                    except Exception as load_err:
                        print(f"WARNING: Checkpoint reload failed: {load_err}")
                        print("Cannot recover from OOM — FSDP state is inconsistent. Exiting.")
                        raise e from load_err
                else:
                    print(f"Maximum retry attempts ({max_retries}) reached. Exiting.")
                    raise
