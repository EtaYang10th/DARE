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

import json
import os

import numpy as np

from verl.trainer.ppo.rollout_method import COLOR_RED, COLOR_RESET


def _append_jsonl(path, entry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def report_easy_penalty_stats(stats, easy_threshold, length_penalty_coeff):
    if length_penalty_coeff <= 0:
        return
    if stats['status'] == 'missing_original_accuracy':
        print(f"{COLOR_RED}[简单题长度惩罚] WARNING: 缺少 original_accuracy，跳过 reward shaping{COLOR_RESET}")
        return
    print(
        f"{COLOR_RED}[简单题长度惩罚] status={stats['status']}, "
        f"easy prompts={stats['easy_prompts']}/{stats['total_prompts']}, "
        f"applied prompts={stats['applied_prompts']}/{stats['total_prompts']}, "
        f"easy responses={stats['easy_responses']}/{stats['total_responses']}, "
        f"applied responses={stats['applied_responses']}/{stats['total_responses']}, "
        f"avg penalty={stats['avg_penalty']:.4f}, "
        f"max penalty={stats['max_penalty']:.4f}, "
        f"avg len={stats['avg_applied_response_len']:.2f}, "
        f"avg len norm={stats['avg_applied_length_norm']:.4f}, "
        f"reward Δmean={stats['reward_delta_mean']:.4f}, "
        f"EASY_THRESHOLD={easy_threshold}, "
        f"LENGTH_PENALTY_COEFF={length_penalty_coeff}{COLOR_RESET}"
    )


def write_easy_penalty_diagnostics(batch, stats, output_dir, epoch, batch_step, global_step, generation_types, easy_threshold):
    easy_debug_dir = os.path.join(output_dir, 'easy_penalty')
    easy_debug_path = os.path.join(
        easy_debug_dir, 'easy_penalty_step_diagnostics.jsonl')
    debug_entry = {
        'epoch': int(epoch),
        'step': int(batch_step),
        'global_step': int(global_step),
        'generation_types': sorted(list(generation_types)),
        'stats': stats,
    }
    if stats['applied_responses'] > 0:
        response_length = int(batch.batch['responses'].shape[-1])
        response_mask = batch.batch['attention_mask'][:, -response_length:].float()
        response_len = response_mask.sum(dim=-1).detach().cpu().numpy()
        raw_seq_reward = batch.batch['token_level_scores'].sum(-1).detach().cpu().numpy()
        shaped_seq_reward = batch.batch['token_level_rewards'].sum(-1).detach().cpu().numpy()
        penalty = raw_seq_reward - shaped_seq_reward
        orig_acc_debug = np.array(
            [float(a) for a in batch.non_tensor_batch['original_accuracy']],
            dtype=float)
        denom = max(1.0 - easy_threshold, 1e-8)
        easy_weight_debug = np.clip(
            (orig_acc_debug - easy_threshold) / denom, 0.0, 1.0)
        applied_idx = np.where(penalty > 1e-12)[0]
        top_idx = applied_idx[np.argsort(-penalty[applied_idx])[:5]]
        uids_debug = batch.non_tensor_batch.get('uid', None)
        debug_entry['top_penalized_examples'] = [
            {
                'batch_pos': int(i),
                'index': int(batch.non_tensor_batch['index'][i]),
                'uid': str(uids_debug[i]) if uids_debug is not None else None,
                'original_accuracy': float(orig_acc_debug[i]),
                'easy_weight': float(easy_weight_debug[i]),
                'response_length': float(response_len[i]),
                'length_norm': float(response_len[i] / max(response_length, 1)),
                'raw_reward': float(raw_seq_reward[i]),
                'shaped_reward': float(shaped_seq_reward[i]),
                'penalty': float(penalty[i]),
            }
            for i in top_idx
        ]
    _append_jsonl(easy_debug_path, debug_entry)
