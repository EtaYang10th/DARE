"""Inspiration strategies for hard-prompt rollout variants."""

import json
import os
import re
import sys
import uuid

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.rollout_method import dataprotoitem_to_dataproto, COLOR_RED, COLOR_RESET


_BOXED_SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."
_GSM8K_SYSTEM_PROMPT = 'Let\'s think step by step and output the final answer after "####".'


_HARD_HINT = (
    "This is a challenging problem. "
    "Please think longer and more carefully. "
    "Break it down step by step before giving your final answer."
)


def inspiration_for_hard(actor_rollout_wg, reward_fn, batch, reward_tensor, gen_batch,
                         n, max_n, metrics, hard_threshold=0.2,
                         tokenizer=None, max_prompt_length=None):
    """Re-generate for prompts whose accuracy is below hard_threshold.

    For each hard prompt (accuracy < hard_threshold), re-roll using a
    hint-augmented prompt that encourages the model to think more carefully.
    From the extra rollouts, keep as many correct rollouts as needed to raise
    the prompt accuracy to at least hard_threshold, replacing the same number
    of wrong originals. The extra generation budget per prompt is (max_n - n).
    """
    log_prefix = f"{COLOR_RED}[hard补样本]"
    log_suffix = COLOR_RESET

    extra_budget = max_n - n
    if extra_budget <= 0:
        return batch, reward_tensor

    uids = batch.non_tensor_batch['uid']
    seen = {}
    for uid in uids:
        if uid not in seen:
            seen[uid] = len(seen)
    unique_uids_ordered = list(seen.keys())

    hard_indices = []
    for i, uid in enumerate(unique_uids_ordered):
        uid_mask = uids == uid
        uid_scores = reward_tensor[uid_mask].sum(-1)
        n_correct = int((uid_scores > 0).sum())
        n_total = int(uid_mask.sum())
        target_correct = int(np.ceil(hard_threshold * n_total))
        if n_correct / n_total < hard_threshold:
            hard_indices.append((i, n_correct, n_total, target_correct))

    if not hard_indices:
        metrics['inspiration_hard/n_hard'] = 0
        return batch, reward_tensor

    n_hard = len(hard_indices)
    metrics['inspiration_hard/n_hard'] = n_hard
    print(f"{log_prefix} {n_hard}/{len(unique_uids_ordered)} 个 prompt 的正确率低于 "
          f"{hard_threshold}（额外 rollout 预算={extra_budget}）{log_suffix}")

    n_repeats = max(1, (extra_budget + n - 1) // n)
    inspired_budget = min(extra_budget, n_repeats * n)

    inspired_input_ids_list = []
    inspired_attention_mask_list = []
    inspired_position_ids_list = []

    use_hint = tokenizer is not None and max_prompt_length is not None
    if use_hint:
        print(f"{log_prefix} 使用 hint-augmented prompt 进行额外 rollout{log_suffix}")
    else:
        print(f"{log_prefix} tokenizer 未提供，回退到原始 prompt{log_suffix}")

    for idx, _existing_correct, _n_total, _target_correct in hard_indices:
        if use_hint:
            # Build hint-augmented prompt
            prompt_pos = idx  # position in gen_batch (one per unique prompt)
            first_batch_idx = idx * n
            extra_info = batch.non_tensor_batch['extra_info'][first_batch_idx]
            question = extra_info.get('question', '') if isinstance(extra_info, dict) else ''
            data_source = batch.non_tensor_batch['data_source'][first_batch_idx]

            user_content = f"{question.strip()}\n\n{_HARD_HINT}"
            chat = [
                {'role': 'system', 'content': _default_system_prompt(data_source)},
                {'role': 'user', 'content': user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False)
            ids, mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
                tokenizer=tokenizer,
                max_length=int(max_prompt_length),
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation='left',
            )
            pos = compute_position_id_with_mask(mask)
            for _ in range(n_repeats):
                inspired_input_ids_list.append(ids[0])
                inspired_attention_mask_list.append(mask[0])
                inspired_position_ids_list.append(pos[0])
        else:
            # Fallback: reuse original prompt as-is
            for _ in range(n_repeats):
                inspired_input_ids_list.append(gen_batch.batch['input_ids'][idx])
                inspired_attention_mask_list.append(gen_batch.batch['attention_mask'][idx])
                inspired_position_ids_list.append(gen_batch.batch['position_ids'][idx])

    inspired_gen_batch = DataProto.from_dict(
        tensors={
            'input_ids': torch.stack(inspired_input_ids_list),
            'attention_mask': torch.stack(inspired_attention_mask_list),
            'position_ids': torch.stack(inspired_position_ids_list),
        },
        meta_info=dict(gen_batch.meta_info),
    )

    world_size = actor_rollout_wg.world_size
    inspired_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
        inspired_gen_batch, world_size)

    print(f"{log_prefix} 正在为 {n_hard} 个 prompt 生成额外 rollout："
          f"每题最多使用 {inspired_budget} 个候选（{n_repeats}x{n}）...{log_suffix}")
    inspired_gen_output_padded = actor_rollout_wg.generate_sequences(
        inspired_gen_batch_padded)

    keep_total = n_hard * n_repeats * n
    if pad_size > 0:
        inspired_gen_output = dataprotoitem_to_dataproto(
            inspired_gen_output_padded[:keep_total])
    else:
        inspired_gen_output = inspired_gen_output_padded

    inspired_non_tensor_indices = []
    for idx, _, _, _ in hard_indices:
        first_batch_idx = idx * n
        inspired_non_tensor_indices.extend([first_batch_idx] * (n_repeats * n))

    inspired_non_tensor = {
        key: val[inspired_non_tensor_indices]
        for key, val in batch.non_tensor_batch.items()
    }
    inspired_batch_for_reward = DataProto(
        batch=inspired_gen_output.batch,
        non_tensor_batch=inspired_non_tensor,
        meta_info=batch.meta_info,
    )

    print(f"{log_prefix} 正在计算额外 rollout 的奖励...{log_suffix}")
    inspired_reward_tensor = reward_fn(inspired_batch_for_reward)

    n_helped = 0
    col_w = len(str(n_hard))

    for j, (orig_idx, existing_correct, n_total, target_correct) in enumerate(hard_indices):
        uid = unique_uids_ordered[orig_idx]
        uid_mask = uids == uid
        uid_indices = np.where(uid_mask)[0]
        needed_correct = target_correct - existing_correct

        inspired_start = j * n_repeats * n
        inspired_end = inspired_start + inspired_budget
        inspired_rewards = inspired_reward_tensor[inspired_start:inspired_end].sum(-1)
        correct_mask_inspired = (inspired_rewards > 0)
        correct_positions = correct_mask_inspired.nonzero(as_tuple=True)[0]
        n_correct_inspired = len(correct_positions)

        wrong_mask_orig = (reward_tensor[uid_mask].sum(-1) == 0).cpu().numpy()
        wrong_global = uid_indices[wrong_mask_orig]
        n_replace = min(needed_correct, n_correct_inspired, len(wrong_global))

        if n_replace == 0:
            status = (f"原始 {existing_correct}:{n_total - existing_correct}"
                      f" -> 目标至少 {target_correct}:{n_total - target_correct}，"
                      f"额外 rollout 命中 {n_correct_inspired}/{inspired_budget} 个正确，"
                      f"替换 0 个")
            line = (f"\r{log_prefix} [{j+1:>{col_w}}/{n_hard}] "
                    f"{status}                    ")
            sys.stdout.write(line + log_suffix + "\n")
            sys.stdout.flush()
            continue

        chosen = np.random.choice(
            correct_positions.cpu().numpy(), n_replace, replace=False)
        replace_targets = np.random.choice(
            wrong_global, n_replace, replace=False)

        for target_idx, inspired_local_idx in zip(replace_targets, chosen):
            actual_idx = inspired_start + int(inspired_local_idx)
            for key in inspired_gen_output.batch.keys():
                batch.batch[key][target_idx] = \
                    inspired_gen_output.batch[key][actual_idx]
            reward_tensor[target_idx] = inspired_reward_tensor[actual_idx]

        final_scores = reward_tensor[uid_mask].sum(-1)
        nc = int((final_scores > 0).sum())
        ni = int((final_scores == 0).sum())
        success = nc >= target_correct
        if success:
            n_helped += 1
        tag = "达标" if success else ("部分提升" if nc > existing_correct else "未提升")
        status = (f"原始 {existing_correct}:{n_total - existing_correct}"
                  f" -> 目标至少 {target_correct}:{n_total - target_correct}，"
                  f"额外 rollout 命中 {n_correct_inspired}/{inspired_budget} 个正确，"
                  f"替换 {n_replace} 个 -> {nc}:{ni} [{tag}]")
        line = (f"\r{log_prefix} [{j+1:>{col_w}}/{n_hard}] "
                f"{status}                    ")
        sys.stdout.write(line + log_suffix + "\n")
        sys.stdout.flush()

    metrics['inspiration_hard/n_helped'] = n_helped
    metrics['inspiration_hard/n_not_helped'] = n_hard - n_helped
    print(f"{log_prefix} 完成：{n_helped}/{n_hard} 个 prompt 达到目标，"
          f"{n_hard - n_helped} 个未达到目标{log_suffix}")

    return batch, reward_tensor


def _default_system_prompt(data_source):
    ds = str(data_source).lower()
    if 'gsm8k' in ds:
        return _GSM8K_SYSTEM_PROMPT
    return _BOXED_SYSTEM_PROMPT


def _format_final_answer(answer_text, data_source):
    if not answer_text:
        return None
    ds = str(data_source).lower()
    if 'gsm8k' in ds:
        return f'#### {answer_text}'
    return f'\\boxed{{{answer_text}}}'


def _extract_final_answer_text(response_text, data_source):
    ds = str(data_source).lower()
    text = response_text.strip()
    if not text:
        return None

    if 'gsm8k' in ds:
        try:
            from verl.utils.reward_score.gsm8k import extract_solution
            ans = extract_solution(text, method='flexible')
            if ans is not None:
                return str(ans).strip()
        except Exception:
            pass

    try:
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        boxed = last_boxed_only_string(text)
        if boxed is not None:
            return remove_boxed(boxed).strip()
    except Exception:
        pass

    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    hash_match = re.search(r'####\s*([^\n]+)', text)
    if hash_match:
        return hash_match.group(1).strip()

    return None


def _split_reasoning_fragments(text):
    text = text.replace('\r', '\n').strip()
    fragments = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        pieces = re.split(r'(?<=[。；;.!?])\s+', line)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                fragments.append(piece)
    if len(fragments) <= 1:
        fragments = [p.strip() for p in re.split(r'(?<=[。；;.!?])\s+', text) if p.strip()]
    return fragments


def _extract_reasoning_snippets(response_text, max_snippets=3, max_chars_per_snippet=160):
    fragments = _split_reasoning_fragments(response_text)
    if not fragments:
        return []

    scored = []
    seen = set()
    for idx, frag in enumerate(fragments):
        frag = ' '.join(frag.split())
        if len(frag) < 12:
            continue
        if '\\boxed' in frag or '####' in frag:
            continue
        low = frag.lower()
        if low in seen:
            continue
        seen.add(low)

        score = 0
        for token in ('=', '\\frac', '\\sqrt', '\\sin', '\\cos', '\\tan', '\\log', '^', '+', '-', '*', '/'):
            if token in frag:
                score += 2
        for token in ('let', 'substitute', 'simplify', 'solve', 'therefore', 'hence', 'thus', 'consider', 'rewrite'):
            if token in low:
                score += 2
        if any(ch.isdigit() for ch in frag):
            score += 1
        if len(frag) > max_chars_per_snippet:
            frag = frag[:max_chars_per_snippet].rstrip() + '...'
        scored.append((score, idx, frag))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:max(1, int(max_snippets))]
    selected.sort(key=lambda x: x[1])
    return [frag for _, _, frag in selected]


def _build_memory_user_content(question, success_response, data_source,
                               max_snippets=3, max_chars_per_snippet=160):
    answer_text = _extract_final_answer_text(success_response, data_source)
    formatted_answer = _format_final_answer(answer_text, data_source)
    snippets = _extract_reasoning_snippets(
        success_response,
        max_snippets=max_snippets,
        max_chars_per_snippet=max_chars_per_snippet,
    )

    lines = [question.strip(), '',
             'Helpful hint from one previous successful attempt on this same problem:']

    if snippets:
        lines.append('- Guided reasoning skeleton:')
        for i, snippet in enumerate(snippets, start=1):
            lines.append(f'  {i}. {snippet}')

    if 'gsm8k' in str(data_source).lower():
        lines.append('Use the hint as guidance, but still reason carefully and give the final answer after "####".')
    else:
        lines.append('Use the hint as guidance, but still reason carefully and give the final answer within \\boxed{}.')
    return '\n'.join(lines)


def _decode_response_text(tokenizer, response_ids):
    valid_ids = response_ids[response_ids != tokenizer.pad_token_id]
    return tokenizer.decode(valid_ids, skip_special_tokens=True).strip()


def _generate_prompt_group(actor_rollout_wg, batch, gen_batch, alloc_n, world_size):
    if alloc_n <= 0 or len(batch) == 0:
        return None, 0

    grp_gen = gen_batch
    grp_gen.meta_info = dict(grp_gen.meta_info)
    grp_gen.meta_info['override_n'] = int(alloc_n)

    grp_gen_padded, pad_sz = pad_dataproto_to_divisor(grp_gen, world_size)
    grp_output_padded = actor_rollout_wg.generate_sequences(grp_gen_padded)

    n_real = len(batch) * int(alloc_n)
    if pad_sz > 0:
        grp_output = dataprotoitem_to_dataproto(grp_output_padded[:n_real])
    else:
        grp_output = grp_output_padded

    grp_batch = batch.repeat(repeat_times=int(alloc_n), interleave=True)
    grp_batch = grp_batch.union(grp_output)
    return grp_batch, n_real


def inspiration_for_hard_memory(actor_rollout_wg, tokenizer, replay_buffer, batch, gen_batch,
                                predicted_labels, n, metrics, hard_threshold=0.2,
                                mix_ratio=0.5, max_prompt_length=1024,
                                max_snippets=3, max_chars_per_snippet=160,
                                output_dir=None, epoch=None, batch_step=None,
                                global_step=None):
    """Budget-neutral same-prompt memory prompting for hard prompts.

    Prompts whose predicted accuracy is at or below ``hard_threshold`` are
    considered hard. If the replay buffer contains successful rollouts from the
    same prompt, a random correct rollout is converted into an answer-anchored
    reasoning skeleton and injected into a memory-augmented prompt. Rollouts are
    then split into plain and memory-guided channels under the original budget.
    """
    log_prefix = f"{COLOR_RED}[hard记忆提示]"
    log_suffix = COLOR_RESET

    if len(batch) == 0:
        return batch, gen_batch

    def _save_prompt_records(records, *, reason):
        if not output_dir:
            return None
        save_dir = os.path.join(output_dir, 'saved_hard_memory_prompts')
        os.makedirs(save_dir, exist_ok=True)
        epoch_tag = 'na' if epoch is None else epoch
        gstep_tag = 'na' if global_step is None else global_step
        bstep_tag = 'na' if batch_step is None else batch_step
        save_path = os.path.join(
            save_dir,
            f'hard_memory_prompts_epoch_{epoch_tag}_step_{gstep_tag}_batch_{bstep_tag}.json',
        )
        payload = {
            'epoch': epoch,
            'global_step': global_step,
            'batch_step': batch_step,
            'reason': reason,
            'hard_threshold': float(hard_threshold),
            'mix_ratio': float(mix_ratio),
            'base_rollout_n': int(n),
            'records': records,
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return save_path

    mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    n_mem = int(round(int(n) * mix_ratio))
    if int(n) > 1 and mix_ratio > 0 and n_mem == 0:
        n_mem = 1
    if int(n) > 1 and mix_ratio < 1 and n_mem == int(n):
        n_mem = int(n) - 1
    n_plain = int(n) - n_mem

    n_prompts = len(batch)
    batch.non_tensor_batch['uid'] = np.array(
        [str(uuid.uuid4()) for _ in range(n_prompts)], dtype=object)
    batch.non_tensor_batch['_prompt_pos'] = np.arange(n_prompts, dtype=object)

    prompt_indices = np.array(batch.non_tensor_batch['index'], dtype=int)
    batch_pred = np.array([float(predicted_labels[int(idx)]) for idx in prompt_indices])

    # Use relative ranking within the batch: treat the lowest hard_threshold
    # fraction as "hard", so that the strategy works regardless of how the
    # upstream sampling distribution concentrates predicted_labels.
    hard_quantile = float(np.quantile(batch_pred, float(hard_threshold)))
    hard_mask = batch_pred <= hard_quantile
    # Ensure at least 1 hard prompt when hard_threshold > 0 and batch is
    # non-trivial, even if quantile collapses (e.g. all identical preds).
    if hard_mask.sum() == 0 and float(hard_threshold) > 0 and n_prompts > 1:
        hard_mask[np.argmin(batch_pred)] = True
    n_hard = int(hard_mask.sum())
    metrics['inspiration_hard_memory/n_hard'] = n_hard

    print(f"{log_prefix} ===== 预算不变 hard-memory 提示策略开始 ====={log_suffix}")
    print(f"{log_prefix} step={global_step}, epoch={epoch}, batch_step={batch_step}, "
          f"总prompt={n_prompts}, hard阈值(batch内{float(hard_threshold)*100:.0f}%分位)={hard_quantile:.3f}, "
          f"batch预测范围=[{batch_pred.min():.3f}, {batch_pred.max():.3f}], "
          f"mixed-channel plain/memory={n_plain}/{n_mem}{log_suffix}")

    if replay_buffer is None or len(replay_buffer) == 0 or n_hard == 0:
        reason = 'no_replay_buffer' if replay_buffer is None or len(replay_buffer) == 0 else 'no_hard_prompt'
        print(f"{log_prefix} 跳过记忆提示："
              f"{'replay buffer为空' if reason == 'no_replay_buffer' else '本batch没有hard prompt'}，"
              f"全部按普通rollout执行{log_suffix}")
        save_path = _save_prompt_records([], reason=reason)
        if save_path is not None:
            print(f"{log_prefix} 本轮构造prompt已保存到: {save_path}{log_suffix}")

        base_batch, _ = _generate_prompt_group(
            actor_rollout_wg=actor_rollout_wg,
            batch=batch,
            gen_batch=gen_batch,
            alloc_n=int(n),
            world_size=actor_rollout_wg.world_size,
        )
        metrics['inspiration_hard_memory/n_with_memory'] = 0
        metrics['inspiration_hard_memory/n_fallback_plain'] = n_hard
        metrics['inspiration_hard_memory/plain_rollouts'] = int(n) * n_prompts
        metrics['inspiration_hard_memory/memory_rollouts'] = 0
        metrics['inspiration_hard_memory/success_pool_size_mean'] = 0.0
        if '_prompt_pos' in base_batch.non_tensor_batch:
            del base_batch.non_tensor_batch['_prompt_pos']
        return base_batch, gen_batch

    buf_indices = np.array(replay_buffer.non_tensor_batch['index'], dtype=int)
    buf_scores = replay_buffer.batch['token_level_scores'].sum(dim=-1)
    buf_correct = (buf_scores > 0).detach().cpu().numpy()

    # Pre-build success index: dataset_idx -> list of buffer positions
    from collections import defaultdict
    _success_index = defaultdict(list)
    for _bp in range(len(buf_indices)):
        if buf_correct[_bp]:
            _success_index[int(buf_indices[_bp])].append(_bp)

    memory_prompt_positions = []
    success_pool_sizes = []
    mem_ids_list = []
    mem_mask_list = []
    mem_pos_list = []
    prompt_records = []
    hard_rank = 0

    for prompt_pos, dataset_idx in enumerate(prompt_indices):
        if not hard_mask[prompt_pos]:
            continue

        hard_rank += 1
        pred_acc = float(predicted_labels[int(dataset_idx)])
        same_prompt_positions = np.array(_success_index.get(int(dataset_idx), []))
        extra_info = batch.non_tensor_batch['extra_info'][prompt_pos]
        question = extra_info.get('question', '') if isinstance(extra_info, dict) else ''
        data_source = batch.non_tensor_batch['data_source'][prompt_pos]

        if len(same_prompt_positions) == 0:
            print(f"{log_prefix} [{hard_rank}/{n_hard}] idx={int(dataset_idx)} 预测成功率={pred_acc:.3f}，"
                  f"未命中同题历史正确rollout -> 回退普通rollout{log_suffix}")
            prompt_records.append({
                'dataset_idx': int(dataset_idx),
                'predicted_accuracy': pred_acc,
                'used_memory': False,
                'reason': 'no_same_prompt_success',
                'question': question,
                'plain_rollouts': int(n),
                'memory_rollouts': 0,
            })
            continue

        chosen_pos = int(np.random.choice(same_prompt_positions))
        success_pool_sizes.append(len(same_prompt_positions))

        response_text = _decode_response_text(
            tokenizer,
            replay_buffer.batch['responses'][chosen_pos],
        )
        if not response_text:
            print(f"{log_prefix} [{hard_rank}/{n_hard}] idx={int(dataset_idx)} 预测成功率={pred_acc:.3f}，"
                  f"命中同题成功{len(same_prompt_positions)}条，但解码为空 -> 回退普通rollout{log_suffix}")
            prompt_records.append({
                'dataset_idx': int(dataset_idx),
                'predicted_accuracy': pred_acc,
                'used_memory': False,
                'reason': 'empty_success_response',
                'question': question,
                'plain_rollouts': int(n),
                'memory_rollouts': 0,
                'success_pool_size': int(len(same_prompt_positions)),
                'chosen_buffer_position': int(chosen_pos),
            })
            continue

        if not question:
            print(f"{log_prefix} [{hard_rank}/{n_hard}] idx={int(dataset_idx)} 预测成功率={pred_acc:.3f}，"
                  f"缺少question字段 -> 回退普通rollout{log_suffix}")
            prompt_records.append({
                'dataset_idx': int(dataset_idx),
                'predicted_accuracy': pred_acc,
                'used_memory': False,
                'reason': 'missing_question',
                'plain_rollouts': int(n),
                'memory_rollouts': 0,
                'success_pool_size': int(len(same_prompt_positions)),
                'chosen_buffer_position': int(chosen_pos),
            })
            continue

        answer_text = _extract_final_answer_text(response_text, data_source)
        formatted_answer = _format_final_answer(answer_text, data_source)
        snippets = _extract_reasoning_snippets(
            response_text,
            max_snippets=max_snippets,
            max_chars_per_snippet=max_chars_per_snippet,
        )
        user_content = _build_memory_user_content(
            question=question,
            success_response=response_text,
            data_source=data_source,
            max_snippets=max_snippets,
            max_chars_per_snippet=max_chars_per_snippet,
        )
        chat = [
            {'role': 'system', 'content': _default_system_prompt(data_source)},
            {'role': 'user', 'content': user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False)
        ids, mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=tokenizer,
            max_length=int(max_prompt_length),
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation='left',
        )
        pos = compute_position_id_with_mask(mask)

        memory_prompt_positions.append(prompt_pos)
        mem_ids_list.append(ids[0])
        mem_mask_list.append(mask[0])
        mem_pos_list.append(pos[0])

        answer_preview = formatted_answer if formatted_answer is not None else '无显式答案锚点'
        snippet_count = len(snippets)
        print(f"{log_prefix} [{hard_rank}/{n_hard}] idx={int(dataset_idx)} pred={pred_acc:.3f}，"
              f"buffer命中={len(same_prompt_positions)}条，选buffer[{chosen_pos}]，"
              f"答案={answer_preview}，骨架={snippet_count}段{log_suffix}")

        prompt_records.append({
            'dataset_idx': int(dataset_idx),
            'predicted_accuracy': pred_acc,
            'used_memory': True,
            'reason': 'same_prompt_success',
            'question': question,
            'data_source': str(data_source),
            'success_pool_size': int(len(same_prompt_positions)),
            'chosen_buffer_position': int(chosen_pos),
            'selected_success_response_preview': response_text[:500],
            'answer_anchor': formatted_answer,
            'snippets': snippets,
            'plain_rollouts': int(n_plain),
            'memory_rollouts': int(n_mem),
            'constructed_user_content': user_content,
            'constructed_prompt_text': prompt_text,
        })

    eligible_mask = np.zeros(n_prompts, dtype=bool)
    if memory_prompt_positions:
        eligible_mask[np.array(memory_prompt_positions, dtype=int)] = True

    n_with_memory = int(eligible_mask.sum())
    n_fallback_plain = int(n_hard - n_with_memory)
    metrics['inspiration_hard_memory/n_with_memory'] = n_with_memory
    metrics['inspiration_hard_memory/n_fallback_plain'] = n_fallback_plain
    metrics['inspiration_hard_memory/plain_rollouts'] = (
        int((~eligible_mask).sum()) * int(n) + n_with_memory * int(n_plain)
    )
    metrics['inspiration_hard_memory/memory_rollouts'] = n_with_memory * int(n_mem)
    metrics['inspiration_hard_memory/success_pool_size_mean'] = (
        float(np.mean(success_pool_sizes)) if success_pool_sizes else 0.0
    )
    metrics['inspiration_hard_memory/discarded_rollouts'] = n_with_memory * int(n_mem)

    print(f"{log_prefix} 汇总：hard={n_hard}/{n_prompts}，"
          f"命中同题成功记忆={n_with_memory}，fallback={n_fallback_plain}，"
          f"总plain rollouts={metrics['inspiration_hard_memory/plain_rollouts']}，"
          f"总memory rollouts={metrics['inspiration_hard_memory/memory_rollouts']}，"
          f"丢弃rollouts={n_with_memory * int(n_mem)}（合并通道优化）{log_suffix}")

    save_path = _save_prompt_records(
        prompt_records,
        reason='constructed' if prompt_records else 'no_constructed_prompt',
    )
    if save_path is not None:
        print(f"{log_prefix} 本轮构造prompt已保存到: {save_path}{log_suffix}")

    # ------------------------------------------------------------------
    # Optimised rollout generation: 2 calls instead of 3.
    #
    # Call 1 (main):  ALL n_prompts original prompts × n  (= baseline)
    #   – For non-memory prompts: keep all n rollouts.
    #   – For memory-eligible prompts: keep only the first n_plain rollouts
    #     (the remaining n_mem rollouts per prompt are discarded).
    #
    # Call 2 (memory): n_with_memory memory-augmented prompts × n_mem
    #   – These are appended to the memory-eligible prompts.
    #
    # This merges the old Channel-A and Channel-B into a single call,
    # eliminating one generate_sequences round-trip.
    # ------------------------------------------------------------------
    world_size = actor_rollout_wg.world_size
    all_parts = []

    # --- Call 1: all prompts × n (plain channel) ---
    print(f"{log_prefix} 合并通道(A+B)：全部 {n_prompts} 个prompt × {int(n)} 条rollout{log_suffix}")
    main_batch, _ = _generate_prompt_group(
        actor_rollout_wg=actor_rollout_wg,
        batch=batch,
        gen_batch=gen_batch,
        alloc_n=int(n),
        world_size=world_size,
    )

    if n_with_memory > 0 and n_mem > 0:
        # For memory-eligible prompts we only keep the first n_plain
        # rollouts out of the n generated, then append n_mem memory
        # rollouts from Call 2.
        #
        # main_batch is interleaved: for each prompt, n consecutive rows.
        # We need to drop the last n_mem rows of each memory-eligible
        # prompt and keep the first n_plain rows.
        prompt_pos_arr = np.array(main_batch.non_tensor_batch['_prompt_pos'], dtype=int)
        keep_mask = np.ones(len(main_batch), dtype=bool)
        for mp in memory_prompt_positions:
            # Find all rows belonging to this prompt
            rows = np.where(prompt_pos_arr == mp)[0]
            # rows should have exactly n entries; drop the last n_mem
            if len(rows) == int(n):
                drop_rows = rows[int(n_plain):]
                keep_mask[drop_rows] = False
        trimmed_main = dataprotoitem_to_dataproto(main_batch[keep_mask.tolist()])
        all_parts.append(trimmed_main)

        # --- Call 2: memory channel ---
        print(f"{log_prefix} 记忆通道：{n_with_memory} 个memory prompt × {int(n_mem)} 条rollout{log_suffix}")
        mem_batch = dataprotoitem_to_dataproto(batch[eligible_mask.tolist()])
        mem_gen_batch = DataProto.from_dict(
            tensors={
                'input_ids': torch.stack(mem_ids_list),
                'attention_mask': torch.stack(mem_mask_list),
                'position_ids': torch.stack(mem_pos_list),
            },
            meta_info=dict(gen_batch.meta_info),
        )
        mem_part, _ = _generate_prompt_group(
            actor_rollout_wg=actor_rollout_wg,
            batch=mem_batch,
            gen_batch=mem_gen_batch,
            alloc_n=int(n_mem),
            world_size=world_size,
        )
        all_parts.append(mem_part)
    else:
        # No memory prompts — keep main_batch as-is (pure baseline path)
        all_parts.append(main_batch)

    combined = DataProto.concat(all_parts) if len(all_parts) > 1 else all_parts[0]
    prompt_pos = np.array(combined.non_tensor_batch['_prompt_pos'], dtype=int)
    sort_order = np.argsort(prompt_pos, kind='stable')
    combined = dataprotoitem_to_dataproto(combined[sort_order.tolist()])
    if '_prompt_pos' in combined.non_tensor_batch:
        del combined.non_tensor_batch['_prompt_pos']

    print(f"{log_prefix} ===== hard-memory 提示策略完成：最终生成 {len(combined)} 条rollout ====={log_suffix}")
    return combined, gen_batch


# def inspiration_for_easy(actor_rollout_wg, reward_fn, batch, reward_tensor, gen_batch,
#                          n, metrics, temperature, easy_threshold=0.9):
#     """Legacy easy-prompt regeneration path.
#
#     This implementation is intentionally disabled and kept only for reference.
#     The active code path for easy prompts is the length-penalty term injected
#     into the actor loss in ``ray_trainer.py``.
#     """
#     pass


def apply_hard_length_reward_shaping(batch, hard_length_threshold, length_bonus_coeff):
    """Apply hard-prompt length bonus to token-level rewards.

    For prompts whose real rollout accuracy is below *hard_length_threshold*,
    give an extra reward proportional to response length on **wrong** responses.
    This encourages the model to think longer on hard problems.

    The bonus for each wrong response is:
        bonus = length_bonus_coeff * hard_weight * length_norm
    where
        hard_weight = clamp((threshold - acc) / threshold, 0, 1)
        length_norm = response_len / max_response_length
    """
    stats = {
        'enabled': False,
        'status': 'disabled',
        'hard_length_threshold': float(hard_length_threshold),
        'length_bonus_coeff': float(length_bonus_coeff),
        'total_responses': int(len(batch)),
        'total_prompts': 0,
        'hard_responses': 0,
        'hard_prompts': 0,
        'applied_responses': 0,
        'applied_prompts': 0,
        'avg_hard_weight': 0.0,
        'avg_applied_length_norm': 0.0,
        'avg_applied_response_len': 0.0,
        'avg_bonus': 0.0,
        'max_bonus': 0.0,
        'total_bonus': 0.0,
        'reward_mean_before': 0.0,
        'reward_mean_after': 0.0,
        'reward_delta_mean': 0.0,
    }

    token_level_scores = batch.batch['token_level_scores']
    # Initialise token_level_rewards if not yet set
    if 'token_level_rewards' not in batch.batch:
        batch.batch['token_level_rewards'] = token_level_scores.clone()
    batch.meta_info['hard_length_bonus_stats'] = stats

    if length_bonus_coeff <= 0 or len(batch) == 0:
        return stats

    orig_acc_arr = batch.non_tensor_batch.get('original_accuracy', None)
    if orig_acc_arr is None:
        stats['status'] = 'missing_original_accuracy'
        batch.meta_info['hard_length_bonus_stats'] = stats
        return stats

    responses = batch.batch['responses']
    response_length = int(responses.shape[-1])
    if response_length <= 0:
        stats['status'] = 'empty_response'
        batch.meta_info['hard_length_bonus_stats'] = stats
        return stats

    device = token_level_scores.device
    work_dtype = torch.float32
    response_mask = batch.batch['attention_mask'][:, -response_length:].to(work_dtype)
    response_len = response_mask.sum(dim=-1)
    length_norm = response_len / max(float(response_length), 1.0)
    raw_sequence_reward = token_level_scores.sum(dim=-1).to(work_dtype)

    # hard_weight: higher when accuracy is further below threshold
    denom = max(float(hard_length_threshold), 1e-8)
    acc_tensor = torch.tensor(
        [float(a) for a in orig_acc_arr],
        dtype=work_dtype, device=device)
    hard_weight = ((float(hard_length_threshold) - acc_tensor) / denom).clamp(0.0, 1.0)

    # Only reward wrong responses
    wrong_mask = (raw_sequence_reward <= 0).to(work_dtype)
    applied_weight = hard_weight * wrong_mask
    bonus = float(length_bonus_coeff) * applied_weight * length_norm

    token_level_rewards = batch.batch['token_level_rewards'].clone()
    last_token_idx = response_len.long().clamp(min=1) - 1
    row_idx = torch.arange(len(batch), device=device)
    token_level_rewards[row_idx, last_token_idx] += bonus.to(token_level_rewards.dtype)
    batch.batch['token_level_rewards'] = token_level_rewards

    # --- Collect stats ---
    hard_mask_cpu = (hard_weight > 0).detach().cpu().numpy()
    applied_mask_cpu = (applied_weight > 0).detach().cpu().numpy()
    length_norm_cpu = length_norm.detach().cpu().numpy()
    response_len_cpu = response_len.detach().cpu().numpy()
    bonus_cpu = bonus.detach().cpu().numpy()
    raw_reward_cpu = raw_sequence_reward.detach().cpu().numpy()
    shaped_reward_cpu = token_level_rewards.sum(dim=-1).detach().cpu().numpy()

    stats['enabled'] = True
    stats['status'] = 'applied' if bool(applied_mask_cpu.any()) else 'no_matching_samples'
    stats['hard_responses'] = int(hard_mask_cpu.sum())
    stats['applied_responses'] = int(applied_mask_cpu.sum())
    stats['reward_mean_before'] = float(raw_reward_cpu.mean()) if len(raw_reward_cpu) > 0 else 0.0
    stats['reward_mean_after'] = float(shaped_reward_cpu.mean()) if len(shaped_reward_cpu) > 0 else 0.0
    stats['reward_delta_mean'] = stats['reward_mean_after'] - stats['reward_mean_before']

    if stats['hard_responses'] > 0:
        stats['avg_hard_weight'] = float(hard_weight[hard_weight > 0].mean().detach().cpu().item())

    if stats['applied_responses'] > 0:
        ab = bonus_cpu[applied_mask_cpu]
        al = length_norm_cpu[applied_mask_cpu]
        ar = response_len_cpu[applied_mask_cpu]
        stats['avg_applied_length_norm'] = float(al.mean())
        stats['avg_applied_response_len'] = float(ar.mean())
        stats['avg_bonus'] = float(ab.mean())
        stats['max_bonus'] = float(ab.max())
        stats['total_bonus'] = float(ab.sum())

    uids_arr = batch.non_tensor_batch.get('uid', None)
    if uids_arr is not None:
        prompt_hard = {}
        prompt_applied = {}
        for i, uid in enumerate(uids_arr):
            if uid not in prompt_hard:
                prompt_hard[uid] = bool(hard_mask_cpu[i])
            prompt_applied[uid] = prompt_applied.get(uid, False) or bool(applied_mask_cpu[i])
        stats['total_prompts'] = len(prompt_hard)
        stats['hard_prompts'] = int(sum(prompt_hard.values()))
        stats['applied_prompts'] = int(sum(prompt_applied.values()))
    else:
        stats['total_prompts'] = int(len(batch))
        stats['hard_prompts'] = stats['hard_responses']
        stats['applied_prompts'] = stats['applied_responses']

    batch.meta_info['hard_length_bonus_stats'] = stats
    return stats
