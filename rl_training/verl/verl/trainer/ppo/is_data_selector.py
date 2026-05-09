"""
IS-based data selection for LLM RL fine-tuning.

Replaces the teacher model difficulty prediction with Self-Normalized
Importance Sampling (SNIS) based estimation. For each question with
historical rollouts in the replay buffer, the success rate under the
current policy is re-estimated using IS weights from the ratio of
current to behavior policy probabilities.

Mathematical formulation:
    For question q with stored rollouts {(y_i, r_i, log π_μ(y_i|q))}:

        log w_i = Σ_t [log π_curr(y_{i,t}|q, y_{i,<t})
                      - log π_μ(y_{i,t}|q, y_{i,<t})]
        w_i = exp(clip(log w_i, -C, C))

        SNIS success rate:  ŝ_q = Σ(w_i r_i) / Σ(w_i)

    Effective Sample Size (ESS) diagnostic:
        ESS = (Σ w_i)² / Σ(w_i²)
    When ESS < threshold, the SNIS estimate is unreliable and we fall
    back to the raw (stale) success rate from the stored rewards.
"""

import torch
import numpy as np
import os
import json
import time
from collections import defaultdict

COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


_ACC_BINS = [0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.001]
_ACC_LABELS = ["0/8+1/16", "1/8±1/16", "2/8±1/16", "3/8±1/16",
               "4/8±1/16", "5/8±1/16", "6/8±1/16", "7/8±1/16", "8/8-1/16"]


def _ascii_histogram(values, title, n_bins=25, width=60, color=COLOR_BLUE,
                     bins=None, bin_labels=None):
    """Render a horizontal ASCII bar chart to stdout.

    Args:
        bins: custom bin edges; when provided, *n_bins* is ignored.
        bin_labels: display labels for each bin (len == len(bins)-1).
    """
    if len(values) == 0:
        print(f"{color}[{title}] 无数据{COLOR_RESET}")
        return

    vals = np.asarray(values, dtype=float)
    n = len(vals)
    v_min, v_max = float(vals.min()), float(vals.max())
    v_mean = float(vals.mean())
    v_std = float(vals.std()) if n > 1 else 0.0
    v_med = float(np.median(vals))
    p5, p25, p75, p95 = [float(x) for x in np.percentile(vals, [5, 25, 75, 95])]

    if bins is not None:
        edges = np.asarray(bins, dtype=float)
    elif v_min == v_max:
        edges = np.array([v_min - 0.5, v_max + 0.5])
    else:
        edges = np.linspace(v_min, v_max, n_bins + 1)

    counts, _ = np.histogram(vals, bins=edges)
    max_c = int(counts.max()) if counts.max() > 0 else 1

    if bin_labels is not None:
        labels = list(bin_labels)
    else:
        labels = [f"{(edges[i] + edges[i + 1]) / 2:.3f}" for i in range(len(counts))]

    lw = max(len(l) for l in labels) + 1
    sep = "─" * (lw + 2 + width + 8)

    lines = [f"[{title}] n={n}"]
    lines.append(sep)
    for i, c in enumerate(counts):
        bar_len = int(c / max_c * width)
        bar = "█" * bar_len
        lines.append(f"{labels[i]:>{lw}} ┤{bar:<{width}} {c}")
    lines.append(sep)
    lines.append(
        f"mean={v_mean:.4f}  std={v_std:.4f}  median={v_med:.4f}")
    lines.append(
        f"p5={p5:.4f}  p25={p25:.4f}  p75={p75:.4f}  p95={p95:.4f}")
    lines.append(
        f"min={v_min:.4f}  max={v_max:.4f}")

    print(color + "\n".join(lines) + COLOR_RESET)


class ISDataSelector:
    """SNIS-based adaptive difficulty estimator.

    Outputs predicted labels (estimated success rates) in the same format
    as the teacher model, for use with the DOTS-style softmax selection:
        P(q) ∝ exp(-|predicted_label_q - α| / τ)
    """

    def __init__(self, clip_range=5.0, ess_threshold=2.0,
                 default_label=0.5, save_dir=None):
        """
        Args:
            clip_range: symmetric clipping bound for log IS weights.
                log w_i is clamped to [-clip_range, +clip_range].
            ess_threshold: minimum Effective Sample Size per question.
                Below this, the SNIS estimate is deemed unreliable and
                the raw stored success rate is used instead.
            default_label: predicted label assigned to questions with
                no historical rollouts. 0.5 places them at the center
                of the selection distribution (maximally informative).
            save_dir: directory for per-epoch JSON diagnostics.
        """
        self.clip_range = clip_range
        self.ess_threshold = ess_threshold
        self.default_label = default_label
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def estimate_difficulty(
        self,
        replay_buffer,
        compute_log_prob_fn,
        dataset_size,
        ref_indices=None,
        ref_labels=None,
        epoch=None,
        fallback_labels=None,
    ):
        """Estimate predicted labels for all questions in the dataset.

        Priority:
            1. Reference set -> ground-truth success rate from fresh rollouts
            2. Replay buffer -> SNIS-estimated success rate
            3. fallback_labels (e.g. teacher model) if provided, else default_label

        Args:
            replay_buffer: DataProto with stored rollouts (may be None).
            compute_log_prob_fn: callable(DataProto) -> DataProto with
                'old_log_probs' under current policy weights.
            dataset_size: total number of questions in training set.
            ref_indices: list of question indices with ground-truth labels.
            ref_labels: list of ground-truth success rates for ref questions.
            epoch: current epoch (for diagnostics file naming).
            fallback_labels: optional torch.Tensor of shape (dataset_size,)
                with teacher model predictions for all questions. When
                provided, used instead of the fixed default_label for
                questions not covered by ref set or IS buffer.

        Returns:
            torch.Tensor of shape (dataset_size,) with predicted labels.
        """
        _use_teacher_fallback = fallback_labels is not None
        if _use_teacher_fallback:
            assert fallback_labels.shape == (dataset_size,), \
                f"fallback_labels shape {fallback_labels.shape} != ({dataset_size},)"
            predicted_labels = fallback_labels.clone()
        else:
            predicted_labels = torch.full((dataset_size,), self.default_label)

        ref_index_set = set()
        if ref_indices is not None and ref_labels is not None:
            for idx, label in zip(ref_indices, ref_labels):
                idx_int = int(idx)
                predicted_labels[idx_int] = float(label)
                ref_index_set.add(idx_int)

        self.last_is_indices = set()

        if replay_buffer is not None and len(replay_buffer) > 0:
            is_labels, raw_labels, diagnostics = self._compute_snis_labels(
                replay_buffer, compute_log_prob_fn)
            self.last_raw_labels = raw_labels

            n_applied = 0
            for q_idx_str, label_val in is_labels.items():
                q_idx = int(q_idx_str)
                if 0 <= q_idx < dataset_size and q_idx not in ref_index_set:
                    predicted_labels[q_idx] = label_val
                    self.last_is_indices.add(q_idx)
                    n_applied += 1

            n_remaining = dataset_size - n_applied - len(ref_index_set)
            diagnostics['n_applied'] = n_applied
            diagnostics['n_ref'] = len(ref_index_set)
            if _use_teacher_fallback:
                diagnostics['n_teacher_fallback'] = n_remaining
                diagnostics['n_default'] = 0
            else:
                diagnostics['n_teacher_fallback'] = 0
                diagnostics['n_default'] = n_remaining

            _remaining_desc = (f"{n_remaining} Teacher预测"
                               if _use_teacher_fallback
                               else f"{n_remaining} 默认值({self.default_label})")
            print(
                f"{COLOR_RED}[IS 选择器] 覆盖情况: "
                f"{len(ref_index_set)} 参考集 + {n_applied} IS估计 + "
                f"{_remaining_desc} "
                f"/ 共 {dataset_size} 题{COLOR_RESET}")

            if is_labels:
                is_vals = np.array(list(is_labels.values()))
                hist, _ = np.histogram(is_vals, bins=_ACC_BINS)
                std_val = float(is_vals.std()) if len(is_vals) > 1 else 0.0
                _ascii_histogram(is_vals, "IS 难度分布 (正确率)",
                                 bins=_ACC_BINS, bin_labels=_ACC_LABELS,
                                 width=55, color=COLOR_RED)
                self.last_distribution = {
                    "n_questions": len(is_vals),
                    "histogram": {l: int(c) for l, c in zip(_ACC_LABELS, hist)},
                    "stats": {
                        "min": float(is_vals.min()),
                        "max": float(is_vals.max()),
                        "mean": float(is_vals.mean()),
                        "std": std_val,
                    },
                }

            if self.save_dir and epoch is not None:
                diag_path = os.path.join(
                    self.save_dir, f'is_diagnostics_epoch_{epoch}.json')
                with open(diag_path, 'w') as f:
                    json.dump(diagnostics, f, indent=2)
        else:
            _remaining_desc = (
                f"{dataset_size - len(ref_index_set)} Teacher预测"
                if _use_teacher_fallback
                else f"{dataset_size - len(ref_index_set)} 默认值"
                     f"(label={self.default_label})")
            print(
                f"{COLOR_RED}[IS 选择器] 无可用回放缓冲区。"
                f"使用 {len(ref_index_set)} 参考集真实标签 + "
                f"{_remaining_desc}{COLOR_RESET}")

        return predicted_labels

    def _compute_snis_labels(self, replay_buffer, compute_log_prob_fn):
        """Core SNIS computation over replay buffer entries.

        For each question in the buffer:
            1. Retrieve stored behavior log probs log π_μ(y_i|q).
            2. Compute current policy log probs log π_curr(y_i|q)
               via a gradient-free forward pass.
            3. Compute clipped log IS weights.
            4. If ESS ≥ threshold → SNIS estimated success rate.
               If ESS < threshold → raw stored success rate (fallback).

        Returns:
            is_labels: dict mapping question index (str) to estimated
                success rate (float).
            diagnostics: dict with summary statistics.
        """
        start_time = time.time()

        stored_log_probs = replay_buffer.batch['old_log_probs']

        print(
            f"{COLOR_CYAN}[IS 选择器] 正在计算当前策略的log概率, "
            f"共 {len(replay_buffer)} 条缓冲区数据...{COLOR_RESET}")
        fwd_start = time.time()
        from verl.protocol import pad_dataproto_to_divisor
        from verl.trainer.ppo.rollout_method import dataprotoitem_to_dataproto
        buf_len = len(replay_buffer)
        padded_buffer, pad_sz = pad_dataproto_to_divisor(replay_buffer, 8)
        current_output = compute_log_prob_fn(padded_buffer)
        del padded_buffer  # free the padded copy immediately
        if pad_sz > 0:
            current_output = dataprotoitem_to_dataproto(current_output[:buf_len])
        current_log_probs = current_output.batch['old_log_probs']
        del current_output  # keep only the tensor we need
        fwd_elapsed = time.time() - fwd_start
        print(
            f"{COLOR_CYAN}[IS 选择器] 前向传播耗时: "
            f"{fwd_elapsed:.3f}s{COLOR_RESET}")

        responses = replay_buffer.batch['responses']
        response_length = responses.size(1)
        attention_mask = replay_buffer.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:].float()

        token_level_scores = replay_buffer.batch['token_level_scores']
        seq_rewards = token_level_scores.sum(dim=-1)
        binary_rewards = (seq_rewards > 0).float()

        log_ratio = (current_log_probs - stored_log_probs) * response_mask
        response_lengths = response_mask.sum(dim=-1).clamp(min=1)
        log_weights_raw = log_ratio.sum(dim=-1)
        log_weights = torch.clamp(
            log_weights_raw, -self.clip_range, self.clip_range)

        indices = replay_buffer.non_tensor_batch['index']
        question_groups = defaultdict(list)
        for i in range(len(replay_buffer)):
            question_groups[str(indices[i])].append(i)

        is_labels = {}
        raw_labels = {}
        ess_values = []
        n_snis = 0
        n_fallback = 0

        for q_str, entry_ids in question_groups.items():
            idx_t = torch.tensor(entry_ids, dtype=torch.long)
            g_log_w = log_weights[idx_t]
            g_rewards = binary_rewards[idx_t]

            max_lw = g_log_w.max()
            stable_w = torch.exp(g_log_w - max_lw)

            w_sum = stable_w.sum()
            w_sq_sum = (stable_w ** 2).sum()
            ess = (w_sum ** 2 / w_sq_sum).item()
            ess_values.append(ess)

            raw_success_rate = g_rewards.mean().item()
            raw_labels[q_str] = raw_success_rate

            if ess < self.ess_threshold:
                is_labels[q_str] = raw_success_rate
                n_fallback += 1
                # print(
                #     f"{COLOR_CYAN}[IS 选择器] 题目 {q_str}: ESS={ess:.2f} < 阈值 {self.ess_threshold}, "
                #     f"回退使用原始成功率 {raw_success_rate:.3f}{COLOR_RESET}")
            else:
                w_norm = stable_w / w_sum
                snis_success_rate = (w_norm * g_rewards).sum().item()
                snis_success_rate = max(0.0, min(1.0, snis_success_rate))
                is_labels[q_str] = snis_success_rate
                n_snis += 1

        elapsed = time.time() - start_time

        avg_ess = float(np.mean(ess_values)) if ess_values else 0.0
        n_buf = len(log_weights)
        avg_abs_lw = float(log_weights.abs().mean().item()) if n_buf > 0 else 0.0
        std_lw = float(log_weights.std().item()) if n_buf > 1 else 0.0
        n_clipped_pos = int((log_weights_raw > self.clip_range).sum().item())
        n_clipped_neg = int((log_weights_raw < -self.clip_range).sum().item())
        n_clipped = n_clipped_pos + n_clipped_neg

        diagnostics = {
            'n_questions_in_buffer': len(question_groups),
            'n_total_entries': n_buf,
            'n_snis_estimated': n_snis,
            'n_ess_fallback': n_fallback,
            'avg_ess': avg_ess,
            'avg_abs_log_weight': avg_abs_lw,
            'log_weight_std': std_lw,
            'n_clipped_weights': n_clipped,
            'n_clipped_pos': n_clipped_pos,
            'n_clipped_neg': n_clipped_neg,
            'clip_range': self.clip_range,
            'ess_threshold': self.ess_threshold,
            'forward_pass_seconds': fwd_elapsed,
            'total_compute_seconds': elapsed,
        }

        print(
            f"{COLOR_CYAN}[IS 选择器] "
            f"{n_snis} 题SNIS估计 + {n_fallback} 题ESS不足回退 "
            f"(共 {len(question_groups)} 题). "
            f"平均ESS={avg_ess:.2f}, |log w|={avg_abs_lw:.3f}, "
            f"σ(log w)={std_lw:.3f}, "
            f"截断={n_clipped}/{n_buf}(+{n_clipped_pos}/-{n_clipped_neg}), "
            f"耗时={elapsed:.3f}s{COLOR_RESET}")

        lw_np = log_weights.detach().cpu().numpy()
        avg_resp_len = float(response_lengths.mean().item())
        _ascii_histogram(
            lw_np,
            f"IS log-weights (序列级求和, 平均响应长度={avg_resp_len:.0f}tok, "
            f"clip=±{self.clip_range})",
            n_bins=25, width=55, color=COLOR_BLUE)

        _ascii_histogram(
            ess_values,
            f"ESS 分布 (阈值={self.ess_threshold})",
            n_bins=15, width=55, color=COLOR_CYAN)

        return is_labels, raw_labels, diagnostics
