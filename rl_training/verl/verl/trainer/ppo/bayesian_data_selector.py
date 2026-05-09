"""
MoPPS-style Bayesian data selection for LLM RL fine-tuning.

Implements a Beta-Bernoulli bandit framework for online prompt difficulty
prediction, following the MoPPS paper (Qu et al., KDD 2026).

Each prompt is modeled as an arm with latent success rate γ ~ Beta(α, β).
After each rollout, the posterior is updated with observed binary rewards.
Thompson Sampling draws from the posterior to predict difficulty without
requiring any additional LLM inference.

Key features vs IS-based selection:
    - Zero forward-pass cost: posterior update is O(N) addition
    - Explicit exploration-exploitation via Thompson Sampling
    - Temporal discounting (λ) for non-stationary training dynamics
    - Works from epoch 0 with uniform prior (no replay buffer needed)

Mathematical formulation:
    Prior:      γ_τ ~ Beta(α₀, β₀)           (default: uniform Beta(1,1))
    Likelihood: r_{t,j} ~ Bernoulli(γ_τ)      (binary reward per response)
    Update:     α' = λ·α + (1-λ)·α₀ + s_t    (s_t = # successes in k rollouts)
                β' = λ·β + (1-λ)·β₀ + (k-s_t)
    Prediction: γ̂_τ ~ Beta(α_τ, β_τ)          (Thompson Sampling)
"""

import torch
import numpy as np
import os
import json
import time
from collections import defaultdict

COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

_ACC_BINS = [0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.001]
_ACC_LABELS = ["0/8+1/16", "1/8±1/16", "2/8±1/16", "3/8±1/16",
               "4/8±1/16", "5/8±1/16", "6/8±1/16", "7/8±1/16", "8/8-1/16"]


def _ascii_histogram(values, title, n_bins=25, width=60, color=COLOR_GREEN,
                     bins=None, bin_labels=None):
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
    lines.append(f"mean={v_mean:.4f}  std={v_std:.4f}  median={v_med:.4f}")
    lines.append(f"p5={p5:.4f}  p25={p25:.4f}  p75={p75:.4f}  p95={p95:.4f}")
    lines.append(f"min={v_min:.4f}  max={v_max:.4f}")

    print(color + "\n".join(lines) + COLOR_RESET)


class BayesianDataSelector:
    """MoPPS-style Beta-Bernoulli bandit difficulty estimator.

    Outputs predicted labels (estimated success rates) in the same format
    as ISDataSelector / teacher model, for seamless integration with the
    existing softmax/beta sampling selection pipeline.
    """

    def __init__(self, alpha0=1.0, beta0=1.0, decay=0.5,
                 target_gamma=0.5, default_label=0.5, save_dir=None):
        """
        Args:
            alpha0: initial Beta prior α parameter (pseudo-successes).
            beta0: initial Beta prior β parameter (pseudo-failures).
            decay: temporal discounting factor λ ∈ (0, 1].
                Lower values emphasize recent feedback (better for
                non-stationary dynamics). Set to 1.0 to disable.
            target_gamma: target success rate for prompt selection
                (prompts closest to this value are most informative).
            default_label: predicted label for prompts with no history.
            save_dir: directory for per-epoch JSON diagnostics.
        """
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.decay = decay
        self.target_gamma = target_gamma
        self.default_label = default_label
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Per-prompt posterior: {question_index_str: [alpha, beta]}
        self.posteriors = {}
        # Track which indices were updated by Bayesian estimation
        self.last_bayesian_indices = set()

    def update_posteriors(self, rollout_indices, rollout_rewards_per_prompt,
                          group_size):
        """Update Beta posteriors with observed rollout feedback.

        Called AFTER each rollout batch completes and rewards are computed.

        Args:
            rollout_indices: list/array of dataset indices that were rolled out.
            rollout_rewards_per_prompt: dict mapping index -> list of binary
                rewards (one per response in the group).
            group_size: number of responses per prompt (k in the paper).
        """
        n_new = 0
        n_updated = 0
        for idx, rewards in rollout_rewards_per_prompt.items():
            idx_str = str(int(idx))
            s = sum(1 for r in rewards if r > 0)  # successes
            f = len(rewards) - s                    # failures

            if idx_str in self.posteriors:
                a_old, b_old = self.posteriors[idx_str]
                # Temporal discounting (Eq.15 in MoPPS paper)
                a_new = self.decay * a_old + (1 - self.decay) * self.alpha0 + s
                b_new = self.decay * b_old + (1 - self.decay) * self.beta0 + f
                n_updated += 1
            else:
                a_new = self.alpha0 + s
                b_new = self.beta0 + f
                n_new += 1

            self.posteriors[idx_str] = [a_new, b_new]

        n_total = len(self.posteriors)
        print(f"{COLOR_GREEN}[Bayesian] 后验更新: "
              f"{n_new} 新增 + {n_updated} 更新 = "
              f"{n_new + n_updated} 本轮, "
              f"共 {n_total} 题有后验{COLOR_RESET}")

    def estimate_difficulty(self, dataset_size, ref_indices=None,
                            ref_labels=None, epoch=None,
                            fallback_labels=None):
        """Predict difficulty for all prompts via Thompson Sampling.

        Priority:
            1. Reference set -> ground-truth success rate from fresh rollouts
            2. Bayesian posterior -> Thompson Sampling predicted success rate
            3. fallback_labels (e.g. teacher) if provided, else default_label

        Args:
            dataset_size: total number of questions in training set.
            ref_indices: list of question indices with ground-truth labels.
            ref_labels: list of ground-truth success rates for ref questions.
            epoch: current epoch (for diagnostics).
            fallback_labels: optional tensor of shape (dataset_size,) with
                teacher model predictions for uncovered questions.

        Returns:
            torch.Tensor of shape (dataset_size,) with predicted labels.
        """
        start_time = time.time()
        _use_teacher_fallback = fallback_labels is not None

        if _use_teacher_fallback:
            assert fallback_labels.shape == (dataset_size,), \
                f"fallback_labels shape {fallback_labels.shape} != ({dataset_size},)"
            predicted_labels = fallback_labels.clone()
        else:
            # NaN 标记未覆盖的题，防止 default=0.5 在 laplace 采样中
            # 被当作"完美难度"而优先选中
            predicted_labels = torch.full((dataset_size,), float('nan'))

        # --- Thompson Sampling for prompts with posteriors ---
        self.last_bayesian_indices = set()
        n_thompson = 0
        ref_index_set = set()

        if ref_indices is not None and ref_labels is not None:
            for idx, label in zip(ref_indices, ref_labels):
                ref_index_set.add(int(idx))

        for i in range(dataset_size):
            i_str = str(i)
            if i_str in self.posteriors and i not in ref_index_set:
                a, b = self.posteriors[i_str]
                # Thompson Sampling: draw from Beta posterior
                sampled_gamma = np.random.beta(a, b)
                predicted_labels[i] = float(sampled_gamma)
                self.last_bayesian_indices.add(i)
                n_thompson += 1

        # --- Override with ground-truth ref labels (highest priority) ---
        if ref_indices is not None and ref_labels is not None:
            for idx, label in zip(ref_indices, ref_labels):
                predicted_labels[int(idx)] = float(label)

        n_ref = len(ref_index_set)
        n_remaining = dataset_size - n_thompson - n_ref
        elapsed = time.time() - start_time

        _remaining_desc = (f"{n_remaining} Teacher预测"
                           if _use_teacher_fallback
                           else f"{n_remaining} 默认值({self.default_label})")
        print(
            f"{COLOR_GREEN}[Bayesian] 覆盖情况: "
            f"{n_ref} 参考集 + {n_thompson} Thompson采样 + "
            f"{_remaining_desc} "
            f"/ 共 {dataset_size} 题 "
            f"(耗时={elapsed:.3f}s){COLOR_RESET}")

        # --- Diagnostics ---
        if n_thompson > 0:
            ts_vals = np.array([
                float(predicted_labels[i]) for i in self.last_bayesian_indices
            ])
            _ascii_histogram(ts_vals, "Bayesian 难度分布 (Thompson采样成功率)",
                             bins=_ACC_BINS, bin_labels=_ACC_LABELS,
                             width=55, color=COLOR_GREEN)

            # Posterior concentration diagnostics
            alpha_vals = []
            beta_vals = []
            for i in self.last_bayesian_indices:
                a, b = self.posteriors[str(i)]
                alpha_vals.append(a)
                beta_vals.append(b)
            mean_alpha = np.mean(alpha_vals)
            mean_beta = np.mean(beta_vals)
            mean_concentration = mean_alpha + mean_beta
            print(f"{COLOR_GREEN}[Bayesian] 后验统计: "
                  f"mean(α)={mean_alpha:.2f}, mean(β)={mean_beta:.2f}, "
                  f"mean(α+β)={mean_concentration:.2f} "
                  f"(浓度越高=越确信){COLOR_RESET}")

        # Save diagnostics
        if self.save_dir and epoch is not None:
            diagnostics = {
                'epoch': epoch,
                'n_ref': n_ref,
                'n_thompson': n_thompson,
                'n_remaining': n_remaining,
                'n_posteriors_total': len(self.posteriors),
                'use_teacher_fallback': _use_teacher_fallback,
                'decay': self.decay,
                'alpha0': self.alpha0,
                'beta0': self.beta0,
                'elapsed_seconds': elapsed,
            }
            if n_thompson > 0:
                ts_vals_list = [float(predicted_labels[i])
                                for i in self.last_bayesian_indices]
                diagnostics['thompson_stats'] = {
                    'mean': float(np.mean(ts_vals_list)),
                    'std': float(np.std(ts_vals_list)),
                    'min': float(np.min(ts_vals_list)),
                    'max': float(np.max(ts_vals_list)),
                }
            diag_path = os.path.join(
                self.save_dir, f'bayesian_diagnostics_epoch_{epoch}.json')
            with open(diag_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)

        return predicted_labels

    def save_state(self, path):
        """Persist posteriors to disk for resume support."""
        state = {
            'posteriors': self.posteriors,
            'alpha0': self.alpha0,
            'beta0': self.beta0,
            'decay': self.decay,
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        print(f"{COLOR_GREEN}[Bayesian] 保存后验状态: "
              f"{len(self.posteriors)} 题 -> {path}{COLOR_RESET}")

    def load_state(self, path):
        """Restore posteriors from disk."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.posteriors = state.get('posteriors', {})
            print(f"{COLOR_GREEN}[Bayesian] 恢复后验状态: "
                  f"{len(self.posteriors)} 题 <- {path}{COLOR_RESET}")
            return True
        return False
