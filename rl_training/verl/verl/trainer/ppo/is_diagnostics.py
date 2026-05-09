"""Per-step IS diagnostics: accuracy breakdown, histogram printing, and JSON logging."""

import os
import json
import numpy as np

from verl.trainer.ppo.rollout_method import COLOR_RED, COLOR_RESET

_HIST_BINS = [0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.001]
_HIST_LABELS = ["0/8+1/16", "1/8±1/16", "2/8±1/16", "3/8±1/16",
                "4/8±1/16", "5/8±1/16", "6/8±1/16", "7/8±1/16", "8/8-1/16"]


def _compute_hist_dist(arr):
    """Compute histogram distribution over [0,1] accuracy values.
    Returns (dist_str, hist_dict, stats_dict).
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return "无数据", {}, {}
    hist, _ = np.histogram(arr, bins=_HIST_BINS)
    dist_str = ", ".join(f"{l}:{int(c)}" for l, c in zip(_HIST_LABELS, hist))
    hist_dict = {l: int(c) for l, c in zip(_HIST_LABELS, hist)}
    stats = {"min": float(arr.min()), "max": float(arr.max()),
             "mean": float(arr.mean()),
             "std": float(arr.std()) if len(arr) > 1 else 0.0}
    return dist_str, hist_dict, stats


def _append_is_json(path, entry):
    """Append an entry to the IS diagnostics JSON file."""
    data = []
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []
    data.append(entry)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_step_diagnostics(batch, orig_acc, group_size, batch_step, epoch,
                         selection_method, is_random_selection,
                         use_teacher, output_dir,
                         ref_indices=None, predicted_labels=None,
                         is_selector=None, is_default_label_str=''):
    """Log per-step accuracy diagnostics.

    In IS mode (selection_method='is' and not is_random_selection):
      - Breaks down batch questions by estimation source (IS / ref / teacher / default)
      - Prints ASCII histograms of actual & predicted accuracy for each source
      - Computes prediction error metrics (MSE, MAE)
      - Saves a structured JSON entry to is_step_diagnostics.json

    In non-IS mode:
      - Prints a simple accuracy distribution summary.

    Args:
        batch: DataProto with non_tensor_batch containing 'index'.
        orig_acc: numpy array of original per-sample accuracy (len == len(batch)).
        group_size: Number of rollouts per prompt (rollout.n).
        batch_step: Current step within the epoch.
        epoch: Current epoch number.
        selection_method: 'is' or 'teacher'.
        is_random_selection: Whether random selection is active.
        use_teacher: Whether teacher model is available.
        output_dir: Directory for saving JSON diagnostics.
        ref_indices: List of reference sample indices (required in IS mode).
        predicted_labels: Tensor of predicted labels for all dataset items.
        is_selector: ISDataSelector instance (required in IS mode).
        is_default_label_str: Raw config string for is_default_label.
    """
    n_questions = len(batch) // group_size
    accs = np.array([float(orig_acc[i * group_size]) for i in range(n_questions)])

    if selection_method == 'is' and not is_random_selection:
        from verl.trainer.ppo.is_data_selector import _ascii_histogram, _ACC_BINS, _ACC_LABELS

        default_label_clean = str(is_default_label_str).strip().strip("'\"")
        default_label_val = float(default_label_clean) if default_label_clean else 0.5

        batch_idx = batch.non_tensor_batch['index']
        ref_index_set = set(int(ri) for ri in ref_indices) if ref_indices is not None else set()
        q_indices = np.array([int(batch_idx[qi * group_size]) for qi in range(n_questions)])
        ref_mask = np.array([qi in ref_index_set for qi in q_indices])

        is_estimated_set = getattr(is_selector, 'last_is_indices', set())
        is_mask = np.array([qi in is_estimated_set for qi in q_indices])
        remaining = ~is_mask & ~ref_mask
        if use_teacher:
            teacher_mask = remaining
            def_mask = np.zeros_like(ref_mask, dtype=bool)
        else:
            teacher_mask = np.zeros_like(ref_mask, dtype=bool)
            def_mask = remaining

        n_is = int(is_mask.sum())
        n_ref = int(ref_mask.sum())
        n_teacher = int(teacher_mask.sum())
        n_def = int(def_mask.sum())
        acc_is = accs[is_mask] if n_is > 0 else np.array([])
        acc_ref = accs[ref_mask] if n_ref > 0 else np.array([])
        acc_teacher = accs[teacher_mask] if n_teacher > 0 else np.array([])
        acc_def = accs[def_mask] if n_def > 0 else np.array([])
        pred_is = np.array([
            float(predicted_labels[qi])
            for qi, m in zip(q_indices, is_mask) if m
        ]) if n_is > 0 else np.array([])
        pred_ref = np.array([
            float(predicted_labels[qi])
            for qi, m in zip(q_indices, ref_mask) if m
        ]) if n_ref > 0 else np.array([])
        pred_teacher = np.array([
            float(predicted_labels[qi])
            for qi, m in zip(q_indices, teacher_mask) if m
        ]) if n_teacher > 0 else np.array([])

        is_raw_labels = getattr(is_selector, 'last_raw_labels', {})
        old_acc = np.array([
            is_raw_labels.get(str(qi), float('nan'))
            for qi, m in zip(q_indices, is_mask) if m
        ]) if n_is > 0 else np.array([])
        old_acc_valid = old_acc[~np.isnan(old_acc)] if len(old_acc) > 0 else np.array([])
        n_old_valid = len(old_acc_valid)

        is_dist_str, is_hist, is_stats = _compute_hist_dist(acc_is)
        pred_dist_str, pred_hist, pred_stats = _compute_hist_dist(pred_is)
        old_dist_str, old_hist, old_stats = _compute_hist_dist(old_acc_valid)
        ref_dist_str, ref_hist, ref_stats = _compute_hist_dist(acc_ref)
        ref_pred_dist_str, ref_pred_hist, ref_pred_stats = _compute_hist_dist(pred_ref)
        teacher_dist_str, teacher_hist, teacher_stats = _compute_hist_dist(acc_teacher)
        teacher_pred_dist_str, teacher_pred_hist, teacher_pred_stats = _compute_hist_dist(pred_teacher)
        def_dist_str, def_hist, def_stats = _compute_hist_dist(acc_def)

        source_parts = [f"{n_is} IS估计", f"{n_ref} 参考集"]
        if n_teacher > 0:
            source_parts.append(f"{n_teacher} Teacher预测")
        if n_def > 0:
            source_parts.append(f"{n_def} 默认值({default_label_val})")
        print(
            f"{COLOR_RED}[Step {batch_step} IS→Fresh] "
            f"{n_questions} 题中 {' + '.join(source_parts)}{COLOR_RESET}")
        _ascii_histogram(
            acc_is,
            f"Step {batch_step} IS实际正确率",
            bins=_ACC_BINS, bin_labels=_ACC_LABELS,
            width=55, color=COLOR_RED)
        _ascii_histogram(
            pred_is,
            f"Step {batch_step} IS预测正确率",
            bins=_ACC_BINS, bin_labels=_ACC_LABELS,
            width=55, color=COLOR_RED)
        _ascii_histogram(
            old_acc_valid,
            f"Step {batch_step} IS旧策略正确率",
            bins=_ACC_BINS, bin_labels=_ACC_LABELS,
            width=55, color=COLOR_RED)
        if n_teacher > 0:
            _ascii_histogram(
                acc_teacher,
                f"Step {batch_step} Teacher实际正确率",
                bins=_ACC_BINS, bin_labels=_ACC_LABELS,
                width=55, color=COLOR_RED)
            _ascii_histogram(
                pred_teacher,
                f"Step {batch_step} Teacher预测正确率",
                bins=_ACC_BINS, bin_labels=_ACC_LABELS,
                width=55, color=COLOR_RED)
        if n_def > 0:
            _ascii_histogram(
                acc_def,
                f"Step {batch_step} 默认值选出",
                bins=_ACC_BINS, bin_labels=_ACC_LABELS,
                width=55, color=COLOR_RED)

        err_parts = []
        is_mse = is_mae = teacher_mse = teacher_mae = float('nan')
        old_vs_new_mse = old_vs_new_mae = float('nan')
        if n_is > 0:
            is_err = pred_is - acc_is
            is_mse = float(np.mean(is_err ** 2))
            is_mae = float(np.mean(np.abs(is_err)))
            err_parts.append(
                f"IS预测vs实际(n={n_is}): MSE={is_mse:.4f} MAE={is_mae:.4f}")
        if n_old_valid > 0:
            old_matched = old_acc_valid
            new_matched = acc_is[:n_old_valid] if n_old_valid == n_is else acc_is[~np.isnan(old_acc)]
            old_new_err = old_matched - new_matched
            old_vs_new_mse = float(np.mean(old_new_err ** 2))
            old_vs_new_mae = float(np.mean(np.abs(old_new_err)))
            err_parts.append(
                f"IS旧策略vs实际(n={n_old_valid}): MSE={old_vs_new_mse:.4f} MAE={old_vs_new_mae:.4f}")
        if n_teacher > 0:
            teacher_err = pred_teacher - acc_teacher
            teacher_mse = float(np.mean(teacher_err ** 2))
            teacher_mae = float(np.mean(np.abs(teacher_err)))
            err_parts.append(
                f"Teacher预测vs实际(n={n_teacher}): MSE={teacher_mse:.4f} MAE={teacher_mae:.4f}")
        if err_parts:
            print(
                f"{COLOR_RED}[Step {batch_step} 预测误差] "
                f"{' | '.join(err_parts)}{COLOR_RESET}")

        is_json_path = os.path.join(output_dir, 'is_step_diagnostics.json')
        step_entry = {
            "epoch": int(epoch),
            "step": int(batch_step),
            "is_actual_accuracy": {
                "n_questions": n_is,
                "histogram": is_hist,
                "stats": is_stats,
            } if n_is > 0 else None,
            "is_predicted_accuracy": {
                "n_questions": n_is,
                "histogram": pred_hist,
                "stats": pred_stats,
            } if n_is > 0 else None,
            "is_gt_of_old_policy": {
                "n_questions": n_old_valid,
                "histogram": old_hist,
                "stats": old_stats,
            } if n_old_valid > 0 else None,
            "ref_actual_accuracy": {
                "n_questions": n_ref,
                "histogram": ref_hist,
                "stats": ref_stats,
            } if n_ref > 0 else None,
            "ref_predicted_accuracy": {
                "n_questions": n_ref,
                "histogram": ref_pred_hist,
                "stats": ref_pred_stats,
            } if n_ref > 0 else None,
            "teacher_actual_accuracy": {
                "n_questions": n_teacher,
                "histogram": teacher_hist,
                "stats": teacher_stats,
            } if n_teacher > 0 else None,
            "teacher_predicted_accuracy": {
                "n_questions": n_teacher,
                "histogram": teacher_pred_hist,
                "stats": teacher_pred_stats,
            } if n_teacher > 0 else None,
            "default_accuracy": {
                "n_questions": n_def,
                "histogram": def_hist,
                "stats": def_stats,
            } if n_def > 0 else None,
            "prediction_error": {
                "is_mse": is_mse if n_is > 0 else None,
                "is_mae": is_mae if n_is > 0 else None,
                "is_n": n_is,
                "old_vs_new_mse": old_vs_new_mse if n_old_valid > 0 else None,
                "old_vs_new_mae": old_vs_new_mae if n_old_valid > 0 else None,
                "old_vs_new_n": n_old_valid,
                "teacher_mse": teacher_mse if n_teacher > 0 else None,
                "teacher_mae": teacher_mae if n_teacher > 0 else None,
                "teacher_n": n_teacher,
            },
        }
        if is_selector is not None and hasattr(is_selector, 'last_distribution'):
            step_entry["is_difficulty_distribution"] = is_selector.last_distribution
        _append_is_json(is_json_path, step_entry)
    else:
        all_dist_str, _, _ = _compute_hist_dist(accs)
        print(
            f"{COLOR_RED}[Step {batch_step} Fresh 实际正确率] 共 {n_questions} 题\n"
            f"  分布: {all_dist_str}{COLOR_RESET}")
