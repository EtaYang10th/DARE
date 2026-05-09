"""Sample attenuation: track and attenuate samples whose accuracy falls outside a target range."""

from collections import Counter
from verl.trainer.ppo.rollout_method import COLOR_RED, COLOR_RESET


def attenuation_update(attenuation_counts, sample_attenuation, sample_key, accuracy):
    """Update attenuation count for a sample based on its rollout accuracy.

    Args:
        attenuation_counts: Dict mapping sample_key -> int count (modified in place).
        sample_attenuation: Tuple (lo, hi) defining the acceptable accuracy range.
        sample_key: String key identifying the sample.
        accuracy: Float accuracy value of the sample.

    Returns True if the count was incremented (outside range), False otherwise.
    """
    lo, hi = sample_attenuation
    if accuracy < lo or accuracy > hi:
        attenuation_counts[sample_key] = attenuation_counts.get(sample_key, 0) + 1
        return True
    return False


def print_attenuation_stats(attenuation_counts, sample_attenuation, decay_factor=0.5):
    """Print distribution of attenuation counts and corresponding selection probabilities.

    Args:
        attenuation_counts: Dict mapping sample_key -> int count.
        sample_attenuation: Tuple (lo, hi) defining the acceptable accuracy range.
        decay_factor: Multiplicative decay applied per attenuation count.
    """
    if not attenuation_counts:
        print(
            f"{COLOR_RED}[Attenuation] Range: {list(sample_attenuation)}, "
            f"衰减因子: {decay_factor}, tracked samples: 0{COLOR_RESET}"
        )
        return
    count_dist = Counter(attenuation_counts.values())
    print(
        f"{COLOR_RED}[Attenuation] Range: {list(sample_attenuation)}, "
        f"衰减因子: {decay_factor}, "
        f"tracked samples: {len(attenuation_counts)}{COLOR_RESET}"
    )
    for c in sorted(count_dist.keys()):
        prob = decay_factor ** c
        print(
            f"{COLOR_RED}  count={c} (select_prob={prob:.6f}): "
            f"{count_dist[c]} samples{COLOR_RESET}"
        )
