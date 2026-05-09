"""
PPO (GAE + Critic) trainer — zero-intrusion extension of RayPPOTrainer.

This module inherits from RayPPOTrainer and adds critic-aware logic
without modifying the parent class.  To revert, simply delete this file.

Key difference from GRPO path:
  - Requires algorithm.adv_estimator = 'gae'
  - Before compute_advantage(), re-computes values with the *current*
    critic for the entire batch (including replay-buffer data).
"""

from __future__ import annotations

import verl.trainer.ppo.ray_trainer as _rt_mod
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage


class RayPPOGAETrainer(RayPPOTrainer):
    """Drop-in replacement that activates the PPO / GAE code path.

    All data-selection, replay-buffer, rollout-strategy logic is inherited
    unchanged from ``RayPPOTrainer``.
    """

    # ------------------------------------------------------------------
    # init_workers: safety assertion only
    # ------------------------------------------------------------------
    def init_workers(self):
        assert self.config.algorithm.adv_estimator == 'gae', (
            "RayPPOGAETrainer requires algorithm.adv_estimator='gae', "
            f"got '{self.config.algorithm.adv_estimator}'"
        )
        super().init_workers()
        assert self.use_critic, (
            "Critic worker was not created — check that Role.Critic is "
            "registered in role_worker_mapping and resource_pool mapping."
        )

    # ------------------------------------------------------------------
    # fit: monkey-patch compute_advantage so that values are always
    #      (re-)computed by the *current* critic before GAE runs.
    #
    #  Why monkey-patch instead of overriding fit()?
    #    fit() is ~900 lines; copying it would be unmaintainable.
    #    compute_advantage is a module-level function called exactly once
    #    inside fit() (L2282).  Patching it is surgical and safe.
    # ------------------------------------------------------------------
    def fit(self):
        _original_fn = _rt_mod.compute_advantage
        trainer_ref = self                       # captured by closure

        def _patched_compute_advantage(data, adv_estimator, **kwargs):
            """Ensure batch.batch['values'] is fresh before GAE."""
            if adv_estimator == 'gae':
                values = trainer_ref.critic_wg.compute_values(data)
                data = data.union(values)
            return _original_fn(data, adv_estimator, **kwargs)

        _rt_mod.compute_advantage = _patched_compute_advantage
        try:
            super().fit()
        finally:
            # Restore original function no matter what
            _rt_mod.compute_advantage = _original_fn
