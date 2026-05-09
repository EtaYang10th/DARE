"""
Entry-point for PPO (GAE + Critic) training.

Mirrors ``main_ppo.py`` but:
  1. Registers ``Role.Critic`` in the worker mapping.
  2. Instantiates ``RayPPOGAETrainer`` instead of ``RayPPOTrainer``.

To revert, simply delete this file — ``main_ppo.py`` is untouched.

Usage:
    python3 -m verl.trainer.main_ppo_gae  algorithm.adv_estimator=gae ...
"""

from verl import DataProto
import torch
import ray
import hydra

# Reuse reward logic from the original entry-point
from verl.trainer.main_ppo import RewardManager
from verl.trainer.ppo.ray_ppo_trainer import RayPPOGAETrainer
from verl.trainer.ppo.teacher_utils import TeacherModelWorker


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
            }
        })
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    # ---- Worker classes ----
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # ---- Role → Worker mapping ----
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),          # always for PPO
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,                     # colocate with actor
    }

    # Ref policy (only when KL loss is actually used)
    kl_loss_coef = config.actor_rollout_ref.actor.get('kl_loss_coef', 0)
    if kl_loss_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # Teacher model (for data selection)
    selection_method = config.data.get('selection_method', 'teacher')
    _is_def_str = str(config.data.get('is_default_label', '')).strip().strip("'\"")
    need_teacher = (selection_method == 'teacher') or \
                   (selection_method == 'is' and not _is_def_str)
    if not config.data.random_selection and need_teacher:
        role_worker_mapping[Role.Teacher] = ray.remote(TeacherModelWorker)
        mapping[Role.Teacher] = global_pool_id

    # Reward model (optional)
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # ---- Reward functions ----
    reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=0,
        format_reward=config.reward_model.format_reward)
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=1,
        format_reward=config.reward_model.format_reward)

    # ---- Build & run trainer ----
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOGAETrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
