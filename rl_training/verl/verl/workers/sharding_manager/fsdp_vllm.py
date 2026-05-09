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

import os
import gc
import ctypes
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.third_party.vllm import vllm_version
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _force_release_cpu_memory():
    """Aggressively return freed CPU memory to the OS.
    
    1. gc.collect() — release Python objects holding C memory
    2. malloc_trim(0) — tell glibc to return free chunks to OS
    3. If MALLOC_TRIM is not enough (glibc fragmentation), this at least
       ensures Python-level garbage is collected.
    """
    gc.collect()
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:
        pass


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # NOTE: We need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator.
        torch.cuda.empty_cache()

        # Log CPU RSS for memory debugging
        try:
            import psutil
            _rss_gb = psutil.Process().memory_info().rss / 1e9
            # print(f"[MEM] sharding_manager __enter__: RSS={_rss_gb:.2f}GB")
        except Exception:
            pass

        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            # spmd mode: vllm 0.7+
            self.inference_engine.wake_up()
            world_size = torch.distributed.get_world_size()
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if world_size != 1 else param) for name, param in params.items()))
            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        _force_release_cpu_memory()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)
        # Log CPU RSS after cleanup
        try:
            import psutil
            _rss_gb = psutil.Process().memory_info().rss / 1e9
            # print(f"[MEM] sharding_manager __enter__ done: RSS={_rss_gb:.2f}GB")
        except Exception:
            pass

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            # spmd mode: vllm 0.7+
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()
        _force_release_cpu_memory()
        # Log CPU RSS after cleanup
        try:
            import psutil
            _rss_gb = psutil.Process().memory_info().rss / 1e9
            # print(f"[MEM] sharding_manager __exit__: RSS={_rss_gb:.2f}GB")
        except Exception:
            pass

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def _get_tp_group(self):
        """Get the tensor parallel torch ProcessGroup from vllm's GroupCoordinator."""
        coordinator = vllm_ps.get_tensor_model_parallel_group()
        # vllm 0.7+: returns GroupCoordinator, extract the underlying ProcessGroup
        return coordinator.device_group

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=self._get_tp_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_group().first_rank,
                              group=self._get_tp_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
