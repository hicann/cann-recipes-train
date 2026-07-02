# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from torchtitan_npu.config.configs import CheckpointConfig, ParallelismConfig
from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.models.qwen3 import model_registry

TRAIN_DATA = os.environ.get(
    "TRAIN_DATA",
    "./assets/medical_r1/train.jsonl",
)
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "assets", "hf", "Qwen3-30B-A3B",
    ),
)

# 5 epochs (~31 steps/epoch)
_STEPS = 156

_CONVERTERS = ModelConvertersContainer.Config(
    converters=[
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
    ],
)


def _process_sample(sample):
    output = f"<think>\n{sample['think']}\n</think>\n\n{sample['answer']}"
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": output},
    ]


def _base_config() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path=MODEL_DIR,
        model_spec=model_registry("30B-A3B"),
        model_converters=_CONVERTERS,
        optimizer=OptimizersContainer.Config(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=5,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=_STEPS,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=1,
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=2,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={"data_files": TRAIN_DATA, "split": "train"},
            sample_processor=_process_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="./checkpoint_medical",
            initial_load_in_hf=True,
            initial_load_path=MODEL_DIR,
            interval=50,
            keep_latest_k=2,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
            async_mode="async",
        ),
    )


def sft_qwen3_30ba3b_medical() -> Trainer.Config:
    """Medical SFT — BSND attention (default)."""
    return _base_config()


def sft_qwen3_30ba3b_medical_tnd() -> Trainer.Config:
    """Medical SFT — NPUVarlenAttention (TND)."""
    from torchtitan_npu.models.qwen3.tnd_config import _enable_npu_varlen_attention

    config = _base_config()
    config.model_spec = _enable_npu_varlen_attention(config.model_spec)
    return config
