# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl_patches.trainer.ppo.ray_trainer import compute_advantage, _validate_config_wrapper, init_workers, fit


def verl_trainer_adaptation():
    ray_trainer.compute_advantage = compute_advantage
    RayPPOTrainer._validate_config = _validate_config_wrapper(RayPPOTrainer._validate_config)
    RayPPOTrainer.init_workers = init_workers
    RayPPOTrainer.fit = fit


verl_trainer_adaptation()
