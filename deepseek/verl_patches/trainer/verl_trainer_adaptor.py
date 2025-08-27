# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl_patches.trainer.ppo.ray_trainer import compute_advantage, _validate_config_wrapper, init_workers, fit


def verl_trainer_adaptation():
    ray_trainer.compute_advantage = compute_advantage
    RayPPOTrainer._validate_config = _validate_config_wrapper(RayPPOTrainer._validate_config)
    RayPPOTrainer.init_workers = init_workers
    RayPPOTrainer.fit = fit


verl_trainer_adaptation()
