# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import logging
from typing import Optional, List

import torch
import torch_npu

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class NPUProfiler:
    def __init__(
        self,
        stage_name: str,
        warm_up: int,
        active: int,
        ranks: Optional[List[int]] = None,
        with_stack: bool = False,
    ):
        """
        Initialize an NPU profiler instance if needed.

        Parameters:
        stage_name : str
            Descriptive name for the profiling stage (e.g., "update_actor")
        warm_up : int
            Number of initial iterations to skip before starting profiling (to avoid capturing startup overhead)
        active : int
            Number of iterations to capture profiling data after warm-up
        ranks : Optional[List[int]], optional
            Device ranks to profile (for multi-device setups), defaults to all available devices
        with_stack : bool, optional
            Flag to enable call stack recording during profiling (adds significant overhead), defaults to False
        """
        # check if profile
        self.do_profile = False
        if ranks is None or int(torch.distributed.get_rank()) in ranks:
            self.do_profile = True
        else:
            return

        # prepare for profiling
        self.rank = torch.distributed.get_rank()
        self.stage_name = stage_name
        prof_dir = f"./profiling/{stage_name}"
        if not os.path.exists(prof_dir):
            os.makedirs(prof_dir)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
        )
        self.prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=warm_up, active=active, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            experimental_config=experimental_config)
        logger.info(
            f"[NPU Profiler] rank_{self.rank} info: stage: {stage_name}, active: {active}, "
            f"warm_up: {warm_up}, ranks: {ranks}, with_stack {with_stack}, file_path: {prof_dir}"
        )

    def start(self):
        if self.do_profile:
            logger.info(f"[NPU Profiler] started for rank_{self.rank} on stage {self.stage_name}")
            self.prof.start()

    def step(self):
        if self.do_profile:
            logger.info(f"[NPU Profiler] stepped for rank_{self.rank} on stage {self.stage_name}")
            self.prof.step()

    def stop(self):
        if self.do_profile:
            logger.info(f"[NPU Profiler] stopped for rank_{self.rank} on stage {self.stage_name}")
            self.prof.stop()
