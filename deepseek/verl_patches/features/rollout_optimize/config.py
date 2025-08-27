# coding=utf-8
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

import dataclasses


@dataclasses.dataclass
class RolloutRebalanceConfig:
    enable = True  # RolloutRebalance特性总开关
    check_interval = 1000  # 间隔多少个step进行一次rebalance检查

    multi_graph = True  # 是否开启多档位编图，如果关闭，rebalance依然会按预编图的档位做均衡调度，但是不会形成明显的性能收益
    graph_batch_sizes = [64, 32, 16, 8, 4]  # 预编图的档位设置

    profile = True  # 是否打印过程中的性能数据
    profile_interval = 100   # 打印间隔步长
