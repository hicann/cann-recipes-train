# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import dataclasses


@dataclasses.dataclass
class RolloutRebalanceConfig:
    enable = True  # RolloutRebalance特性总开关
    check_interval = 1000  # 间隔多少个step进行一次rebalance检查

    multi_graph = True  # 是否开启多档位编图，如果关闭，rebalance依然会按预编图的档位做均衡调度，但是不会形成明显的性能收益
    graph_batch_sizes = [64, 32, 16, 8, 4]  # 预编图的档位设置

    profile = True  # 是否打印过程中的性能数据
    profile_interval = 100   # 打印间隔步长
