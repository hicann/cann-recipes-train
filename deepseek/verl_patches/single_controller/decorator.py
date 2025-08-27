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

import sys
from verl.single_controller.base.decorator import (
    Dispatch,
    DISPATCH_MODE_FN_REGISTRY,
    _split_args_kwargs_data_proto,
    dispatch_megatron_compute,
    collect_megatron_compute_data_proto
)


def dispatch_compute_data_proto_with_megatron_dp_ranks(worker_group, *args, **kwargs):
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)

    # NOTE: add megatron dp ranks of each rank to facilitate D2D tensor transfer implementation (reshard)
    args[0].meta_info["global_megatron_dp_ranks"] = [worker_group.get_megatron_rank_info(rank=i).dp_rank
                                                     for i in range(worker_group.world_size)]
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


# apply patch
DISPATCH_MODE_FN_REGISTRY[Dispatch.MEGATRON_COMPUTE_PROTO] = {
    "dispatch_fn": dispatch_compute_data_proto_with_megatron_dp_ranks,
    "collect_fn": collect_megatron_compute_data_proto,
}