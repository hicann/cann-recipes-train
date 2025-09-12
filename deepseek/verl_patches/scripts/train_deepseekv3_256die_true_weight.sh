# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

pkill -9 python
ray stop --force

cp ./verl_patches/trainer/config/ppo_mindspeed_trainer.yaml ./verl/trainer/config/
cp ./verl_patches/trainer/main_ppo_npu.py ./verl/trainer/

export RAY_DEDUP_LOGS=0            # 0: disable ray's log folding 1: enable ray's log folding
export HYDRA_FULL_ERROR=1          # display the accurate error stack

SOCKET_IFNAME="Your SOCKET IFNAME"

ulimit -n 32768
mkdir logs

export HCCL_IF_BASE_PORT=24703
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
export MALLOC_MMAP_THRESHOLD_=512768
export LCAL_COMM_ID=127.0.0.1:27001


NNODES=16                          # number of nodes
NPUS_PER_NODE=16                   # the number of npus for each node
MASTER_ADDR="IP FOR MASTER NODE"   # modify it to correspond to the IP of the master node
SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"  # modify it to the communication network card of the current node
# obtain the current node IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

# configure environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export WORLD_SIZE=$(($NNODES*$NPUS_PER_NODE))
export MASTER_PORT=29444

export PROFILE_UPDATE=0               # enable profiling sampling in the update stage
export ASCEND_LAUNCH_BLOCKING=0       # debug usage, which seriously affects performance after use, but the error stack is accurate

export ASCEND_GLOBAL_EVENT_ENABLE=0   # whether to display the Ascend EVENT log; 1 is for display
export ASCEND_SLOG_PRINT_TO_STDOUT=0  # does the Ascend log screen out? 0 does not screen. By default, it is at /root/ascend/log. 1 directly screens
export ASCEND_GLOBAL_LOG_LEVEL=3      # log level: 3: ERROR  2:EVENT  1:INFO   0:DEBUG

export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export HCCL_IF_BASE_PORT=64247
export HCCL_HOST_SOCKET_PORT_RANGE="auto"

export PRINT_MEMORY=1                 # print the existing records at each stage to assist in locating OOM issues

# If set to 1, it will use collective communication to directly transfer data between stages,
# which can greatly optimize the time consumption of ray sending megatron computing tasks
# NOTE: this is not supported when using balance batch for now.
export D2D_DATA_TRANSFER=1

export VLLM_USE_V1=1                            # use the V1 engine of vLLM
export VLLM_ENABLE_GRAPH_MODE=1                 # enable vLLM graph mode
export HCCL_OP_EXPANSION_MODE=AIV               # enable the communication mode of AIV
export VLLM_ENABLE_MC2=1                        # enable MC2 communication
export VLLM_DP_SIZE=128                         # configure the DP size of vLLM
export VLLM_ENABLE_FIX_ROUTE=0                  # 0:Turn off the fixed route of vLLM; 1:enable the fixed route of vLLM
# under the configuration of the vLLM log level of INFO, enable this configuration, print the time of prefill and decode
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1

export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export VLLM_ASCEND_ACL_OP_INIT_MODE=0
export VLLM_VERSION='0.9.1'                     # the version of vLLM
export TASK_QUEUE_ENABLE=2                      # enable level2 optimization of the sent queue of the ascend operator
export HCCL_BUFFSIZE=300                        # the buffer size of HCCL


if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # the master node starts
  ray start --head --port 6766 --dashboard-host=0.0.0.0 --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # determine whether device_count is equal to NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash ./verl_patches/scripts/internal/train_grpo_deepseekv3_256die_true_weight.sh
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # the child node attempts to register ray with the master node until successful
  while true; do
      # try to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # check if the connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 999999