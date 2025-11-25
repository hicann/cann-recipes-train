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

#!/bin/bash

## add files
cp -r /workspace/verl/verl ./
cp -r /workspace/verl/recipe/r1_ascend ./
cp /workspace/verl/scripts/converter_hf_to_mcore.py ./
cp /workspace/verl/recipe/dapo/config/* ./verl/trainer/config/
cp /workspace/verl/recipe/dapo/*py ./verl/trainer/
mkdir "megatron"
cp -r /workspace/Megatron-LM/megatron/core ./megatron/core
cp -r /workspace/MindSpeed/mindspeed ./
cp -r /workspace/vllm/vllm ./
cp -r /workspace/vllm-ascend/vllm_ascend ./

## apply patch
for PATCH_FILE in $(find ./patches -type f -name "*.patch"); do
    PATCH_REL_PATH=$(realpath --relative-to=. "$PATCH_FILE")

    git apply -p3 --ignore-whitespace "$PATCH_FILE"

    if [ $? -ne 0 ]; then
        echo "[FAIL]: $PATCH_REL_PATH" >&2
        exit 1
    fi
    echo "[SUCCESS]: $PATCH_REL_PATH"
done
