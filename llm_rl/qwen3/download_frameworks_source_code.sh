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

set -ex

HOME_DIR=$(pwd)

mkdir -p /workspace && cd /workspace

git clone https://gitcode.com/GitHub_Trending/ve/verl.git
cd verl
git checkout v0.6.0
git cherry-pick -n -X theirs 448c6c3
cd -

git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd -

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout f6688c61bcfe45243ee5eb34c6f013b1e06eca81
cd -

git clone https://gitcode.com/GitHub_Trending/vl/vllm.git
cd vllm
git checkout v0.14.1
cd -

git clone https://gitcode.com/gh_mirrors/vl/vllm-ascend.git
cd vllm-ascend
git checkout v0.14.0rc1
git submodule update --init --recursive
cd -

cd $HOME_DIR