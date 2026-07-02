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

set -ex

HOME_DIR=$(pwd)

rm -rf /workspace
mkdir -p /workspace && cd /workspace

git clone https://gitcode.com/GitHub_Trending/ve/verl.git
cd verl
git checkout e9aa879bc61821621a36881ea305eaa0785520c1
cd -

git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git
cd torchtitan
git checkout ac13e536c84e7f6647b14fa9375c3c8a8a2b8578
cd -

git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
git checkout 29bbc8ba5bee5daf63f8a0c09512038449ffaf37
cd -

git clone https://gitcode.com/gh_mirrors/vl/vllm-ascend.git
cd vllm-ascend
git checkout v0.15.0rc1
git submodule update --init --recursive
cd -

cd $HOME_DIR
