# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
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

# The host machine for building this container image needs internet access
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11

# Install OS dependencies (using Huawei open source mirror)
RUN apt-get update && \
    apt-get install -y \
    # utils
    ca-certificates vim curl \
    gcc g++ git wget make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev \
    # MindSpore 2.2.0
    libgmp-dev \
    # MindSpeed-RL
    patch net-tools libjemalloc2 && \
    apt-get clean && \
    # Change permissions of CANN installation parent directory to allow ma-user write access
    chmod o+w /usr/local

RUN useradd -m -d /home/ma-user -s /bin/bash -g 100 -u 1000 ma-user

########################################################################

# Set default user and working directory for the container image
USER ma-user
WORKDIR /home/ma-user

# Download installation files to /tmp directory in the base container image
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_25.3.1-1-Linux-aarch64.sh -O /tmp/Miniconda3-py311_25.3.1-1-Linux-aarch64.sh

# Install Miniconda3 to /home/ma-user/miniconda3 directory in the base container image
RUN bash /tmp/Miniconda3-py311_25.3.1-1-Linux-aarch64.sh -b -p /home/ma-user/miniconda3

ENV PATH=$PATH:/home/ma-user/miniconda3/bin

# Install vllm==v0.11.0
RUN source /home/ma-user/miniconda3/bin/activate && \
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    git clone -b v0.11.0 --depth 1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    VLLM_TARGET_DEVICE=empty pip install -e . && \
    cd ..

# Install vllm-ascend==v0.11.0
RUN source /home/ma-user/miniconda3/bin/activate && \
    git clone -b v0.11.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git

# Switch to root user for installation
USER root

RUN source /home/ma-user/miniconda3/bin/activate && \
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    cd /home/ma-user/vllm-ascend && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    export SOC_VERSION=Ascend910B3 && \
    # https://github.com/vllm-project/vllm-ascend/pull/1242#discussion_r2149670165
    pip install -e . && \
    cd ..

# Switch back to ma-user
USER ma-user

# Install torch==2.7.1 and torch-npu==2.7.1
RUN source /home/ma-user/miniconda3/bin/activate && \
    pip install --no-cache-dir torch==2.7.1 torch-npu==2.7.1

########################################################################

# Install verl dependencies
RUN cd /home/ma-user/ && \
    source /home/ma-user/miniconda3/bin/activate && \
    git clone https://github.com/volcengine/verl.git && \
    cd verl && \
    git checkout c651b7b4207e408875f132c4226969ef3495d408 && \
    pip install -r requirements-npu.txt && \
    # Fix ray==2.46.0 startup issue
    pip install click==8.2.1 && \
    # Fix local code execution dependency issue
    # Fix pyext installation issue on python3.11
    # https://github.com/volcengine/verl/blob/c651b7b4207e408875f132c4226969ef3495d408/scripts/install_vllm_sglang_mcore.sh#L22
    pip install git+https://github.com/ShaohonChen/PyExt.git@py311support && \
    pip install -e . && \
    cd ..

########################################################################

# Cleanup
USER root

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /root/.cache && \
    rm -rf /root/ascend && \
    rm -rf /home/ma-user/.cache
