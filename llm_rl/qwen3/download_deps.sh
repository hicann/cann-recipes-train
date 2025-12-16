set -ex

HOME_DIR=$(pwd)

mkdir -p /workspace && cd /workspace

# 下载verl源码
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.6.0
git fetch origin pull/3427/head
git cherry-pick -n -X theirs 448c6c3
cd -

# 下载Megatron-LM源码
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd -

# 下载MindSpeed源码
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout f6688c61bcfe45243ee5eb34c6f013b1e06eca81
cd -

# 下载vLLM源码
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.0
cd -

# 下载vLLM-Ascend源码
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.11.0rc0
cd -

cd $HOME_DIR