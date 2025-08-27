# DeepSeekV3 RLHF on NPU
本sample主要是DeepseekV3模型在NPU上进行RL后训练的适配点介绍，基于[veRL开源框架](https://github.com/volcengine/verl)，通过一系列补丁对开源代码进行面向NPU的适配改造。

---

# 1. Quick Start
##  1.1 镜像下载&创建
下载vLLM-Ascend提供的镜像，可以快速配置环境：
```shell
镜像下载命令：docker pull quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler
```
镜像使用：
```shell
# 创建docker，以A3服务器为例，自行修改docker_name
docker run -itd \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci8 \
--device=/dev/davinci9 \
--device=/dev/davinci10 \
--device=/dev/davinci11 \
--device=/dev/davinci12 \
--device=/dev/davinci13 \
--device=/dev/davinci14 \
--device=/dev/davinci15 \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /data/:/data/ \
-v /home/:/home/ \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /dev/shm:/dev/shm \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--net=host \
--name docker_name \
--privileged quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler /bin/bash
# 启动容器
docker exec -it -u root docker_name bash
```
##  1.2 安装依赖
在执行本sample之前，需安装配套的昇腾软件栈及其他软件，列表如下：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">25.2.0</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">8.2.RC1</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>torch</td>
    <td rowspan="2">2.5.1</td>
  </tr>
  <tr>
    <td>torch_npu</td>
  </tr>
  <tr>
    <td>apex</td>
    <td rowspan="1">0.1</td>
  </tr>
  <tr>
    <td>ray</td>
    <td>2.42.1</td>
  </tr>
  <tr>
    <td>vllm</td>
    <td>0.9.1</td>
  </tr>
  <tr>
    <td>Megatron-LM</td>
    <td>core_r0.8.0</td>
  </tr>
</table>

### 1.2.1 CANN安装

```shell
# 驱动固件安装
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full

# 其他CANN包安装
bash Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --install
bash Atlas-A3-cann-kernels_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/nnal/atb/set_env.sh
```

[昇腾辅助软件](https://gitee.com/ascend/pytorch#昇腾辅助软件)中有更多关于PyTorch和CANN的版本信息。

### 1.2.2 PTA安装

```shell
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1.dev*.whl
```


## 1.3 准备源码
```shell
# 本sample所在代码仓
git clone https://gitee.com/ascend/cann-recipes.git

# veRL框架
git clone https://github.com/volcengine/verl.git    # 从github下载，请确保网络能访问
cd verl
git checkout 54c9b7364c2d188b2ba4107404cfa3c2b446df19
git fetch origin pull/3054/head && git merge FETCH_HEAD
cp -r verl ../cann-recipes/training/rl/deepseekv3/
cd .. 

# vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1 
cp -r vllm ../cann-recipes/training/rl/deepseekv3/
cd ..

# vLLM-Ascend
git clone -b main https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout e1d282d7cc017f7e8075074a6981532045801a73   # v0.9.1-dev branch
git fetch origin pull/1474/head && git merge FETCH_HEAD # 拉取额外优化代码
git fetch origin pull/2058/head && git merge FETCH_HEAD # 拉取额外优化代码
cp -r vllm-ascend ../cann-recipes/training/rl/deepseekv3/
cd ..

# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git # 从github下载，请确保网络能访问  
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../cann-recipes/training/rl/deepseekv3/
cd ..

# MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout v2.0.0_core_r0.8.0     # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../cann-recipes/training/rl/deepseekv3/
cd ..

# MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout v2.0.0
cp -r mindspeed_llm ../cann-recipes/training/rl/deepseekv3/
cd ..

# 进入本sample目录
cd ./cann-recipes/training/rl/deepseekv3/
pip install -r requirements.txt
```

## 1.4 侵入式修改veRL代码
```python
# 在verl/workers/megatron_workers.py: line21 处插入以下代码：
from verl_patches import prelude_patch
```

## 1.5 准备训练数据集与模型
数据集放入 ./data, 数据集准备参考: [veRL官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
模型放入 ./DeepSeek-V3-hf, 模型下载地址：[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## 1.6 执行RL后训练
```shell
# 本sample目录下启动DeepSeekV3的RL后训练
bash ./verl_patches/scripts/train_deepseekv3_256die_random_init.sh # 基于随机权重的训练脚本 
```
