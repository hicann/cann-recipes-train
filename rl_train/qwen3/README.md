# Qwen3-235B-A22B RL训练优化实践样例

## 概述
本样例针对Qwen3-235B-A22B模型，基于[veRL开源框架](https://github.com/volcengine/verl)，使用veRL原生支持的MindSpeed和vLLM-Ascend框架，完成RL训练全流程的优化适配。

1. **GRPO算法RL训练**：基于Atlas A3 64卡集群，加载真实权重，使用deepscaler数据集，Prefill/Decode阶段长度分别为2K与32K，最优系统吞吐可达到120TPS/卡，性能测试结果如下：

   | 基础模型             | 机器型号     | GBS | n_samples | step | max_prompt_length | max_response_length | 端到端TPS |
   |---------------------|----------|-----|-----------|--------- | ----------|------------|---------|
   | Qwen3-235B-A22B    | Atlas A3 64卡 | 512 | 16        | 1 |  2048             | 32768      | 120     |


2. **DAPO算法RL训练**：基于Atlas A3 64卡集群，加载真实权重，使用dapo-math-17k数据集，Prefill/Decode阶段长度分别为2K与34K，性能测试结果如下：

   | 基础模型             | 机器型号     | GBS | n_samples | step | max_prompt_length | max_response_length | 首步推理时间 | num_gen_batches |
   |---------------------|----------|-----|-----------|--------- | ----------|------------|---------|---------|
   | Qwen3-235B-A22B    | Atlas A3 64卡 | 128 | 16        | 1 |  2048             | 34816      | 6620s     | 2 |

## 硬件要求
产品型号：Atlas A3 系列

最少卡数：64张A3

## 文件说明
|上级目录|文件路径|说明|
|-------|--------|--------|
|megatron|[0001-megatron-bugfix-state_ten-verification.patch](patches/megatron/0001-megatron-bugfix-state_ten-verification.patch)|在处理优化器状态时新增空值判断，避免因空值导致的运行异常|
|megatron|[0002-megatron-feature-enable_hdp.patch](patches/megatron/0002-megatron-feature-enable_hdp.patch)|在ROPE中增加HDP相关处理逻辑，`USE_HDP`开启时，使能HDP功能|
|mindspeed|[0001-mindspeed-bugfix-builder.patch](patches/mindspeed/0001-mindspeed-bugfix-builder.patch)|兼容openeuler24.03版本下编译头文件缺失|
|mindspeed|[0002-mindspeed-feature-enable_hdp.patch](patches/mindspeed/0002-mindspeed-feature-enable_hdp.patch)|在Ring Attention中增加HDP相关处理逻辑，`USE_HDP`开启时，使能HDP功能|
|verl|[0001-verl-feature-enable_alltoall_overlap.patch](patches/verl/0001-verl-feature-enable_alltoall_overlap.patch)|根据`USE_ALLTOALL_OVERLAP`环境变量调整权重加载逻辑|
|verl|[0002-verl-feature-set_use_tqdm_true.patch](patches/verl/0002-verl-feature-set_use_tqdm_true.patch)|开启tqdm进度条，便于实时观测推理进度|
|verl|[0003-verl-feature-recompute_old_log_prob.patch](patches/verl/0003-verl-feature-recompute_old_log_prob.patch)|对于GRPO on-policy算法，可以使用`log_prob.detach()`代替`old_log_prob`减少一次前向计算，添加控制参数配置和开启免计算时`ppo_epochs=1`校验|
|verl|[0004-verl-feature-data_rebalance.patch](patches/verl/0004-verl-feature-data_rebalance.patch)|为缓解多卡推理时长尾负载不均，新增`data rebalance`，通过配置项控制启用；启用时执行固定重排序确保多卡之间推理prompt尽可能均衡|
|verl|[0005-verl-feature-moe_alltoallv.patch](patches/verl/0005-verl-feature-moe_alltoallv.patch)|支持EP使用ALLToALLV做无通信冗余的reshard，通过专家参数定向路由方案优化内存使用和通信性能|
|verl|[0006-verl-feature-weight_converter_alltoall_overlap.patch](patches/verl/0006-verl-feature-weight_converter_alltoall_overlap.patch)|完善Mcore到HF模型参数名转换逻辑|
|verl|[0007-verl-feature-onload_offload.patch](patches/verl/0007-verl-feature-onload_offload.patch)|1.添加TorchAir图模式相关配置 2.由于NPU上vLLM的sleep模式可能存在内存卸载不干净的问题，改为手动实现NPU上Rollout模型及KV Cache的卸载和模型加载|
|verl|[0008-verl-bugfix-enable_compile.patch](patches/verl/0008-verl-bugfix-enable_compile.patch)|NPU上MindSpeed训练框架会无效化torch.compile规避训练侧的compile失败，在推理时开启compile|
|verl|[0009-verl-feature-support_EPLB.patch](patches/verl/0009-verl-feature-support_EPLB.patch)|`VLLM_ENABLE_EPLB`开启时，使能推理的EPLB|
|verl|[0010-verl-feature-enable_hdp.patch](patches/verl/0010-verl-feature-enable_hdp.patch)|`USE_HDP`开启时，使能HDP功能|
|vllm|[0001-vllm-feature-disable_gc.patch](patches/vllm/0001-vllm-feature-disable_gc.patch)|在decode step前关闭gc，避免因内存管理导致host bound影响推理性能|
|vllm|[0002-vllm-feature-kv_cache_configs.patch](patches/vllm/0002-vllm-feature-kv_cache_configs.patch)|【手动卸载KV Cache】实现KV Cache可获取，通过初始化卸载KV Cache确保每次初始化始终调用初次申请的config，保证内存一致性|
|vllm_ascend|[0001-vllm_ascend-feature-initialize_kv_cache.patch](patches/vllm_ascend/0001-vllm_ascend-feature-initialize_kv_cache.patch)|【手动卸载KV Cache】避免在KV Cache初始化时多次调用AttentionBackend初始化|
|vllm_ascend|[0002-vllm_ascend-feature-chunk_moe.patch](patches/vllm_ascend/0002-vllm_ascend-feature-chunk_moe.patch)|针对MoE计算场景分块处理优化，解决prefill阶段可能引起的峰值内存过高|
|vllm_ascend|[0003-vllm_ascend-feature-enable_zero_tp_to_ep.patch](patches/vllm_ascend/0003-vllm_ascend-feature-enable_zero_tp_to_ep.patch)|零冗余TP转EP通信方案，将o_proj的AllReduce算子替换为ReduceScatter算子，减少冗余通信|
|vllm_ascend|[0004-vllm_ascend-feature-dummy_run_load_balance.patch](patches/vllm_ascend/0004-vllm_ascend-feature-dummy_run_load_balance.patch)| 在dummy_run阶段强制负载均衡，优化内存分配|
|vllm_ascend|[0005-vllm_ascend-feature-support_EPLB.patch](patches/0005-vllm_ascend-feature-support_EPLB.patch) | `VLLM_ENABLE_EPLB`开启时，使能推理的EPLB|


## 基于Dockerfile构建环境
> 环境搭建可以基于Dockerfile快速实现，我们已经在Dockerfile里配置了必要的昇腾软件和其他第三方软件的依赖。如果遇到网络不通等问题，也可以参考附录中的[手动准备环境](#手动准备环境)章节。

1. 基于Dockerfile创建镜像。
   ```bash
   # 请预先下载本样例提供的Dockerfile文件：Dockerfile.vllm_ascend.mindspeed.qwen3
   # 随后基于该Dockerfile创建docker image。请传入镜像名称，例如 your_image_name
   docker build -t your_image_name -f Dockerfile.vllm_ascend.mindspeed.qwen3 .

   # 请设置容器名称，例如 your_docker_name，镜像名称同上一步
   container_name=your_docker_name
   image_name=your_image_name:latest

   # 执行docker run命令创建容器，可通过-v按需挂载宿主机目录至容器
   docker run -itd \
   --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
   -v /usr/local/sbin/:/usr/local/sbin/ \
   -v /data/:/data/ \
   -v /home/:/home/ \
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /dev/shm:/dev/shm \
   --net=host \
   --name ${container_name} \
   --privileged ${image_name} /bin/bash

   # 执行docker exec命令进入容器
   docker exec -it -u root ${container_name} bash
   ```
   也可通过当前目录**run_container.sh**构建镜像：
   ```bash
   bash run_container.sh your_docker_name your_image_name:version
   ```

2. 安装所需的python依赖：
   ```bash
   # 安装依赖的python库
   pip install -r requirements.txt
   ```

3. 源码准备并使能patch修改：
   可通过 **build_project.sh** 一键执行，在当前目录下（cann-recipes-train/rl_train/qwen3）运行：
   ```bash
   bash build_project.sh
   ```

## 数据集准备
本样例中GRPO使用的deepscaler数据集准备方法与DeepSeek示例相同，可参考[数据集准备](../deepseek/README.md#数据集准备)，将处理后的训练数据放在 `data/deepscaler` 目录下。

DAPO使用的dapo-math-17k数据集，验证集使用AIME，可参考[DAPO数据准备](https://github.com/volcengine/verl/blob/main/recipe/dapo/prepare_dapo_data.sh)，将训练数据放在 `data/dapo_math` 目录下。

gsm8k等其他数据集准备方法可参考[verl官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)。

## 模型权重准备
本样例使用的Qwen3-235B-A22B模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放至样例的./Qwen3-235B-A22B-hf目录下（不加载权重实验也需要执行这步操作）
mkdir ./Qwen3-235B-A22B-hf
pip install modelscope
modelscope download --model Qwen/Qwen3-235B-A22B config.json tokenizer.json tokenizer_config.json generation_config.json vocab.json --local_dir ./Qwen3-235B-A22B-hf

# 下载Qwen3-235B-A22B完整权重至指定目录，例如 your_hf_weights（此步骤需要目录所在磁盘有440GB以上空间）
modelscope download --model Qwen/Qwen3-235B-A22B --local_dir your_hf_weights

# source环境变量，根据实际CANN安装目录调整
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 配置环境变量 使能all_to_all_overlap
export USE_ALLTOALL_OVERLAP=1

# 执行权重切分，切分后权重保存至指定目录，例如 your_sharded_weights
torchrun --nproc_per_node 16 --nnodes ${NNODES} --node_rank ${NODE_RANK} converter_hf_to_mcore.py \
--hf_model_path ${hf_model} --output_path ${output_path} --trust_remote_code --use_cpu_initialization
```

## RL后训练执行

在本样例代码根目录下启动Qwen3-235B-A22B的RL后训练。

```bash
# 请注意，以下bash启动脚本中的内容需要手动配置
# source脚本路径：  根据实际CANN安装目录调整
# MASTER_ADDR：    ray集群主节点的IP地址，每个节点的脚本配置一致
# SOCKET_IFNAME：  集群中各节点自己的网卡名，可通过ifconfig命令查看

bash train_qwen3_235b_128die.sh
```

可对 `train_qwen3_235b_128die.sh` 104行进行修改，实现随机权重训练GRPO算法、真实权重训练GRPO算法、真实权重训练DAPO算法，对应修改如下：
| 训练 | 104行修改|
|------|----------|
| 随机权重训练 GRPO算法 | `bash ./internal/train_grpo_qwen3_235b_128die_random_init.sh` |
| 真实权重训练 GRPO算法 | `bash ./internal/train_grpo_qwen3_235b_128die_true_weight.sh` |
| 真实权重训练 DAPO算法 | `bash ./internal/train_dapo_qwen3_235b_128die_true_weight.sh` |

## 附录
### 手动准备环境

1. 创建CANN 8.3.RC1镜像。

   ```bash
   # 镜像下载
   docker pull quay.io/ascend/cann:8.3.rc1-a3-openeuler24.03-py3.11

   # 执行以下脚本创建容器，请传入容器名称，如 your_docker_name
   docker run -itd \
   --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
   -v /usr/local/sbin/:/usr/local/sbin/ \
   -v /data/:/data/ \
   -v /home/:/home/ \
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /dev/shm:/dev/shm \
   --net=host \
   --name your_docker_name \
   --privileged quay.io/ascend/cann:8.3.rc1-a3-openeuler24.03-py3.11 /bin/bash

   # 执行docker exec命令进入容器
   docker exec -it -u root your_docker_name bash
   ```

2. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-train.git

    # 安装依赖的python库
    cd cann-recipes-train/rl_train/qwen3
    pip install -r requirements.txt
    ```

3. 下载依赖的开源框架代码。

   为了让使用者和开发者直观了解我们基于开源代码做的修改，本样例中只包含patch代码，其他框架代码需要拉取。

   在当前目录（cann-recipes-train/rl_train/qwen3）执行如下命令，需注意，请确保环境能够正常连通网络。
   ```bash
   set -ex

   mkdir -p asset && cd asset

   # 下载verl源码
   git clone https://github.com/volcengine/verl.git
   cd verl
   git checkout v0.6.0
   git fetch origin pull/3427/head && \
   git cherry-pick -n -X theirs 448c6c3 && \
   git fetch origin pull/4030/head && \
   git cherry-pick -n f2d57afe && \
   git cherry-pick -n 566ca6ce
   cp -r verl ../../
   cd -

   # 下载Megatron-LM源码
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.12.1
   mkdir -p ../../megatron
   cp -r megatron/core/ ../../megatron
   cd -

   # 下载MindSpeed源码
   git clone https://gitcode.com/Ascend/MindSpeed.git
   cd MindSpeed
   git checkout f6688c61bcfe45243ee5eb34c6f013b1e06eca81
   cp -r mindspeed ../../
   cd -

   # 下载vLLM源码
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.11.0
   cp -r vllm ../../
   cd -

   # 下载vLLM-Ascend源码
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   git checkout v0.11.0rc0
   cp -r vllm_ascend ../../
   cd -

   # 回到项目目录
   cd ../
   ```

4. 源码编译安装vLLM和vLLM-Ascend。
   
   vLLM:
   ```bash
   VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
   python3 -m pip uninstall -y triton && \
   python3 -m pip cache purge
   ```

   vLLM-Ascend:
   ```bash
   export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi && \
   source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
   source /usr/local/Ascend/nnal/atb/set_env.sh && \
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
   export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/c++/12:/usr/include/c++/12/`uname -i`-openEuler-linux && \
   python3 -m pip install -v -e ./asset/vllm-ascend/ --exists-action=i --extra-index https://download.pytorch.org/whl/cpu/
   ```

5. 对源码添加patch代码。

   为了使能patch修改，需要依次应用**patches**目录下的patch文件。
   ```bash
   git apply -p3 --ignore-whitespace patches/xxx.patch
   ```
