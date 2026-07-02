# 基于 TorchTitan-NPU 的 Qwen3-30B-A3B SFT 训练实践样例

## 概述

本样例基于 [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) 框架，对 `Qwen3-30B-A3B` 模型进行医学领域监督微调（SFT），并通过医学问答 Keyword Recall 指标验证训练收益。

样例使用 [Medical R1](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data) 医学问答数据集，训练阶段采用 MoE 并行配置完成单机多卡全参微调，评测阶段使用 vLLM + vLLM-Ascend，对原始模型和 SFT 模型进行同口径对比。

本文档重点覆盖环境准备、源码准备、数据与权重准备、训练配置、训练启动、实验结果。

## 使用的产品型号

| 项目 | 规格 |
| --- | --- |
| 产品型号 | Atlas A3 系列 |
| 推荐卡数 | 16 卡（CP=2, EP=8, TP=2） |
| CANN 版本 | 9.0.0 |
| Python | 3.11 |
| 训练框架 | torchtitan-npu |
| 推理框架 | vLLM + vLLM-Ascend |

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `README.md` | 中文训练与评测说明（本文档） |
| `README_EN.md` | 英文说明 |
| `config_registry_medical.py` | torchtitan-npu Qwen3-30B-A3B 医学 SFT 配置 |
| `run_medical_sft.sh` | 训练启动脚本（复制到 torchtitan-npu 目录后执行） |
| `prepare_medical_r1_dataset.py` | Medical R1 数据集切分工具 |
| `figures/training_loss.png` | 训练 Loss 下降曲线 |

## 环境准备

### 1. 启动容器

可直接使用昇腾官方公开镜像。镜像中已包含本样例所需的 CANN 软件栈与 Python 环境。以下为单机 16 卡容器启动示例：

```shell
# 获取Docker镜像
docker pull quay.io/ascend/cann:9.0.0-beta.1-a3-ubuntu22.04-py3.11

# 构建Docker容器
docker run -itd \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${HOST_WORKSPACE}:${HOST_WORKSPACE} \
  -w ${HOST_WORKSPACE} \
  --net=host \
  --shm-size=128g \
  --privileged \
  --name qwen3_30b_medical_sft \
  quay.io/ascend/cann:9.0.0-beta.1-a3-ubuntu22.04-py3.11 \
  /bin/bash

  # 进入容器
  docker exec -it qwen3_30b_medical_sft bash
```

进入容器后初始化 CANN 环境。不同部署方式下 CANN 路径可能不同，请根据实际环境调整：

```shell
# Docker 镜像默认路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 若为 conda 安装，路径可能为（以 CANN 9.0.0 为例）
source /home/developer/Ascend/cann-9.0.0/set_env.sh
source /home/developer/Ascend/nnal/atb/set_env.sh
```

### 2. 安装 torchtitan-npu

```shell
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
pip install -r requirements.txt
pip install -e .
```

## 数据集准备

从 ModelScope 手动下载数据文件，放到 torchtitan-npu 的 assets 目录下：

```shell
cd /path/to/torchtitan-npu
mkdir -p assets
# 手动从 https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data/files
# 下载 r1_data_example.jsonl 到 assets/ 目录
ls assets/r1_data_example.jsonl
```

然后使用本 recipe 提供的切分脚本生成训练集和测试集：

```shell
python /path/to/recipe/prepare_medical_r1_dataset.py \
  --input ./assets/r1_data_example.jsonl \
  --output ./assets/medical_r1
```

切分结果：

| 数据集 | 样本数 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 约 2,166 | SFT 训练 |
| `test.jsonl` | 约 241 | Keyword Recall 评测 |

## 模型权重准备

从 ModelScope 下载 `Qwen3-30B-A3B` 权重（约 60 GB），并在 torchtitan-npu 源码目录下创建软链接：

```shell
pip install modelscope
mkdir -p /data/models/Qwen3-30B-A3B

modelscope download \
  --model Qwen/Qwen3-30B-A3B \
  --local_dir /data/models/Qwen3-30B-A3B

cd /path/to/torchtitan-npu
mkdir -p assets/hf
ln -sf /data/models/Qwen3-30B-A3B assets/hf/Qwen3-30B-A3B
```

## 训练配置

### 注册训练配置

将本 recipe 目录下的 `config_registry_medical.py` 复制到 torchtitan-npu 源码中：

```shell
cp /path/to/recipe/config_registry_medical.py \
  /path/to/torchtitan-npu/torchtitan_npu/models/qwen3/config_registry_medical.py
```

然后在 `torchtitan_npu/models/qwen3/config_registry.py` 末尾添加：

```python
from torchtitan_npu.models.qwen3.config_registry_medical import (
    sft_qwen3_30ba3b_medical,
    sft_qwen3_30ba3b_medical_tnd,
)
```

### 并行策略

单机 16 卡 MoE 并行（CP=2, EP=8, TP=2）：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `NGPU` | 16 | 总卡数 |
| `context_parallel_degree` | 2 | 上下文并行 |
| `tensor_parallel_degree` | 2 | 张量并行 |
| `expert_parallel_degree` | 8 | 128 个专家按 EP 维切分 |
| `pipeline_parallel_degree` | 1 | 不启用 PP |
| `data_parallel_shard_degree` | -1 | FSDP 分片自动推导 |

### 超参数

| 配置项 | 推荐值 | 说明 |
| --- | --- | --- |
| `steps` | 156 | 训练步数（5 epoch，约 31 步/epoch） |
| `lr` | 2e-5 | 学习率 |
| `warmup_steps` | 5 | 预热步数 |
| `local_batch_size` | 1 | 单卡 batch size |
| `seq_len` | 4096 | 训练序列长度 |
| `activation_checkpoint` | selective | 选择性重计算 |
| `TRAIN_DATA` | 切分后的训练集 | 训练数据路径，在脚本中通过 `TRAIN_DATA` 环境变量指定 |
| `MODEL_DIR` | `assets/hf/Qwen3-30B-A3B` | HF 权重路径 |

### Attention 选择

本样例默认使用 **TND（NPUVarlenAttention，CANN FA v3）** 进行训练。BSND（SDPA）配置仅作为代码参考保留，因上游未对 ChatDataLoader + BSND 做完整适配，实际操作中仅 TND 可用。

| 配置函数 | Attention 类型 | 说明 |
| --- | --- | --- |
| `sft_qwen3_30ba3b_medical` | BSND（SDPA） | 仅作为参考 |
| `sft_qwen3_30ba3b_medical_tnd` | TND（NPUVarlenAttention） | **推荐，验证可用** |

## 启动训练

将本 recipe 的启动脚本复制到 torchtitan-npu 目录后执行：

```shell
cp /path/to/recipe/run_medical_sft.sh /path/to/torchtitan-npu/
cd /path/to/torchtitan-npu
bash run_medical_sft.sh
```

脚本会使用环境变量 `NGPU=16` 和 `CONFIG=sft_qwen3_30ba3b_medical_tnd` 启动训练（TND 版本）。日志输出示例（EP=8 实测数据）：

```text
step:    1  loss:  1.45426  memory:  37.73GiB(61.58%)  tps:    59   69.018s  (编译)
step:    2  loss:  1.39178  memory:  52.27GiB(85.31%)  tps:   798    5.135s
step:    3  loss:  1.26931  memory:  52.31GiB(85.37%)  tps:  1215    3.370s
step:   10  loss:  1.02183  memory:  52.44GiB(85.59%)  tps:   993    4.126s
step:   20  loss:  0.95751  memory:  52.44GiB(85.59%)  tps:  1199    3.416s
step:   31  loss:  0.70617  memory:  52.44GiB(85.59%)  tps:  1345    3.046s   ← epoch 1 结束
step:   32  loss:  0.67716  memory:  52.44GiB(85.59%)  tps:   701    5.842s
step:   50  loss:  0.58786  memory:  52.50GiB(85.69%)  tps:  1010    4.056s
step:   62  loss:  0.34057  memory:  52.56GiB(85.79%)  tps:  1177    3.479s   ← epoch 2 结束
step:   63  loss:  0.33076  memory:  52.56GiB(85.79%)  tps:   803    5.102s
step:   90  loss:  0.19230  memory:  52.56GiB(85.79%)  tps:   733    5.590s
step:   93  loss:  0.16940  memory:  52.56GiB(85.79%)  tps:  1014    4.040s   ← epoch 3 结束
step:   94  loss:  0.16507  memory:  52.56GiB(85.79%)  tps:  1286    3.185s
step:  120  loss:  0.08754  memory:  52.62GiB(85.88%)  tps:   942    4.349s
step:  124  loss:  0.08219  memory:  52.62GiB(85.88%)  tps:  1257    3.260s   ← epoch 4 结束
step:  125  loss:  0.08480  memory:  52.62GiB(85.88%)  tps:  1274    3.215s
step:  150  loss:  0.04411  memory:  52.62GiB(85.88%)  tps:   918    4.462s
step:  155  loss:  0.04376  memory:  52.62GiB(85.88%)  tps:  1199    3.416s
step:  156  loss:  0.04450  memory:  52.62GiB(85.88%)  tps:  1244    3.292s   ← 结束（epoch 5）
```

## 模型导出

训练配置中启用 `last_save_in_hf=True` 后，最终 checkpoint 会导出为 HuggingFace 格式。将原始模型配置与 SFT 权重整理到同一目录即可用于推理：

```shell
mkdir -p /data/models/Qwen3-30B-A3B-SFT
cp /data/models/Qwen3-30B-A3B/*.json   /data/models/Qwen3-30B-A3B-SFT/
cp /data/models/Qwen3-30B-A3B/tokenizer* /data/models/Qwen3-30B-A3B-SFT/
cp checkpoint_medical/step-156/*.safetensors* /data/models/Qwen3-30B-A3B-SFT/
```

## 实验结果

### 评测说明

本实验使用基于 jieba 分词 + POS 词性过滤的关键词提取方法，从参考答案和模型回答中提取关键词，计算以下指标：

- **Recall（召回率）** = 命中的参考答案关键词数 / 参考答案关键词总数，衡量参考答案的关键信息被模型覆盖了多少
- **Precision（精确率）** = 命中的参考答案关键词数 / 模型输出的关键词总数，衡量模型输出的关键词中有多少是有效的
- **F1** = Recall 和 Precision 的调和平均

### Keyword Recall 对比

以下数据来自同口径评测（241 条医学问答，jieba + POS 关键词提取），展示预训练 Base 模型与医学 CPT 中间 checkpoint（step 156，效果最优）的对比。

| 模型 | Recall | Precision | F1 |
| :--- | ---: | ---: | ---: |
| Base（Qwen3-30B-A3B） | 53.83% | 25.16% | 33.30% |
| **SFT（epoch 5）** | **62.45%** | **28.06%** | **37.82%** |
| **提升** | **+8.62pp** | **+2.90pp** | **+4.52pp** |

SFT 后在三个指标上均有提升，模型不仅记住了更多关键信息，而且输出更准确精炼。

### 输出格式对比

| 指标 | Base | SFT |
| --- | ---: | ---: |
| 平均输出长度 | 1,061 字符 | **831 字符**（-21.7%） |
| 格式错乱样本数（`</think>` 重复） | 199/241 | **9/241** |

典型样本："意识由哪两部分组成"

| 项目 | Base 模型 | SFT |
| --- | --- | --- |
| **回答** | `</think> 意识的组成...觉醒状态...意识内容...`（Markdown 列表 + 3 个冗余 `</think>`） | `意识由两部分组成：意识内容和意识的开头系统...`（对话式段落） |
| **Recall** | 52.4% | **95.2%** |
| **长度** | 392 字符 | 287 字符 |

### 训练过程指标

| 指标 | 实测值 |
| --- | --- |
| 稳定单步耗时 | 约 3.2-3.5s |
| 稳定显存占用 | 约 52.6 GiB/卡（85.9%） |
| Loss 起点（step 1） | 1.45 |
| Loss 终点（step 156） | 0.045 |
| 总耗时（156 步） | 约 8-9 分钟 |

## 常见问题

### 1. 训练开始 loss 异常高

若 loss 起点明显高于预期（如 12 左右），检查是否正确加载 HF 预训练权重。重新训练前删除 checkpoint 目录：

```shell
rm -rf checkpoint_medical
```

### 2. NPU 显存不足

检查是否有残留进程占用 NPU 显存，确保设置了 `PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"`。必要时在 entry.py 入口处添加 `torch.npu.set_per_process_memory_fraction(1.0)`。

### 3. HCCL 通信超时

多卡训练可能出现 HCCL watchdog timeout。若偶发出现，通常重启训练可恢复；若频繁出现，需检查 HCCL 网络配置和节点间通信。
