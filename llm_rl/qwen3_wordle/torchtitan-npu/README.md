# Qwen3 Wordle TorchTitan-NPU 训练后端

本目录将 Wordle GRPO 的 Actor/Reference Model 训练引擎切换为 TorchTitan-NPU FSDP2，保留原有模型、数据、奖励函数及 vLLM rollout。原有 FSDP 启动入口保持独立，两个后端的训练负载与算法参数对齐。

## 环境与安装

- 单机 2 张可见 Ascend NPU，CANN 9.0 及配套 NNAL（ATB）加速库。
- Python 3.11、GCC 11 或更高版本、uv 0.12.0。
- 框架版本和源码提交见 [versions.env](versions.env)。

以下命令均在 `llm_rl/qwen3_wordle` 目录执行：

```bash
python3 -m pip install --user "uv==0.12.0"
export PATH="$(python3 -m site --user-base)/bin:${PATH}"
bash torchtitan-npu/setup_backend.sh
```

安装脚本创建独立的 `.venv/` 环境，将框架源码下载到 `runtime_sources/`，应用 NPU 兼容补丁并检查环境，不修改原有 FSDP 环境。为避免 CANN 构建时遗漏对象文件，请勿将项目放在含隐藏目录的路径下。

安装时应用以下补丁：

| 补丁 | 用途 |
| --- | --- |
| `0001-verl-feature-torchtitan_npu.patch` | 接入 NPU、卸载与权重同步，传递实际训练步数 |
| `0001-torchtitan-bugfix-qwen3_npu_init.patch` | 适配 Qwen3 的 NPU 初始化 |
| `0002-torchtitan-bugfix-qwen3_1_7b_seq_length.patch` | 调整 Qwen3-1.7B 序列长度配置 |
| `0001-vllm-bugfix-ascend_dependencies.patch` | 对齐 vLLM-Ascend 依赖 |
| `0001-vllm_ascend-bugfix-torch_2_12_build.patch` | 适配 PyTorch 2.12 构建 |
| `0002-vllm_ascend-bugfix-build_python_torch_npu.patch` | 使用构建环境定位 torch-npu |
| `0003-vllm_ascend-bugfix-cann_moe_headers.patch` | 复制新版 CANN 拆分出的通信头文件 |
| `0001-torchair-bugfix-hint_int_import.patch` | 适配 TorchAir 的 hint_int 导入 |

## 模型与数据

已有 Wordle SFT 模型权重和数据可直接复用。默认使用以下路径：

- 模型：`models/Qwen3-1.7B-Wordle-SFT`
- 训练集：`data/wordle_train.parquet`
- 验证集：`data/wordle_test.parquet`

尚未准备时，在安装完成后执行：

```bash
torchtitan-npu/.venv/bin/python3 prepare_data.py \
    --seed 42 --num_train 2000 --num_test 20 --output_dir data
TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchtitan-npu/.venv/bin/modelscope download \
    --model misumisumisu/Qwen3-1.7B-Wordle-SFT \
    --local_dir models/Qwen3-1.7B-Wordle-SFT
```

## 启动训练

```bash
bash torchtitan-npu/run_qwen3_1.7b_wordle_torchtitan_npu.sh
```

默认采用两卡 FSDP2 分片、Actor 参数及优化器 offload、Ref 参数 offload、激活重计算与 TND 变长注意力。训练配置为 batch 128、每题采样 8 条、最大响应长度 4096、单条样本最大长度 5120、熵系数 0.004，共训练 5 个 epoch。

模型与数据在其他位置时，同时指定三个路径：

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-Wordle-SFT \
TRAIN_FILE=/path/to/wordle_train.parquet \
TEST_FILE=/path/to/wordle_test.parquet \
bash torchtitan-npu/run_qwen3_1.7b_wordle_torchtitan_npu.sh
```

可选：保持上述训练负载，仅运行 3 步，关闭训练期间的定期验证与 checkpoint 保存（仍执行训练前验证）：

```bash
TOTAL_TRAINING_STEPS=3 SAVE_FREQ=-1 TEST_FREQ=-1 \
bash torchtitan-npu/run_qwen3_1.7b_wordle_torchtitan_npu.sh
```

正式训练默认每 5 步验证、每 25 步保存 checkpoint，保存目录为 `torchtitan-npu/checkpoint/<experiment_name>/`。长期训练前请预留模型及优化器状态的存储空间，并按需自行清理历史 checkpoint。
