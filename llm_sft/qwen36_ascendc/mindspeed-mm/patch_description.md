`mindspeed-mm.patch` 是相对于基线
`aaed711fd4750857f104e8d832766ed915ff9ef0` 生成的补丁，并可在该 commit 的干净工作树上直接应用。补丁共涉及 31 个文件：13 个新增、18 个修改。

### 1. FSDP2 按 epoch 训练

涉及：

- `mindspeed_mm/fsdp/train/epoch_schedule.py`
- `mindspeed_mm/fsdp/train/trainer.py`
- `mindspeed_mm/fsdp/train/train_engine.py`
- `mindspeed_mm/fsdp/params/training_args.py`
- `mindspeed_mm/fsdp/tasks/funasr/trainer.py`

具体包括：

- 新增 epoch schedule 辅助逻辑；标准 FSDP2 `Trainer` 在构建 dataloader 后，将 `train_epochs` 换算为 optimizer iteration，并覆盖 `train_iters`。`train_epochs` 未设置时，仍按 `train_iters` 运行。
- 标准 FSDP2 `Trainer` 在 epoch 模式下将 `save_interval` 解读为 epoch 数，换算为明确的 optimizer 保存步；未设置 `train_epochs` 时仍保持按 optimizer step 的原有语义。
- epoch 边界按 `ceil(epoch * micro_batches_per_epoch / gradient_accumulation_steps)` 取整，保存点落在跨过边界的第一个 optimizer step，因此不会因数据量或梯度累积不能整除而向下截断。
- 同时兼容普通 DataLoader、`PrefetchGradAccDataLoader` 包装器和项目的 `BaseRandomBatchSampler`；后者按有效样本数（排除不完整尾 batch）及全局 micro-batch 大小计算每个 epoch 的 micro-batch 数。
- 校验空数据集、非法 epoch、非法 `save_interval`，以及每个 epoch 的 micro-batch 数小于梯度累积步数等情况。
- 显式拒绝配置字段拼写错误 `train_epoches`，要求使用 `train_epochs`。
- FunASR 的 split-based dataloader 拒绝通用的 `training.train_epochs`，继续使用其专用的 `max_epochs` 与 `train_iters`；其训练循环仍按 iteration 处理保存。
- 通用 `TrainEngine` 在训练结束时避免重复保存刚刚在同一 iteration 保存过的 checkpoint。
- 迁移文档补充说明：按 epoch 续训时应保持数据集长度、梯度累积步数和并行配置不变，否则 checkpoint 的 iteration 与 dataloader epoch 位置无法对应。

### 2. LoRA 初始化和保存逻辑

涉及：

- `mindspeed_mm/fsdp/utils/lora_utils.py`
- `mindspeed_mm/fsdp/params/lora_args.py`
- `mindspeed_mm/fsdp/train/trainer.py`
- `mindspeed_mm/fsdp/train/train_engine.py`
- `docs/zh/features/lora_finetune_fsdp2.md`

主要变化：

- 新增 `is_pure_lora_training()`；要求存在可训练参数且所有可训练参数名都属于 LoRA adapter，才判定为纯 LoRA 训练。
- 新增 `initialize_lora_weights_after_materialization()`，解决 meta device 初始化后通过 `to_empty()` 导致 LoRA 权重未真正初始化的问题。
- 在启用 meta device 的 LoRA 流程中，先 materialize 参数，再初始化基座权重，最后按 PEFT 语义重新初始化 LoRA adapter；即使后续加载基座 checkpoint，也不会留下未初始化的 adapter 存储。
- 支持重新初始化 `True`、`False` 和 `"gaussian"` 三种方式；其他字符串初始化方法在该 meta 流程中会被拒绝。
- 新增 `training.lora.save_full_model`，默认值为 `false`。
- 纯 LoRA 训练默认只保存：
  `lora_adapter_iteration_xxx.safetensors`
- 如果存在解冻的非 LoRA 参数，例如 `lm_head`，仍然保存完整模型 checkpoint。
- 设置 `save_full_model: true` 时，在 adapter 文件之外额外保存包含完整模型状态的 DCP；默认的 adapter-only 行为不会改变。
- 文档将 `pretrained_lora_path` 示例更新为具体的 `lora_adapter_iteration_xxx.safetensors` 文件，并补充 adapter-only checkpoint 和严格断点续训的区别。

### 3. Checkpoint 保存 dtype

涉及：

- `mindspeed_mm/fsdp/checkpoint/dcp_checkpointer.py`
- `mindspeed_mm/fsdp/params/training_args.py`
- `mindspeed_mm/fsdp/train/train_engine.py`
- `mindspeed_mm/fsdp/tasks/funasr/train_engine.py`

新增：

- `training.save_dtype`，支持 `fp16`、`bf16`、`fp32`；未设置时保持原有保存 dtype。
- 保存模型 state dict 时递归转换浮点 tensor 的 dtype。
- 只影响完整模型 checkpoint 保存，不改变训练过程中的参数 dtype，也不改变 adapter-only 文件的保存逻辑。
- FunASR 保存逻辑同步传递该参数。

### 4. 数据处理和 SFT packing

涉及：

- `mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py`
- `mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py`
- `cvt_data_format_jsonl2json.sh`

新增：

- `shuffle_before_packing`，默认关闭。
- `shuffle_before_packing_seed`，默认 `42`。
- 在 Qwen2VL/Hugging Face 数据集路径中，仅当 `stage=sft`、启用 packing、离线预处理且数据集非 streaming 时生效。
- shuffle 顺序为：数据对齐后、tokenize/packing 前。
- 启用该选项但前置条件冲突（streaming 数据集或 `preprocess_on_fly=true`）时显式报错，而不是执行近似 shuffle；未启用时不改变原有行为。
- 新增 JSONL 转 JSON 工具：
  - 支持 `instruction/input/output`
  - 支持 `prompt/response`
  - 支持单文件和 glob 批量转换
  - 批量模式将静态输入根目录下的层级映射到输出根目录
  - 对每行 JSON 语法、对象类型和必需字段进行校验，生成统一的 user/assistant `messages` 结构并附带空的 `images` 列表；`instruction` 非空时与 `input` 用换行拼接，`prompt/response` 则直接映射为 user/assistant 内容

### 5. Qwen3.5 checkpoint 转换

涉及：

- `checkpoint/vlm_model/converters/qwen3_5.py`
- `cvt_ckpt_dcp2hf.sh`
- `cvt_ckpt_dcp2hf_batch.sh`
- `cvt_ckpt_hf2dcp.sh`

主要变化：

- Qwen3.5 DCP 转 HF 后，自动从原始 HF 目录复制存在的 tokenizer 相关文件，并在复制成功后更新输出目录及文件权限，例如：
  `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`vocab.json` 等。
- 新增 DCP 到 HF、HF 到 DCP 的快捷脚本；HF→DCP 脚本固定使用 `ckpt/hf_path/Qwen3.6-27B` 与 `ckpt/dcp_path/Qwen3.6-27B`，DCP→HF 脚本固定原始 HF 模型路径，输入/输出目录通过参数传入。
- 新增批量处理脚本：
  - 批量转换数字命名的 `iter_*` checkpoint。
  - 批量合并数字 iteration 的 LoRA adapter。
  - 通过 `LORA_MERGE_DEVICE=cpu|npu` 选择 CPU/NPU 合并。
  - 通过 `CHECKPOINT_POSTPROCESS_DRY_RUN=1` 预览处理计划而不执行转换。
  - 发现 LoRA adapter 文件时，读取 YAML 中的基座模型路径、LoRA alpha 和 rank，并校验配置。
  - 处理 DCP `iter_*` 目录期间保护并在退出时恢复 `latest_checkpointed_iteration.txt`。

### 6. 训练启动和 NPU 调度脚本

新增：

- `run_qwen36_sft.sh`
- `run_train_nohup.sh`
- `scripts/wait_for_npu_idle.sh`

功能包括：

- `run_train_nohup.sh` 接收配置文件和可选 NPU 数量；`run_qwen36_sft.sh` 提供对应的 16 卡/8 卡调用示例。
- 校验 NPU 数量和 `ASCEND_RT_VISIBLE_DEVICES`。
- 从 YAML 中读取 `training.save`。
- 后台启动训练。
- 记录 PID、workflow 日志。
- 训练成功后尝试执行 checkpoint 转换和 LoRA 合并；训练失败时不启动后处理。
- 通过 `npu-smi` 轮询 NPU 显存，等设备空闲后再启动训练。
- 支持指定设备、显存阈值、查询超时和轮询间隔，分别由 `NPU_DEVICE_IDS`、`IDLE_MEMORY_THRESHOLD_MB`、`QUERY_TIMEOUT_SECONDS`、`CHECK_INTERVAL_SECONDS` 控制。

同时修改 `examples/qwen3_6/finetune_qwen3_6_27B.sh`：

- `NPUS_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT` 可通过环境变量配置，并校验 NPU 数量和端口范围。
- 未设置 `MASTER_PORT` 时使用 `torchrun --standalone` 自动选择 rendezvous；设置后使用静态 rendezvous。
- 从命令行接收 YAML 配置路径，不再写死配置文件；日志文件名包含配置名。
- 使用 `pipefail` 检测训练失败，失败时返回非零状态；保留训练耗时和吞吐统计。
- 默认不再自动 source 固定路径的 CANN 环境脚本，需按实际环境手动配置。
- 补充 setuptools `<82.0.0` 限制提示及不同 CANN 版本对应的 `triton-ascend` 版本提示（CANN 8.5.0 使用 3.2.0，CANN 9.0.0 使用 3.2.1）。

`run_qwen36_sft.sh` 本身是启动示例注释，提供 16 卡默认启动和通过 `ASCEND_RT_VISIBLE_DEVICES` 执行 8 卡训练的命令。

### 7. Qwen3.6 示例文档

修改 `examples/qwen3_6/README.md`：

- 更新示例仓库 clone 地址，并补充 CANN 9.0.0 对应的 `triton-ascend==3.2.1` 安装命令。
- 补充 `train_iters`/`train_epochs`、`save_interval` 的配置示例及 `train_epoches` 拼写提示。

### 8. 文档、忽略规则和测试

其他改动：

- `.gitignore` 新增 `.claude/`、`.opencode/`、`cache_dir`、`ckpt`、`outputs`、`data`、`logs`、`tmp`、`MindSpeed` 等目录。
- 更新 FSDP2 开发者迁移文档和 LoRA 微调文档。
- 新增/修改测试，覆盖：
  - `train_epoches` 拼写校验、epoch schedule、DataLoader/`BaseRandomBatchSampler` 归一化和 checkpoint 边界
  - LoRA 纯训练判断、meta device 下 LoRA 初始化，以及 adapter-only/完整/混合训练保存分支
  - `instruction/input/output` 与 `prompt/response` 两种 JSONL 转换格式及缺失字段错误
  - Qwen2VL SFT packing 前 shuffle 的开关、前置条件和执行顺序

本 patch 没有新增 `save_dtype`、checkpoint 转换脚本、训练启动脚本或 NPU 空闲轮询脚本的端到端测试；FunASR 相关改动也没有对应的新增专项测试。
