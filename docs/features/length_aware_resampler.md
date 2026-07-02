# Length-Aware Resampler：基于历史 Response 长度的 Rollout 重采样优化

## 1. 简介
RL On-Policy 训练的 Rollout 阶段通常会遇到明显的 response 长尾问题：不同 prompt 的生成长度差异很大，而单个 generation batch 内往往会同时混入极长和极短样本。这样会带来两个直接问题：

1. 同一批次中，短样本很快结束，长样本持续占用推理资源，导致整体吞吐被少量长尾样本拖慢。
2. 某些 batch 的生成长度远高于平均值，容易放大 step 时间抖动，影响训练吞吐稳定性。

为缓解这一问题，本优化在 `verl` 的 RL 数据采样路径中新增了 `LengthAwareEpochSampler`。该 sampler 会利用上一轮 rollout 收集到的 response 长度统计，在下一轮 epoch 开始时对样本顺序进行重排，把预估长度接近的 prompt 尽量放到邻近 batch 中。

## 2. 使用说明

### 2.1 Patch 位置
本特性以 Git Patch 的形式交付，文件位于：

```text
llm_rl/qwen3/verl-mindspeed/patches/verl/0019-verl-feature-length_aware_resampler.patch
```

在 `llm_rl/qwen3/verl-mindspeed/` 目录下执行以下命令即可应用全部 patch：

```bash
bash apply_all_patches.sh
```

### 2.2 Hydra 配置
应用 patch 后，可通过以下配置启用 Length-Aware Resampler：

```bash
data.sampler.class_path=pkg://verl.experimental.dataset.length_bucket_sampler
data.sampler.class_name=LengthAwareEpochSampler
+data.sampler.bucket_size=1024
+data.sampler.ema_decay=0.7
+data.sampler.shuffle_batch_blocks=True
data.dataloader_num_workers=0
data.shuffle=False
```

建议保持训练 dataloader 的 `drop_last=True`。当前 qwen3 样例中的训练 dataloader 默认就是按该方式构建。

### 2.3 可选长尾保护
在某些任务中，即使做了长度重排，仍然可能存在极个别 batch 的 response 长度远超平均值。为此本特性提供了一个可选的 `max_tokens` 动态裁剪能力：

```bash
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=1
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=2.0
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=10000
```

其裁剪逻辑为：

```text
cap = min(response_length, max(min_tokens, factor * expected_len))
```

其中 `expected_len` 为当前 batch 样本在历史统计中的平均 response 长度估计值。

### 2.4 最小可复现脚本
本样例提供了一个最小可复现脚本：

```text
llm_rl/qwen3/verl-mindspeed/internal/train_grpo_qwen3_resampler_example.sh
```

该脚本保留了以下内容：

1. 16 卡 GRPO 的基本训练参数
2. Megatron 并行配置和 dist checkpoint 配置项
3. resampler 与 rollout 长尾保护相关环境变量
4. 与 qwen3 当前 patch 体系兼容的 rollout / actor 配置

同时移除了与本特性无关的内容：

1. 个人模型和数据集路径
2. profile 相关配置
3. draft train / draft dump 相关配置

使用前请按实际环境设置以下占位变量：

```bash
export MODEL_PATH=/path/to/model
export DISTCP_PATH=/path/to/dist_checkpoint
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/test.parquet
```

如果奖励函数路径不在默认位置，也可以额外指定：

```bash
export REWARD_FUNCTION_PATH=/path/to/reward_function.py
```

在 `llm_rl/qwen3/verl-mindspeed/` 目录下执行：

```bash
bash internal/train_grpo_qwen3_resampler_example.sh
```

脚本默认启用以下行为：

1. 开启 `LengthAwareEpochSampler`
2. 开启 rollout 长尾保护
3. 保持 `data.dataloader_num_workers=0`
4. 默认不保存 checkpoint；如需保存，可设置 `SAVE_CKPT_ENABLE=1`

如果需要切回 eager baseline 且关闭 resampler，可设置：

```bash
export VLLM_EAGER_BASELINE_NO_RESAMPLE=1
```

## 3. 实现方案

### 3.1 基于历史 rollout 统计的长度估计
新增 `LengthAwareEpochSampler`，继承 `AbstractCurriculumSampler`。核心做法如下：

1. 每个样本维护一个 response 长度估计值。
2. 在训练 step 完成后，通过 `response_mask.sum(-1)` 统计真实 response 长度。
3. 将统计结果按样本 id 聚合，并使用 EMA 更新样本的长度估计。
4. 在下一轮 epoch 开始时，依据估计长度排序，并按 batch block 做可选打散。

这样做的目标不是精确预测每条样本的输出长度，而是在 batch 级别减少长短样本混排。

### 3.2 稳定样本 id 回写
为了把 rollout 统计稳定回写到 dataset 样本，本优化为 `RLHFDataset.__getitem__` 返回的样本新增 `dataset_item_idx` 字段。该字段使用 dataset 内部的样本索引，保证不同 epoch 间可以回到同一条原始样本。

### 3.3 训练集按完整 batch 对齐
本优化会将训练集样本数向下截断到 `gen_batch_size` 的整数倍。这样在 `drop_last=True` 的前提下，每个 epoch 看到的是固定 prompt 集合，避免尾部零散样本在 epoch 切换时扰乱长度统计。

### 3.4 Rollout 阶段动态下发 response cap
当 dataloader sampler 为 `AbstractCurriculumSampler` 时，trainer 会在生成前读取当前 batch 的历史长度估计，并据此计算一个可选的 `response_max_tokens_cap`。随后在 `vLLM rollout` 侧通过覆写 `SamplingParams.max_tokens` 的方式，把该 cap 下发给本批次生成请求。

这样做有两个特点：

1. 不修改全局 rollout `response_length` 配置，只对当前 batch 生效。
2. 如果计算得到的 cap 不小于原始 `response_length`，则保持原行为不变。

## 4. 收益与适用场景
本优化更适用于以下场景：

1. prompt 长度相近，但 response 长度差异很大的 RL 数据集。
2. rollout 阶段存在明显长尾 batch，导致 step 时间抖动较大。
3. 关注 generation 吞吐稳定性，希望减少“极长样本拖慢整批”的现象。

在这类场景下，Length-Aware Resampler 通常能带来以下收益：

1. 降低 generation batch 内部的长短样本混排程度。
2. 减少极端长尾 batch 的出现频率。
3. 提升 rollout 吞吐稳定性，并改善 step 时间分布。

由于收益与数据分布、`n`、`gen_batch_size`、`response_length` 等配置强相关，建议结合以下指标做实际观测：
目前在n=16, gen_batch_size=32, response_length=16384场景下，qwen30b-a3b-moe模型的rollouts阶段的吞吐量能够提升约20%。

1. `timing_s/generate_sequences`
2. `perf/throughput` 或 rollout 吞吐相关指标
3. `rollout/response_max_tokens_cap`

## 5. 注意事项

1. 使用 curriculum sampler 时，建议设置 `data.dataloader_num_workers=0`，避免 dataloader worker 持有过期 sampler 状态。
2. 如果你的数据集 response 长度分布本身较为集中，该特性的收益可能有限。
3. `response_max_tokens_cap` 是一个保守的长尾保护机制。若任务对完整长响应有强依赖，应谨慎设置 `factor` 和 `min_tokens`。
