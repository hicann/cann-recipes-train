# History Tree Dynamic RL 投机解码

## 1. 简介

Dynamic RL 是面向 RL Rollout 长序列采样的投机解码策略。它把同一套 Rollout 请求在解码过程中的历史响应复用能力和 EAGLE3 草稿模型能力组合起来，在不同 batch size 和不同阶段下动态选择更合适的草稿来源：

- `history_tree`：从上一轮 Rollout 产生的响应中提取高 reward 片段，构建固定长度历史草稿缓存，在后续相同 prompt 或相似上下文中直接给出草稿 token。
- `eagle3`：使用独立 EAGLE3 draft model 生成草稿 token，适合较小 batch size 下的长尾解码阶段。
- `dynamic_rl`：同时接入 `history_tree` 与 `eagle3`，按 batch size、历史缓存预热状态和在线 timing 统计动态切换。

本文档覆盖当前 patch 已实现的 Dynamic RL 相关能力。

## 2. 使用说明

### 2.1 最小启动配置

在 veRL 启动脚本中配置 Rollout 投机方法为 `dynamic_rl`，并指定 EAGLE3 draft model：

```bash
actor_rollout_ref.rollout.spec_method='dynamic_rl'
actor_rollout_ref.rollout.eagle3_draft_model='/path/to/Qwen3-30B-moe-eagle3'
actor_rollout_ref.rollout.spec_num_speculative_tokens=4
```

当前 EAGLE3 分支使用 chain draft，只需要配置 draft model 路径和草稿 token 数。

如果只验证历史响应复用分支，可将 `spec_method` 设置为 `history_tree`，此时不需要 `eagle3_draft_model`：

```bash
actor_rollout_ref.rollout.spec_method='history_tree'
actor_rollout_ref.rollout.spec_num_speculative_tokens=4
```

仓库内提供了 Qwen3-30B 16 卡 GRPO 最小复现脚本：

```bash
cd llm_rl/qwen3/verl-mindspeed
MODEL_PATH=/path/to/Qwen3-30B-A3B \
DISTCP_PATH=/path/to/Qwen3-30B-A3B_megatron \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/test.parquet \
bash internal/train_grpo_qwen3_30b_16die_dynamic_rl.sh
```

脚本默认使用 `/home/data/Qwen3-30B-moe-eagle3` 作为 EAGLE3 draft model 路径。如需覆盖，可在脚本末尾通过 Hydra 参数传入：

```bash
bash internal/train_grpo_qwen3_30b_16die_dynamic_rl.sh \
    actor_rollout_ref.rollout.eagle3_draft_model=/path/to/Qwen3-30B-moe-eagle3
```

### 2.2 配置项

| 配置 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `actor_rollout_ref.rollout.spec_method` | veRL Hydra 参数 | `null` | 可设为 `history_tree`、`eagle3` 或 `dynamic_rl`。 |
| `actor_rollout_ref.rollout.eagle3_draft_model` | veRL Hydra 参数 | `null` | EAGLE3 draft model 路径；`dynamic_rl` 与 `eagle3` 需要。未配置时会尝试自动查找常见本地路径。 |
| `actor_rollout_ref.rollout.spec_num_speculative_tokens` | veRL Hydra 参数 | `4` | 每次生成的草稿 token 数。 |
| `VLLM_HISTORY_TREE_MAX_SPEC_REQS` | 环境变量 | `64` | history tree 分支参与投机的最大请求数。 |
| `VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ` | 环境变量 | `8` | batch size 小于等于该值时允许考虑 EAGLE3 分支。 |
| `VLLM_DYNAMIC_RL_EAGLE_PROBE_MAX_BSZ` | 环境变量 | `2` | timing guard 策略下，允许主动探测 EAGLE3 的最大 batch size。 |
| `VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE` | 环境变量 | `VLLM_HISTORY_TREE_MAX_SPEC_REQS` | history tree 分支的 batch size 上限。 |
| `VLLM_DYNAMIC_RL_POLICY` | 环境变量 | `timing_guard` | 动态切换策略。支持 `timing_guard`、`history_first`、`threshold`、`threshold_only`。 |
| `VLLM_DYNAMIC_RL_HISTORY_WARMUP_RECORDS` | 环境变量 | `1` | 至少同步多少条历史记录后认为 history cache 已预热。 |
| `VLLM_DYNAMIC_RL_EAGLE_COLD_START_GUARD` | 环境变量 | `1` | 未完成 history 预热前，是否禁止冷启动 EAGLE3。 |
| `VLLM_DYNAMIC_RL_COMPARE_AFTER_STEPS` | 环境变量 | `64` | 每个 batch bucket 中，history 分支累计多少步后开始比较 EAGLE3。 |
| `VLLM_DYNAMIC_RL_EAGLE_PROBE_STEPS` | 环境变量 | `8` | EAGLE3 探测步数。 |
| `VLLM_DYNAMIC_RL_SCORE_EMA_ALPHA` | 环境变量 | `0.3` | tokens/ms 在线评分的 EMA 更新系数。 |
| `VLLM_DYNAMIC_RL_SWITCH_MARGIN` | 环境变量 | `1.02` | 切换分支需要超过对方评分的倍率。 |
| `VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_STEPS` | 环境变量 | `8192` | EAGLE3 探测效果较差后回退 history 的冷却步数。 |
| `VLLM_DYNAMIC_RL_EAGLE_MAX_COOLDOWN_STEPS` | 环境变量 | `65536` | EAGLE3 冷却步数上限。 |
| `VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_GROWTH` | 环境变量 | `4.0` | 连续 bad probe 后冷却步数增长倍率。 |
| `VLLM_DYNAMIC_RL_EAGLE_BAD_STEP_RATIO` | 环境变量 | `0.98` | EAGLE3 单步 tokens/ms 低于 history 评分该倍率时，判定为 bad step。 |
| `VLLM_DYNAMIC_RL_EAGLE_ENFORCE_EAGER` | 环境变量 | `0` | `dynamic_rl` 的 EAGLE3 分支默认启用 draft graph；设为 `1` 时走 eager 路径。 |
| `VLLM_ASCEND_SPEC_TIMING` | 环境变量 | 关闭 | 是否采集 speculative decoding 分段耗时。 |
| `VLLM_ASCEND_SPEC_TIMING_LOG_EVERY` | 环境变量 | `1` | timing 日志采集步长。 |
| `VLLM_ASCEND_SPEC_TIMING_FIRST_N` | 环境变量 | `20` | 前 N 步强制采集 timing。 |

## 3. 实现方案

### 3.1 veRL 接入

veRL 侧新增 Rollout 配置 `spec_method`、`eagle3_draft_model`、`spec_num_speculative_tokens`。当 `spec_method` 为 `history_tree` 或 `dynamic_rl` 时，训练主流程在每轮 Rollout 前后传递历史记录，并由各 Rollout worker 更新本地 history cache。

Rollout 结束后，trainer 会从 batch 中提取：

- prompt token ids，用于生成稳定的 `prompt_id`；
- response token ids，用于构造历史草稿；
- token-level reward，用于在同一 prefix 下选择更优候选。

这些记录通过 `history_tree_records` 传入 vLLM Rollout worker，并同步到 proposer 的 history cache。

相关代码：

- [0020-verl-feature-enable_history_tree_dynamic_rl.patch](../../llm_rl/qwen3/verl-mindspeed/patches/verl/0020-verl-feature-enable_history_tree_dynamic_rl.patch)

### 3.2 vLLM 配置与 EAGLE3 适配

vLLM 侧将 `history_tree` 和 `dynamic_rl` 注册为 speculative method，并复用 EAGLE3 的 draft model 配置路径。`dynamic_rl` 在 EAGLE 分支中会被标准化为 EAGLE3 chain draft 行为，确保 auxiliary hidden states 与 EAGLE3 draft head 正确启用。

相关代码：

- [0004-vllm-feature-enable_history_tree_dynamic_rl.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm/0004-vllm-feature-enable_history_tree_dynamic_rl.patch)

### 3.3 vLLM-Ascend proposer

vLLM-Ascend 侧新增两个 proposer：

- `HistoryRolloutProposer`：维护按 prompt 隔离的 `FixedDraftHistoryCache`，同时保留全局 fallback cache。它从历史响应中按固定 prefix 长度提取候选草稿，并按 reward 优先更新。
- `DynamicProposer`：内部持有 `HistoryRolloutProposer` 与 `EagleProposer`，在每个 decode step 基于 batch bucket、预热状态和 timing 统计选择当前分支。

`dynamic_rl` 默认策略 `timing_guard` 的核心逻辑：

1. 大 batch 直接使用 history rollout，避免 EAGLE3 在高 batch 下引入额外开销。
2. history cache 未预热时，优先收集历史分支 timing。
3. 小 batch 进入长尾阶段后，按 bucket 记录 history 和 EAGLE3 的 tokens/ms EMA。
4. 只有当 EAGLE3 评分超过 history 评分并满足 `VLLM_DYNAMIC_RL_SWITCH_MARGIN` 时才切换。
5. 如果 EAGLE3 探测效果差，则回退 history 并进入 cooldown。

相关代码：

- [0007-vllm_ascend-feature-tree_attention_backend.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0007-vllm_ascend-feature-tree_attention_backend.patch)：为 EAGLE3 tree draft 提供 Ascend tree attention、稀疏输入准备 kernel 和自定义算子库路径补充。
- [0008-vllm_ascend-feature-tree_rejection_sampler.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0008-vllm_ascend-feature-tree_rejection_sampler.patch)：新增 `AscendTreeRejectionSampler`，用于 tree draft 验证和采样结果回填。
- [0009-vllm_ascend-feature-history_tree_cache_proposer.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0009-vllm_ascend-feature-history_tree_cache_proposer.patch)：新增 history cache、`HistoryRolloutProposer` 和 speculative method 枚举。
- [0010-vllm_ascend-feature-eagle3_tree_draft.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0010-vllm_ascend-feature-eagle3_tree_draft.patch)：扩展 `EagleProposer`，支持 EAGLE3 tree draft、chain fallback 和稀疏 tree 输入桥接。
- [0011-vllm_ascend-feature-dynamic_rl_proposer.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0011-vllm_ascend-feature-dynamic_rl_proposer.patch)：新增 `DynamicProposer` 并注册 `history_tree`、`dynamic_rl` speculative method。
- [0012-vllm_ascend-feature-history_dynamic_rl_worker_integration.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0012-vllm_ascend-feature-history_dynamic_rl_worker_integration.patch)：在 NPU worker 中接入 History Tree/Dynamic RL 分支选择、timing 统计、tree rejection sampler 和 draft token 同步。
- [0013-vllm_ascend-bugfix-dynamic_rl_moe_compat.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0013-vllm_ascend-bugfix-dynamic_rl_moe_compat.patch)：修复 Dynamic RL 场景下 MoE gating renorm 和 chunk MoE 结果写回兼容问题。

## 4. 指标与观测

Dynamic RL 会通过 Rollout `meta_info["metrics"]` 汇总投机解码指标，包括总体指标和按分支拆分的指标：

| 指标 | 说明 |
| --- | --- |
| `speculative_decoding/effective_percent` | 有效投机次数 / 总投机次数。 |
| `speculative_decoding/effective_length` | 有效投机平均接受长度。 |
| `speculative_decoding/predict_times` | 总投机次数。 |
| `speculative_decoding/effective_times` | 有效投机次数。 |
| `speculative_decoding/total_right_length` | 总接受草稿长度。 |
| `speculative_decoding/dynamic_rl/eagle_*` | EAGLE3 分支独立统计。 |
| `speculative_decoding/dynamic_rl/history_*` | history rollout 分支独立统计。 |

如需分析分支切换原因，可开启 speculative timing：

```bash
export VLLM_ASCEND_SPEC_TIMING=1
export VLLM_ASCEND_SPEC_TIMING_LOG_EVERY=200
export VLLM_ASCEND_SPEC_TIMING_FIRST_N=5
```

timing 信息中会记录 `dynamic_mode`、`dynamic_reason`、`dynamic_bucket`、`dynamic_history_ready`、`dynamic_eagle_tokens_per_ms`、`dynamic_history_tokens_per_ms` 等字段，用于判断当前 step 选择 history 还是 EAGLE3。

## 5. 使能效果

我们在 Qwen3-30B-A3B MoE 模型上，于 Deepscaler 数学推理数据集的真实 GRPO Rollout 场景中进行了端到端验证。

相关配置：

- 数据集：Deepscaler train/test parquet。
- 最大 prompt 长度（`max_prompt_length`）：1024。
- 最大输出长度（`max_response_length`）：16384。
- 训练 batch 大小（`train_batch_size`）：64。
- Rollout 采样数（`rollout_n`）：8。
- Rollout 最大请求数（`max_num_seqs`）：64。
- Rollout 模型张量并行（`tensor_model_parallel_size`）：4。
- 卡数：单机 16 卡。
- 投机 token 个数（`spec_num_speculative_tokens`）：4。
- Dynamic RL 策略：`VLLM_DYNAMIC_RL_POLICY=timing_guard`，`VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE=64`，`VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ=8`。
- `dynamic_rl + length-aware` 同时启用 Length-Aware Resampler，并在日志中动态下发 `rollout/response_max_tokens_cap=10000`。

性能收益：

单步总推理时间对应日志中的 `timing_s/generate_sequences`，单步总时间对应 `perf/time_per_step`。

Qwen3-30B-A3B（HistorySpec，非 length-aware）

| 策略 | 取值 step | 平均输出长度 | 单步总推理时间 / s | 单步总时间 / s | 吞吐 / tokens/s |
| --- | --- | --- | --- | --- | --- |
| baseline | 1-10 | 5916.21 | 1202.77 | 1511.84 | 126.86 |
| HistorySpec（`history_tree`） | 1-10 | 6227.08 | 1048.85 | 1369.69 | 148.79 |
| 收益 | - | - | 12.80% | 9.40% | +17.28% |

Qwen3-30B-A3B（Dynamic RL，非 length-aware）

| 策略 | 取值 step | 平均输出长度 | 单步总推理时间 / s | 单步总时间 / s | 吞吐 / tokens/s |
| --- | --- | --- | --- | --- | --- |
| baseline | 1-10 | 5916.21 | 1202.77 | 1511.84 | 126.86 |
| Dynamic RL | 1-10 | 6215.13 | 1073.45 | 1392.66 | 146.59 |
| 收益 | - | - | 10.75% | 7.88% | +15.55% |

Qwen3-30B-A3B（Dynamic RL + Length-Aware）

| 策略 | 取值 step | response cap | 平均输出长度 | 单步总推理时间 / s | 单步总时间 / s | 吞吐 / tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1-10 | - | 5916.21 | 1202.77 | 1511.84 | 126.86 |
| Dynamic RL + Length-Aware | 1-10 | 10000 | 6187.13 | 998.65 | 1213.48 | 170.16 |
| 收益 | - | - | - | 16.97% | 19.74% | +34.13% |

## 6. 约束与注意事项

- `dynamic_rl` 需要可加载的 EAGLE3 draft model；如果只验证历史复用能力，请使用 `history_tree`。
- 当前 EAGLE3 分支使用 chain draft，只通过 `spec_num_speculative_tokens` 控制每步草稿长度。
- `dynamic_rl` 的第一轮 Rollout 没有历史响应可用，history cache 需要在后续 Rollout 中逐步预热。
- 如果使用 MoE 模型，仍需按模型并行方式配置 `USE_ALLTOALL_OVERLAP`、`ALL_TO_ALL_RESHARD`、`VLLM_ENABLE_EXPERT_PARALLEL` 等基础 MoE 环境变量。
