# cann-recipes-train

## 🚀Latest News
- [2026/02] 新增Qwen3系列模型RL训练使能npugraph_ex图模式样例
- [2025/12] 新增Qwen3系列模型RL训练使能SAM投机推理、tool agent RL样例
- [2025/11] Qwen3模型长序列RL样例首次上线。
- [2025/10] DeepSeek-R1、Qwen2.5模型样例首次上线。

## 🎉概述
cann-recipes-train仓库旨在针对LLM与多模态模型训练业务中的典型模型、算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型训练。


## ✨实践列表

|实践|简介|
|-----|-----|
|[DeepSeek-R1 RL训练优化样例](llm_rl/deepseek/README.md) |基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群实现GRPO算法的高吞吐RL训练，并达到120TPS/卡的系统吞吐量。|
|[基于verl框架的Qwen2.5强化学习（入门样例）](llm_rl/qwen2_5/verl_npu_demo/README.md) |基于Qwen2.5-1.5B-Instruct模型，采用verl强化学习框架，在MATH-lighteval数学推理数据集上进行了训练。本样例只需要单卡Atlas A2环境，帮助大家快速上手，使用昇腾NPU完成RL训练任务。|
|[Qwen3-235B-A22B RL训练优化样例](llm_rl/qwen3/README.md) | 基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群实现GRPO/DAPO算法的**长序列 2k+32k**训练，GRPO达到120TPS/卡的系统吞吐量。|
|[Qwen3-32B RL训练使能SAM投机推理样例](llm_rl/qwen3/README.md) | 基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群，GRPO/DAPO算法的2k+32k训练场景下，使能**SAM投机推理特性**，达成**10%性能提升**。|
|[Qwen3 tool agent RL训练样例](agent_rl/qwen3_tool_agent/README.md) |基于verl/recipe中的retool项目，调用Sandbox工具，使能`asyncLLM`和`agent_loop`特性，在昇腾NPU上完成端到端agent RL训练任务。|

## 特性介绍
本项目在探索最佳实践的过程中引入了如下特性：

|特性|介绍|
|----|---|
|SAM无损投机推理 |[docs/features/sam_speculative_decoding.md](docs/features/sam_speculative_decoding.md)|
|RL On-Policy 推理场景的序列级均衡调度引擎| [docs/features/rollout_rebalance.md](docs/features/rollout_rebalance.md)|

## 📖目录结构说明

```
├── docs                         # 优化技术介绍文档
├── llm_rl                       # llm强化学习训练相关代码
│  ├── deepseek                  # deepseek强化学习训练相关代码
│  ├── qwen2_5                   # Qwen2.5强化学习训练相关代码
│  ├── qwen3                     # Qwen3强化学习训练相关代码
│  └── ...
├── agent_rl                     # agent强化学习训练相关代码
│  ├── qwen3_tool_agent
│  └── ...
├── multimodal_rl                # 多模态强化学习训练相关代码
├── llm_sft                      # llm有监督微调训练相关代码
├── CONTRIBUTION.md
├── README.md
└── ...
```


## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- 许可证

    cann-recipes-train仓涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应许可证文本可查阅[LICENSE](./LICENSE)
- [免责声明](DISCLAIMER.md)
