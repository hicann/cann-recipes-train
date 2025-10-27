# cann-recipes-train

## 🚀Latest News
- [2025/10] DeepSeek-R1、Qwen2.5模型样例首次上线。

## 🎉概述
cann-recipes-train仓库旨在针对LLM与多模态模型训练业务中的典型模型、算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型训练。


## ✨实践列表

|实践|简介|
|-----|-----|
|[DeepSeek-R1 RL训练优化样例](rl_train/deepseek/README.md) |基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群实现GRPO算法的高吞吐RL训练，并达到120TPS/卡的系统吞吐量。|
|[基于verl框架的Qwen2.5强化学习（入门样例）](rl_train/qwen2_5/verl_npu_demo/README.md) |基于Qwen2.5-1.5B-Instruct模型，采用verl强化学习框架，在MATH-lighteval数学推理数据集上进行了训练。本样例只需要单卡Atlas A2环境，帮助大家快速上手，使用昇腾NPU完成RL训练任务。|

## 📖目录结构说明

```
├── docs                         # 优化技术介绍文档
├── rl_train
│  ├── deepseek                  # deepseek强化学习训练相关代码
│  ├── qwen2_5                   # Qwen2.5强化学习训练相关代码
│  └── ...
└── CONTRIBUTION.md
└── README.md
└── ...
```


## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证]

cann-recipes-train仓涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应许可证文本可查阅[LICENSE](./LICENSE)
- [免责声明](DISCLAIMER.md)
