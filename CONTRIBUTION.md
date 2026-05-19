# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则、签署CLA协议，并熟悉源码仓的贡献流程。

开发者在准备本地代码与提交PR时，请重点关注以下几点：

1. **PR描述**：提交PR时，请按照模板仔细填写业务背景、目的、方案等信息。
2. **重大修改讨论**：若涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：反馈BUG、建议特性、改进文档、修复问题、新增样例。

## 样例贡献流程

感谢您参与样例建设！为了确保贡献顺畅，建议您遵循 **“RFC讨论 -> 代码开发 -> 提交PR”** 的流程。

### 1. 提交 RFC，达成方案共识
在正式开发前，建议先提交 Issue 进行 RFC（请求评议），包含：
- **背景与动机**：解决什么场景需求，提供什么参考价值。
- **核心设计**：技术思路、关键模块（模型选型、流程架构等）。
- **预期目标**：核心功能、精度指标或性能表现。
- **排期**：预计完成时间。

### 2. 提交 PR，完善交付内容
方案达成共识且代码开发完成后，请提交 PR 并关联 RFC Issue。请在开发前先了解本节要求，避免交付件遗漏。

#### 2.1 PR 内容
完整样例的 PR 内容需包含，如果是已有样例修改，请根据实际情况增减：
- **样例代码**：技术实现代码。
- **优化文档**：说明功能适配、性能优化的重点（原因、方法、收益）。
- **README文档**：
  - **简述**：核心内容（模型、设备、精度/性能数据）。
  - **操作步骤**：详细的复现流程（从环境准备到执行）。

#### 2.2 提交 PR 后需要做的事情
- **触发流水线检查**：请在评论区评论 `compile` 进行流水线检查；如果流水线失败，请根据提示修复问题或者申请屏蔽。
- **发起评审**：需要代码 review 时，请在评论区 `@committer` 或 `@maintainer` 协助处理，需要不少于两位 `committer` 或 `maintainer` 检视通过，相关人员列表可参见 [cann-community](https://gitcode.com/cann/community/blob/master/CANN/sigs/recipes/README.md#-sig%E6%88%90%E5%91%98members)。
- **新样例推广**：如果提交的是全新样例，欢迎在 SIG 例会上做简要介绍，帮助更多开发者了解样例的适用场景与使用方式。sig议题申报及会议时间请参见 [recipes-sig会议](https://gitcode.com/cann/community/blob/master/CANN/sigs/recipes/README.md#-%E8%81%94%E7%B3%BB%E6%96%B9%E5%BC%8F-contact--participation)。
- **合入条件**：具备以下标签后，PR会自动合入：
  1. `ci-pipeline-passed`（流水线检查通过）
  2. `cann-cla/yes`（PR提交人和代码作者签署cla协议）
  3. `lgtm`（有 committer/maintainer 评论2个 `/lgtm` 或 1个 `/lgtm` + 1个 `/approve`）
  4. `approve`（有 committer/maintainer 评论 `/approve`）

### 3. 开发规范

#### 3.1 通用规范
所有新增样例均应遵循以下要求：
- **目录组织**：样例应按训练任务类别放入对应目录，并以模型名命名（如 `llm_rl/qwen3`），包含当前样例执行所需的代码和文档。
- **代码风格**：请使用 `pre-commit` 工具确保代码满足基本规范，安装后会在 git commit 时自动检查：
  ```shell
  pip install pre-commit
  pre-commit install
  ```

- **无二进制文件**：除必要的文档配图外，代码库中不应包含二进制文件。
- **数据集**：涉及第三方数据集时，仅在文档说明下载和使用方式，**勿**直接提交数据集。
- **公共代码验证**：若修改涉及公共代码，需确保通过 CI 验证。
- **算子依赖**：涉及算子新增或修改，请先合入算子仓，再推进样例合入。
- **License 合规**：检查使用的 License 是否合规（推荐 Apache 2.0 或 MIT），并正确标注版权。

#### 3.2 框架代码修改规范
当样例需要修改框架本身代码（如 `verl`、`vllm`、`vllm_ascend` 等）时，需要额外遵循以下规范。

**修改原则：**
- **大修改（完整替换/Monkey Patch）**：针对文件修改量大或逻辑复杂的情况，请在 `patches/<框架名>` 下**复刻原仓目录结构**，放置修改后的完整 Python 文件。
- **小修改（Git Patch）**：针对少量代码修改（Bugfix/参数调整），请基于**特性（Feature）粒度**将多个文件修改合并为一个 `.patch` 文件，置于 `patches/<框架名>` 根目录下。
  - **Patch 命名规范：**`编号-框架名-类型-特性名.patch` （类型：`feature` 或 `bugfix`；特性名：全小写，下划线连接）
  - **示例**：`0001-verl-feature-enable_alltoall_overlap.patch`

**目录结构示例：**
```text
qwen3                                          # 样例名
├── patches                                    # 修改补丁总目录
│   ├── verl                                   # 框架A (如 verl)
│   │   ├── trainer                            # [大修改] 目录，对应原仓路径
│   │   │   └── verl_trainer_adaptor.py        # 完整替换或Monkey Patch文件
│   │   ├── utils                              # [大修改] 目录，对应原仓路径
│   │   │   └── ...
│   │   └── 0001-verl-feature-enable_alltoall_overlap.patch  # [小修改] 特性粒度 patch
│   └── vllm_ascend                            # 框架B(如vllm_ascend)
│       └── ...
├── internal                                   # 框架无关的内部文件
├── Dockerfile                                 # 环境部署文件
├── prelude_patch.py                           # Monkey Patch 加载的入口文件（如有）
├── requirements.txt                           # 依赖库（需锁定版本）
└── README.md                                  # 说明文档
```

如果新增或修改了已有样例 patch，请在项目根目录下执行 `./ci/validate_all_projects.sh` 进行校验。当前该脚本仅校验 `llm_rl/qwen3` 样例，主要检查 patch 文件命名、框架源码下载与拷贝，以及 patch 能否成功应用。

涉及框架代码修改时，校验过程包含以下步骤：
1. 下载样例所需的框架代码，如verl, vllm, vllm_ascend等。
2. 将框架代码复制到样例文件夹下，复刻原仓目录结构。
3. 依次用`git apply`应用所有patch，确保改动能够正确应用，没有冲突。

---

如有任何疑问，欢迎在社区交流。再次感谢您的支持！
