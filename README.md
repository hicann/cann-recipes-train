# 简介
cann-recipes-train仓库旨在针对LLM/多模态模型训练业务中的典型算法、模型，提供基于昇腾CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型训练

# 📣 Latest News
- [Aug 20, 2025]: 🚀  支持DeepSeek-R1模型

# 已支持的算法算法

- [DeepSeek-R1](deepseek/README.md)：

# 版本配套

- **CANN：8.2.RC1**
  
  本代码仓的编译执行依赖CANN开发套件包（cann-toolkit），CANN开发套件包的获取及安装方法请参见配套版本的[CANN软件安装](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。

- **Ascend Extension for PyTorch：7.1.0**
  
  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本代码仓支持的Ascend Extension for PyTorch版本为`7.1.0`，但各样例依赖的
  PyTorch版本有所不同，请根据各样例的`README.md`选择适配的PyTorch版本。
  
  Ascend Extension for PyTorch的安装方法请参见[Ascend Extension for PyTorch](https://gitee.com/ascend/pytorch#安装)。

- **固件驱动**：本仓支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1&driver=Ascend+HDK+25.2.0)”页面查询。

# License

# 免责声明
## 致cann-recipes-train使用者
1. cann-recipes-train提供的模型实现样例，仅供使用者参考，不可用于商业目的。
2. 如您在使用cann-recipes-train过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交issue，我们将及时审视并解决。


## 致数据集所有者
如果您不希望您的数据集在cann-recipes-train仓的模型样例中被提及，或希望更新模型样例中关于您的数据集描述，请在GitCode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对cann-recipes-train的理解和贡献。
