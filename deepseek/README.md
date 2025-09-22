# DeepSeek-V3 RL训练优化实践样例

## 概述

本样例针对DeepSeek-V3模型，基于[verl开源框架](https://github.com/volcengine/verl)，配合MindSpeed-LLM和vLLM-Ascend框架，完成RL训练全流程的优化适配。

## 基于Dockerfile构建环境
> 环境搭建可以基于Dockerfile快速实现，我们已经在Dockerfile里配置了必要的昇腾软件和其他第三方软件的依赖。如果遇到网络不通等问题，也可以参考附录中的[手动准备环境](#手动准备环境)章节。

1. 基于Dockerfile创建镜像
   ```shell
   # 创建docker image，请传入镜像名称，如your_image_name
   docker build -t your_image_name -f Dockerfile.vllm_ascend.mindspeed.deepseekV3 .
   # 执行以下脚本创建容器，请传入容器名称，如your_docker_name，镜像名称同上一步
   bash run_container.sh your_docker_name your_image_name:latest
   ```

2. 源码准备
   ```shell
   # 下载本sample所在代码仓
   git clone https://gitcode.com/cann/cann-recipes-train.git
   # 添加Dockerfile中已经准备好的依赖文件
   cp -r /workspace/verl/verl ./cann-recipes-train/deepseek/
   cp -r /workspace/vllm/vllm ./cann-recipes-train/deepseek/
   cp -r /workspace/vllm-ascend/vllm_ascend ./cann-recipes-train/deepseek/
   cp -r /workspace/Megatron-LM/megatron ./cann-recipes-train/deepseek/
   cp -r /workspace/MindSpeed/mindspeed ./cann-recipes-train/deepseek/
   cp -r /workspace/MindSpeed-LLM/mindspeed_llm ./cann-recipes-train/deepseek/
   # 进入本sample目录
   cd ./cann-recipes-train/deepseek/
   pip install -r requirements.txt
   ```
   修改veRL源码使能patch修改：
   ```python
   # 在verl/workers/megatron_workers.py: line21 处插入以下代码：
   from verl_atches import prelude_patch
   ```

## 数据集与模型准备

- 数据集准备方法请参考[verl官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)，并将其放入数据集放入`./data`路径。

- 模型请从[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)下载，并将其放入`./DeepSeek-V3-hf`路径。

## RL后训练执行

在本样例代码下载根目录下启动DeepSeekV3的RL后训练。

```shell
# 基于随机权重的训练脚本
bash ./verl_patches/scripts/train_deepseekv3_256die_random_init.sh
 # 基于真实权重的训练脚本 
bash ./verl_patches/scripts/train_deepseekv3_256die_true_weight.sh
```

## 性能数据
基于Atlas 900 A3 SuperPoD超节点128卡集群，加载真实权重，Prefill/Decode阶段长度分别为1K与3K，系统吞吐可达到120TPS/卡。
| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_tokens | 端到端TPS | 
|---------------------|----------|-----|-----------|-------------------|------------|---------| 
| DeepSeek-R1-671B    | Atlas 900 A3 SuperPoD | 512 | 16        | 1024              | 3072       | 120     |


## 附录
### 手动准备环境

1. 创建vLLM-Ascend镜像。

   ```shell
   # 镜像下载
   docker pull quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler

   # 执行以下脚本创建容器，请传入容器名称，如your_docker_name
   bash run_container.sh your_docker_name quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler
   ```
2. 在容器中安装CANN软件包与Ascend Extension for PyTorch软件包。
   - **CANN：8.2.RC1**

      请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载如下软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)进行安装。

      - 开发套件包：`Ascend-cann-toolkit_${version}_linux-${arch}.run`、
      - 二进制算子包：`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`
      - NNAL加速包：`Ascend-cann-nnal_${version}_linux-${arch}.run`

   - **Ascend Extension for PyTorch：7.1.0**

      Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.1.0`，PyTorch版本为`2.5.1`。

      请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=7.1.0&cann=8.2.RC1)下载`Ascend Extension for PyTorch 7.1.0-PyTorch2.5.1`软件包，并参考[Ascend Extension for PyTorch安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)进行安装。
      
      > 说明：本样例需要安装apex库，请参考[apex](https://gitee.com/ascend/apex)构建安装。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-train.git

    # 安装依赖的python库
    cd cann-recipes-train/deepseek
    pip3 install -r requirements.txt
    ```

4. 下载依赖的开源框架代码。
   
   为了让使用者和开发者直观了解我们基于开源代码做的修改，本样例中只包含patch代码，其他框架代码需要拉取。

   返回cann-recipes-train项目代码上级目录，即执行git clone命令时所在目录，并执行如下命令，需注意，请确保环境能够正常连通网络。
   ```shell
   # 下载verl源码
   git clone https://github.com/volcengine/verl.git
   cd verl
   git checkout 54c9b7364c2d188b2ba4107404cfa3c2b446df19
   cp -r verl ../cann-recipes-train/deepseek/
   cd ..

   # 下载vLLM源码
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.9.1 
   cp -r vllm ../cann-recipes-train/deepseek/
   cd ..

   # 下载vLLM-Ascend源码
   git clone -b v0.9.1rc2 https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   cp -r vllm_ascend ../cann-recipes-train/deepseek/
   cd ..

   # 下载Megatron-LM源码
   git clone https://github.com/NVIDIA/Megatron-LM.git  
   cd Megatron-LM
   git checkout core_r0.8.0
   cp -r megatron ../cann-recipes-train/deepseek/
   cd ..

   # 下载MindSpeed源码
   git clone https://gitee.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout v2.0.0_core_r0.8.0
   pip install -r requirements.txt 
   cp -r mindspeed ../cann-recipes-train/deepseek/
   cd ..

   # 下载MindSpeed-LLM源码
   git clone https://gitee.com/ascend/MindSpeed-LLM.git
   cd MindSpeed-LLM
   git checkout v2.0.0
   cp -r mindspeed_llm ../cann-recipes-train/deepseek/
   cd ..
   ```

5. 修改verl代码。

   为了使能patch修改，需要修改以下verl源码。
   ```python
   # 在verl/workers/megatron_workers.py: line21处插入以下代码：
   from verl_atches import prelude_patch
   ```