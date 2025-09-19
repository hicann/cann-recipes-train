# DeepSeek-V3 RL训练优化实践样例
本样例针对DeepSeek-V3模型，基于[veRL开源框架](https://github.com/volcengine/verl)，配合MindSpeed-LLM和vllm-ascend框架，完成RL训练全流程的优化适配。

---

# 环境准备

##  镜像创建

使用vLLM-Ascend提供的镜像，可以快速配置环境：
```shell
镜像下载命令：docker pull quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler
```
镜像使用：
```shell
# 执行以下脚本创建容器，请传入容器名称，如your_docker_name
bash run_container.sh your_docker_name
```

##  软件包安装

1、为确保版本匹配，在容器中重新安装对应的CANN软件。
   本样例使用的**PyTorch版本为2.5.1**。
   
   请参见[版本配套](../README.md#版本配套)获取并安装配套版本的CANN开发套件包、CANN二进制算子包、NNAL加速包、Ascend Extension for PyTorch。

   本样例需要安装apex库，请参考[apex](https://gitee.com/ascend/apex)构建安装。


2、安装依赖的python库。
```
pip3 install -r requirements.txt
```

3、准备源码，为了让使用者和开发者直观了解我们基于开源代码做的修改，本样例中只包含patch代码，其他代码需要使用以下命令从各自框架拉取
```shell
# 本sample所在代码仓
git clone https://gitcode.com/cann/cann_recipes_train.git

# veRL框架
git clone https://github.com/volcengine/verl.git    # 从github下载，请确保网络能访问
cd verl
git checkout 54c9b7364c2d188b2ba4107404cfa3c2b446df19
cp -r verl ../cann_recipes_train/deepseek/
cd ..

# vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1 
cp -r vllm ../cann_recipes_train/deepseek/
cd ..

# vLLM-Ascend
git clone -b v0.9.1rc2 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
cp -r vllm_ascend ../cann_recipes_train/deepseek/
cd ..

# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git # 从github下载，请确保网络能访问  
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../cann_recipes_train/deepseek/
cd ..

# MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout v2.0.0_core_r0.8.0     # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../cann_recipes_train/deepseek/
cd ..

# MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout v2.0.0
cp -r mindspeed_llm ../cann_recipes_train/deepseek/
cd ..
```

4、修改veRL代码
   为使能patch修改，需要修改1处veRL源码
```python
# 在verl/workers/megatron_workers.py: line21 处插入以下代码：
from verl_atches import prelude_patch
```

## 准备训练数据集与模型
数据集放入 ./data, 数据集准备参考: [veRL官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
模型放入 ./DeepSeek-V3-hf, 模型下载地址：[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## 执行RL后训练
```shell
# 本sample目录下启动DeepSeekV3的RL后训练
bash ./verl_patches/scripts/train_deepseekv3_256die_random_init.sh # 基于随机权重的训练脚本
bash ./verl_patches/scripts/train_deepseekv3_256die_true_weight.sh # 基于真实权重的训练脚本 
```

## 性能数据
基于Atlas 900 A3 SuperPoD超节点128卡集群，加载真实权重，Prefill/Decode阶段长度分别为1K与3K，系统吞吐达到120tps/卡。
| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_tokens | 端到端 tps | 
|---------------------|----------|-----|-----------|-------------------|------------|---------| 
| DeepSeek-R1-671B    | Atlas 900 A3 SuperPoD | 512 | 16        | 1024              | 3072       | 120     |