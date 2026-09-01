# Qwen3-30B-A3B Medical SFT Training Example

This example uses the [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) framework to fine-tune `Qwen3-30B-A3B` on a medical domain SFT task. Training effectiveness is measured via Keyword Recall on medical Q&A samples.

The [Medical R1](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data) dataset (question/think/answer three-field format) is used for training. The MoE parallelism config (EP=8) enables full-parameter fine-tuning on a single node with 16 cards. Evaluation uses vLLM + vLLM-Ascend to compare the base model, CPT checkpoint, and SFT model under the same conditions.

## Supported Products

| Item | Spec |
| --- | --- |
| Product | Atlas A3 series |
| Recommended cards | 16 (EP=8) |
| CANN version | 9.0.0 |
| Python | 3.11 |
| Training framework | torchtitan-npu |
| Inference framework | vLLM + vLLM-Ascend |

## Files

| File | Description |
| --- | --- |
| `README_EN.md` | This document |
| `README.md` | Chinese documentation |
| `config_registry_medical.py` | torchtitan-npu Qwen3-30B-A3B medical SFT config |
| `run_medical_sft.sh` | Training launch script (copy to torchtitan-npu dir before running) |
| `prepare_medical_r1_dataset.py` | Medical R1 dataset split tool |

## Environment Setup

### 1. Docker Container

Use an Ascend training image with CANN 9.0.0 and Python 3.11 pre-installed. Example for single-node 16-card setup:

```shell
docker run -itd \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home:/home \
  -v /data:/data \
  --net=host \
  --shm-size=128g \
  --privileged \
  --name qwen3_30b_medical_sft \
  cann:9.0.0-a3-openeuler24.03-py3.11 \
  /bin/bash
```

Initialize CANN after entering the container. CANN paths vary by deployment method — adjust according to your environment:

```shell
# Docker image default path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Conda installation path (example for CANN 9.0.0)
source /home/developer/Ascend/cann-9.0.0/set_env.sh
source /home/developer/Ascend/nnal/atb/set_env.sh

# If the system libstdc++ is too old
export LD_PRELOAD=/path/to/conda/envs/torchtitan/lib/libstdc++.so.6
```

### 2. Install torchtitan-npu

```shell
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
pip install -r requirements.txt
pip install -e .
```

## Dataset

Download `r1_data_example.jsonl` from ModelScope and place it in the `assets` directory of torchtitan-npu:

```shell
cd /path/to/torchtitan-npu
mkdir -p assets
# Manually download from
# https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data/files
# to assets/
ls assets/r1_data_example.jsonl
```

Then split using the provided script:

```shell
python /path/to/recipe/prepare_medical_r1_dataset.py \
  --input ./assets/r1_data_example.jsonl \
  --output ./assets/medical_r1
```

Split result:

| Dataset | Samples | Use |
| --- | ---: | --- |
| `train.jsonl` | ~2,166 | SFT training |
| `test.jsonl` | ~241 | Keyword Recall evaluation |

## Model Weights

Download `Qwen3-30B-A3B` weights (~60 GB) from ModelScope and create a symlink in the torchtitan-npu source directory:

```shell
pip install modelscope
mkdir -p /data/models/Qwen3-30B-A3B

modelscope download \
  --model Qwen/Qwen3-30B-A3B \
  --local_dir /data/models/Qwen3-30B-A3B

cd /path/to/torchtitan-npu
mkdir -p assets/hf
ln -sf /data/models/Qwen3-30B-A3B assets/hf/Qwen3-30B-A3B
```

## Training Configuration

### Config Registration

Copy `config_registry_medical.py` to the torchtitan-npu source:

```shell
cp /path/to/recipe/config_registry_medical.py \
  /path/to/torchtitan-npu/torchtitan_npu/models/qwen3/config_registry_medical.py
```

Then append the following to `torchtitan_npu/models/qwen3/config_registry.py`:

```python
from torchtitan_npu.models.qwen3.config_registry_medical import (
    sft_qwen3_30ba3b_medical,
    sft_qwen3_30ba3b_medical_tnd,
)
```

### Parallelism Strategy

Single-node 16-card MoE parallelism (CP=2, EP=8, TP=2):

| Parameter | Value | Description |
| --- | ---: | --- |
| `NGPU` | 16 | Total cards |
| `context_parallel_degree` | 2 | Context parallelism |
| `tensor_parallel_degree` | 2 | Tensor parallelism |
| `expert_parallel_degree` | 8 | 128 experts sharded along EP |
| `pipeline_parallel_degree` | 1 | PP disabled |
| `data_parallel_shard_degree` | -1 | FSDP full shard (mesh size = 4) |

### Hyperparameters

| Config | Recommended Value | Description |
| --- | --- | --- |
| `steps` | 156 | Training steps (5 epochs, ~31 steps/epoch) |
| `lr` | 2e-5 | Learning rate |
| `warmup_steps` | 5 | Warmup steps |
| `local_batch_size` | 1 | Per-device batch size |
| `seq_len` | 4096 | Sequence length |
| `activation_checkpoint` | selective | Selective recomputation |
| `TRAIN_DATA` | Split training set | Training data path, set via `TRAIN_DATA` env var |
| `MODEL_DIR` | `assets/hf/Qwen3-30B-A3B` | HF weights path |

### Sample Format

This example uses the R1 think template format, wrapping the dataset's `think` field in `<think>` tags:

```python
def _process_sample(sample):
    output = f"<think>\n{sample['think']}\n</think>\n\n{sample['answer']}"
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": output},
    ]
```

### Attention Variants

| Config function | Attention type | Description |
| --- | --- | --- |
| `sft_qwen3_30ba3b_medical` | BSND (SDPA) | Reference only |
| `sft_qwen3_30ba3b_medical_tnd` | TND (NPUVarlenAttention) | **Recommended, validated** |

Use `CONFIG=sft_qwen3_30ba3b_medical_tnd` for TND variant.

## Training

### Launch

Copy the launch script to the torchtitan-npu directory and execute:

```shell
cp /path/to/recipe/run_medical_sft.sh /path/to/torchtitan-npu/
cd /path/to/torchtitan-npu
bash run_medical_sft.sh
```

The script uses environment variables `NGPU=16` and `CONFIG=sft_qwen3_30ba3b_medical_tnd` for the TND variant. Example log output (EP=8):

```text
step:    1  loss:  1.45426  memory:  37.73GiB(61.58%)  tps:    59   69.018s  (compilation)
step:    2  loss:  1.39178  memory:  52.27GiB(85.31%)  tps:   798    5.135s
step:    3  loss:  1.26931  memory:  52.31GiB(85.37%)  tps:  1215    3.370s
step:   10  loss:  1.02183  memory:  52.44GiB(85.59%)  tps:   993    4.126s
step:   20  loss:  0.95751  memory:  52.44GiB(85.59%)  tps:  1199    3.416s
step:   31  loss:  0.70617  memory:  52.44GiB(85.59%)  tps:  1345    3.046s   ← epoch 1 end
step:   32  loss:  0.67716  memory:  52.44GiB(85.59%)  tps:   701    5.842s
step:   50  loss:  0.58786  memory:  52.50GiB(85.69%)  tps:  1010    4.056s
step:   62  loss:  0.34057  memory:  52.56GiB(85.79%)  tps:  1177    3.479s   ← epoch 2 end
step:   63  loss:  0.33076  memory:  52.56GiB(85.79%)  tps:   803    5.102s
step:   90  loss:  0.19230  memory:  52.56GiB(85.79%)  tps:   733    5.590s
step:   93  loss:  0.16940  memory:  52.56GiB(85.79%)  tps:  1014    4.040s   ← epoch 3 end
step:   94  loss:  0.16507  memory:  52.56GiB(85.79%)  tps:  1286    3.185s
step:  120  loss:  0.08754  memory:  52.62GiB(85.88%)  tps:   942    4.349s
step:  124  loss:  0.08219  memory:  52.62GiB(85.88%)  tps:  1257    3.260s   ← epoch 4 end
step:  125  loss:  0.08480  memory:  52.62GiB(85.88%)  tps:  1274    3.215s
step:  150  loss:  0.04411  memory:  52.62GiB(85.88%)  tps:   918    4.462s
step:  155  loss:  0.04376  memory:  52.62GiB(85.88%)  tps:  1199    3.416s
step:  156  loss:  0.04450  memory:  52.62GiB(85.88%)  tps:  1244    3.292s   ← end (epoch 5)
```

### Model Export

With `last_save_in_hf=True`, the final checkpoint is exported in HuggingFace format:

```shell
mkdir -p /data/models/Qwen3-30B-A3B-SFT
cp /data/models/Qwen3-30B-A3B/*.json   /data/models/Qwen3-30B-A3B-SFT/
cp /data/models/Qwen3-30B-A3B/tokenizer* /data/models/Qwen3-30B-A3B-SFT/
cp checkpoint_medical/step-156/*.safetensors* /data/models/Qwen3-30B-A3B-SFT/
```

## Evaluation Results

### Evaluation Method

This experiment uses a jieba-based keyword extraction method with POS tagging (n, v, a, i, j, l) to extract keywords from both reference answers and model outputs, then computes:

- **Recall** = matched reference keywords / total reference keywords
- **Precision** = matched reference keywords / total model keywords
- **F1** = harmonic mean of Recall and Precision
- **Think Rate** = proportion of outputs containing `<think>` reasoning

Evaluation data: 241 medical Q&A samples. Base model, CPT intermediate checkpoint, and SFT model are compared under the same conditions.

### Keyword Recall Comparison

| Model | Recall | Precision | F1 |
| :--- | ---: | ---: | ---: |
| Base (Qwen3-30B-A3B) | 53.83% | 25.16% | 33.30% |
| **CPT Checkpoint (step 156)** | **62.45%** | **28.06%** | **37.82%** |
| **Improvement** | **+8.62pp** | **+2.90pp** | **+4.52pp** |

### Output Format Comparison

| Metric | Base | CPT |
| --- | ---: | ---: |
| Avg output length | 1,061 chars | **831 chars** (-21.7%) |
| Format errors (repeated `</think>`) | 199/241 | **9/241** |

Sample: "What are the two components of consciousness?"

| Item | Base Model | CPT Checkpoint |
| --- | --- | --- |
| **Answer** | `</think> The components... arousal... content...` (Markdown list + 3x `</think>`) | `Consciousness consists of two parts: the content and the switch system...` (conversational) |
| **Recall** | 52.4% | **95.2%** |
| **Length** | 392 chars | 287 chars |

### Training Metrics

| Metric | Value |
| --- | --- |
| Stable step time | ~3.2-3.5s |
| Stable memory | ~52.6 GiB/card (85.9%) |
| Loss start (step 1) | 1.45 |
| Loss end (step 156) | 0.045 |
| Total time (156 steps) | ~8-9 minutes |

> **Note**: With CP=2, TP=2 the memory usage per card is ~52.6 GiB (85.9%).

## FAQ

### 1. Loss starts abnormally high

If initial loss is significantly higher than expected (e.g., ~12), check whether HF pretrained weights were loaded correctly. Delete the checkpoint directory before re-running:

```shell
rm -rf checkpoint_medical
```

### 2. NPU out of memory

Check for residual processes occupying NPU memory and ensure `PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"` is set. If necessary, add `torch.npu.set_per_process_memory_fraction(1.0)` at the entry.py entry point.

### 3. HCCL communication timeout

Multi-card training may trigger HCCL watchdog timeout. If intermittent, restarting training usually resolves it. If frequent, check HCCL network configuration and inter-node communication.
