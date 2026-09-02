# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=quay.io/ascend/cann@sha256:83920900a4b2fdb207d66f39a1c3a24c2ddd00cd638b040eb576632cf4e83c43
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CANN_RELEASE_BASE=https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260722000328244
ARG CANN_TOOLKIT_FILE=Ascend-cann-toolkit_9.2.0~weekly.20260722.01_linux-aarch64.run
ARG CANN_TOOLKIT_SHA256=3e653b6c75372d3b579a41aa68110df96cf2cebd080c9c0561c85a4f4176cebd
ARG CANN_A3_OPS_FILE=Ascend-cann-A3-ops_9.2.0~weekly.20260722.01_linux-aarch64.run
ARG CANN_A3_OPS_SHA256=656d75d825d181e1d1914c0cd974dd209a13124ce32a4e525c66eb45a3c1d53f

RUN set -eux; mkdir -p /tmp/cann; cd /tmp/cann; \
    curl -fL --retry 5 -O "${CANN_RELEASE_BASE}/${CANN_TOOLKIT_FILE}"; \
    echo "${CANN_TOOLKIT_SHA256}  ${CANN_TOOLKIT_FILE}" | sha256sum -c -; \
    chmod +x "${CANN_TOOLKIT_FILE}"; ./${CANN_TOOLKIT_FILE} --upgrade --quiet; \
    curl -fL --retry 5 -O "${CANN_RELEASE_BASE}/${CANN_A3_OPS_FILE}"; \
    echo "${CANN_A3_OPS_SHA256}  ${CANN_A3_OPS_FILE}" | sha256sum -c -; \
    chmod +x "${CANN_A3_OPS_FILE}"; \
    ./${CANN_A3_OPS_FILE} --install --install-for-all --quiet; \
    grep -qx 'Version=9.2.0' /usr/local/Ascend/ascend-toolkit/latest/opp/version.info; \
    grep -qx 'timestamp=20260722_000328244' /usr/local/Ascend/ascend-toolkit/latest/opp/version.info; \
    rm -rf /tmp/cann

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260719%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCH_WHEEL_FILE=torch-2.14.0.dev20260719+cpu-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCH_WHEEL_SHA256=772d52c9ddd7ea0efd2c551b182adf4c05cd7b7886a450d3687be82c025cb104
ARG TORCH_NPU_WHEEL_URL=https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/pta_daily/20260808/torch_npu-2.14.0.dev20260808-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCH_NPU_WHEEL_FILE=torch_npu-2.14.0.dev20260808-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCH_NPU_WHEEL_SHA256=d82122f1bbbce7043b07b12d73e6f12cfaedfaf51fad7dac7c3e1adea090b547
ARG TORCHVISION_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cpu/torchvision-0.29.0.dev20260720%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCHVISION_WHEEL_FILE=torchvision-0.29.0.dev20260720+cpu-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCHVISION_WHEEL_SHA256=8305f8923e8b7563f3dfebfd254826a9b6bb22cf9262a38c4bad793d91d53839
ARG TRITON_ASCEND_WHEEL_URL=https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
ARG TRITON_ASCEND_WHEEL_FILE=triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
ARG TRITON_ASCEND_WHEEL_SHA256=3e7b638ef04c333fd3eeeec35076c6636b708480f01a01d0f5a4c005ccb95521

RUN set -eux; mkdir /tmp/wheels; cd /tmp/wheels; \
    curl -fL --retry 5 -o "${TORCH_WHEEL_FILE}" "${TORCH_WHEEL_URL}"; \
    echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_FILE}" | sha256sum -c -; \
    curl -fL --retry 5 -o "${TORCH_NPU_WHEEL_FILE}" "${TORCH_NPU_WHEEL_URL}"; \
    echo "${TORCH_NPU_WHEEL_SHA256}  ${TORCH_NPU_WHEEL_FILE}" | sha256sum -c -; \
    curl -fL --retry 5 -o "${TORCHVISION_WHEEL_FILE}" "${TORCHVISION_WHEEL_URL}"; \
    echo "${TORCHVISION_WHEEL_SHA256}  ${TORCHVISION_WHEEL_FILE}" | sha256sum -c -; \
    curl -fL --retry 5 -o "${TRITON_ASCEND_WHEEL_FILE}" "${TRITON_ASCEND_WHEEL_URL}"; \
    echo "${TRITON_ASCEND_WHEEL_SHA256}  ${TRITON_ASCEND_WHEEL_FILE}" | sha256sum -c -; \
    python3 -m pip install --no-cache-dir --force-reinstall \
      "./${TORCH_WHEEL_FILE}" "./${TORCH_NPU_WHEEL_FILE}" \
      "./${TORCHVISION_WHEEL_FILE}" \
      numpy==1.26.4 scipy==1.13.1; \
    python3 -m pip install --no-cache-dir \
      pybind11==3.0.4 PyYAML==6.0.3 ninja==1.13.0 safetensors==0.8.0 \
      datasets==4.7.0 transformers==5.13.0 tokenizers==0.22.2 \
      torchdata==0.11.0 tyro==1.0.15 einops==0.8.2 spmd_types==0.2.1 \
      flash-linear-attention==0.5.2 fla-core==0.5.2 \
      tensorboard==2.21.0 wandb==0.28.0 \
      "./${TRITON_ASCEND_WHEEL_FILE}"; \
    rm -rf /tmp/wheels /root/.cache/pip

ARG TORCHTITAN_REPOSITORY=https://gitcode.com/GitHub_Trending/to/torchtitan.git
ARG TORCHTITAN_COMMIT=c91448d20480c7b294314e68976823050002ebec
ARG TORCHTITAN_NPU_REPOSITORY=https://gitcode.com/cann/torchtitan-npu.git
ARG TORCHTITAN_NPU_COMMIT=2afd4c01aa5b1bc9a860a2bd67bdd212b1b1a8f7

RUN set -eux; \
    fetch_commit() { \
      target="$1"; repository="$2"; commit="$3"; \
      git init "${target}"; \
      git -C "${target}" remote add origin "${repository}"; \
      attempt=1; \
      until git -c http.version=HTTP/1.1 -C "${target}" fetch --depth=1 origin "${commit}"; do \
        test "${attempt}" -lt 5; \
        sleep "$((attempt * 2))"; \
        attempt="$((attempt + 1))"; \
      done; \
      git -C "${target}" checkout --detach FETCH_HEAD; \
      test "$(git -C "${target}" rev-parse HEAD)" = "${commit}"; \
    }; \
    fetch_commit /workspace/torchtitan "${TORCHTITAN_REPOSITORY}" "${TORCHTITAN_COMMIT}"; \
    fetch_commit /workspace/torchtitan-npu "${TORCHTITAN_NPU_REPOSITORY}" "${TORCHTITAN_NPU_COMMIT}"

ENV TORCHTITAN_ROOT=/workspace/torchtitan \
    TORCHTITAN_NPU_SRC_DIR=/workspace/torchtitan-npu \
    PYTHONPATH=/opt/cannbot_torchtitan:/workspace/torchtitan-npu:/workspace/torchtitan \
    PYTHONUNBUFFERED=1 \
    HCCL_CONNECT_TIMEOUT=3600 \
    HCCL_EXEC_TIMEOUT=3600 \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,multi_stream_lazy_reclaim:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CPU_AFFINITY_CONF=1 \
    TASK_QUEUE_ENABLE=2 \
    STREAMS_PER_DEVICE=32 \
    TRITON_DISABLE_PRECOMPILE=true

COPY . /opt/cannbot_torchtitan
RUN set -eux; \
    test "$(git -C /workspace/torchtitan rev-parse HEAD)" = "${TORCHTITAN_COMMIT}"; \
    test "$(git -C /workspace/torchtitan-npu rev-parse HEAD)" = "${TORCHTITAN_NPU_COMMIT}"; \
    test -f /workspace/torchtitan-npu/torchtitan_npu/override/qwen3_5/parallelize.py; \
    test -f /workspace/torchtitan-npu/torchtitan_npu/override/qwen3_5/varlen_attention.py; \
    test -f /workspace/torchtitan-npu/torchtitan_npu/override/common/optimizer.py; \
    triton_utils="$(python3 -c 'from pathlib import Path; import triton; print(Path(triton.__file__).resolve().parent / "backends/ascend/utils.py")')"; \
    sed -i 's/"-std=c++17"/"-std=c++20"/g' "${triton_utils}"; \
    grep -q -- '"-std=c++20"' "${triton_utils}"; \
    chmod +x /opt/cannbot_torchtitan/run_train.sh; \
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation -e /workspace/torchtitan; \
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation -e /workspace/torchtitan-npu; \
    python3 -m compileall -q /workspace/torchtitan/torchtitan \
      /workspace/torchtitan-npu/torchtitan_npu /opt/cannbot_torchtitan

WORKDIR /opt/cannbot_torchtitan
CMD ["/bin/bash"]
