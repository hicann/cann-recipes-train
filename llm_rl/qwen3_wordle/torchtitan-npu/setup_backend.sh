#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

BACKEND_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=${VENV_DIR:-"${BACKEND_DIR}/.venv"}
SOURCES_DIR=${SOURCES_DIR:-"${BACKEND_DIR}/runtime_sources"}
PATCH_DIR="${BACKEND_DIR}/patches"
PYTHON_BIN=${PYTHON_BIN:-python3}
UV_BIN=${UV_BIN:-uv}
CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}
export CMAKE_BUILD_PARALLEL_LEVEL
MAX_JOBS=${MAX_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL}}
export MAX_JOBS
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

# shellcheck disable=SC1091
source "${BACKEND_DIR}/versions.env"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

source_first_existing() {
    local candidate
    for candidate in "$@"; do
        if [[ -f "${candidate}" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "${candidate}"
            set -u
            echo "Loaded environment: ${candidate}"
            return 0
        fi
    done
    return 1
}

source_atb_environment() {
    local candidate
    for candidate in \
        /home/developer/Ascend/nnal/atb/set_env.sh \
        /usr/local/Ascend/nnal/atb/set_env.sh; do
        if [[ -f "${candidate}" ]]; then
            set +u
            # The pinned PyTorch wheel uses the C++11 ABI.  Do not let the
            # host Python environment select a different ATB library variant.
            # shellcheck disable=SC1090
            source "${candidate}" --cxx_abi=1
            set -u
            echo "Loaded environment: ${candidate} (--cxx_abi=1)"
            return 0
        fi
    done
    return 1
}

clone_at_commit() {
    local name=$1
    local url=$2
    local commit=$3
    local destination="${SOURCES_DIR}/${name}"

    if [[ ! -e "${destination}" ]]; then
        mkdir -p "${destination}"
        git -C "${destination}" init --quiet
        git -C "${destination}" remote add origin "${url}"
        git -C "${destination}" fetch --quiet --depth 1 origin "${commit}"
        git -C "${destination}" checkout --quiet --detach FETCH_HEAD
    fi

    if [[ ! -d "${destination}/.git" ]]; then
        echo "${destination} exists but is not a git repository" >&2
        exit 1
    fi

    local actual_commit
    actual_commit=$(git -C "${destination}" rev-parse HEAD)
    if [[ "${actual_commit}" != "${commit}" ]]; then
        echo "${name} is at ${actual_commit}; expected ${commit}." >&2
        echo "Remove only ${destination} and rerun this script." >&2
        exit 1
    fi
}

apply_git_patch_once() {
    local repository=$1
    local patch_file=$2

    if [[ ! -d "${repository}/.git" ]]; then
        echo "Git repository not found: ${repository}" >&2
        exit 1
    fi
    if [[ ! -f "${patch_file}" ]]; then
        echo "Patch not found: ${patch_file}" >&2
        exit 1
    fi

    if git -C "${repository}" apply --check "${patch_file}" >/dev/null 2>&1; then
        git -C "${repository}" apply "${patch_file}"
        echo "Applied $(basename "${patch_file}") to ${repository}"
    elif git -C "${repository}" apply -R --check "${patch_file}" >/dev/null 2>&1; then
        echo "Already applied: $(basename "${patch_file}")"
    else
        echo "Patch is neither applicable nor already applied:" >&2
        echo "  repository: ${repository}" >&2
        echo "  patch: ${patch_file}" >&2
        exit 1
    fi
}

apply_directory_patch_once() {
    local directory=$1
    local patch_file=$2

    if [[ ! -d "${directory}" ]]; then
        echo "Patch target directory not found: ${directory}" >&2
        exit 1
    fi

    if patch --dry-run --silent --batch --forward -p1 -d "${directory}" \
        < "${patch_file}" >/dev/null 2>&1; then
        patch --silent --batch --forward -p1 -d "${directory}" < "${patch_file}"
        echo "Applied $(basename "${patch_file}") to ${directory}"
    elif patch --dry-run --silent --batch --forward -R -p1 -d "${directory}" \
        < "${patch_file}" >/dev/null 2>&1; then
        echo "Already applied: $(basename "${patch_file}")"
    else
        echo "Patch is neither applicable nor already applied:" >&2
        echo "  directory: ${directory}" >&2
        echo "  patch: ${patch_file}" >&2
        exit 1
    fi
}

require_command "${PYTHON_BIN}"
require_command "${UV_BIN}"
require_command gcc
require_command g++
require_command git
require_command patch

actual_uv_version=$("${UV_BIN}" --version | awk '{print $2}')
if [[ "${actual_uv_version}" != "${REQUIRED_UV_VERSION}" ]]; then
    echo "uv ${REQUIRED_UV_VERSION} is required; got ${actual_uv_version}." >&2
    exit 1
fi

python_version=$(
    "${PYTHON_BIN}" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
if [[ "${python_version}" != "${REQUIRED_PYTHON_VERSION}" ]]; then
    echo "Python ${REQUIRED_PYTHON_VERSION} is required; got ${python_version}." >&2
    exit 1
fi

if [[ -z "${UV_DEFAULT_INDEX:-}" ]]; then
    UV_DEFAULT_INDEX=$(
        "${PYTHON_BIN}" -m pip config get global.index-url 2>/dev/null || true
    )
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://repo.huaweicloud.com/repository/pypi/simple}
fi
export UV_DEFAULT_INDEX

gcc_major=$(gcc -dumpfullversion -dumpversion | cut -d. -f1)
if (( gcc_major < REQUIRED_GCC_MAJOR )); then
    echo "GCC ${REQUIRED_GCC_MAJOR}+ is required; got $(gcc -dumpversion)." >&2
    exit 1
fi
export CC=${CC:-gcc}
export CXX=${CXX:-g++}

if ! source_first_existing \
    /home/developer/Ascend/cann-9.0.0/set_env.sh \
    /home/developer/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/cann/set_env.sh; then
    echo "CANN set_env.sh was not found." >&2
    exit 1
fi
if ! source_atb_environment; then
    echo "NNAL/ATB is required. Install the NNAL package matching CANN first." >&2
    exit 1
fi

mkdir -p "${SOURCES_DIR}"
clone_at_commit \
    verl \
    https://gitcode.com/GitHub_Trending/ve/verl.git \
    "${VERL_COMMIT}"
clone_at_commit \
    torchtitan \
    https://gitcode.com/GitHub_Trending/to/torchtitan.git \
    "${TORCHTITAN_COMMIT}"
clone_at_commit \
    torchtitan-npu \
    https://gitcode.com/cann/torchtitan-npu.git \
    "${TORCHTITAN_NPU_COMMIT}"
clone_at_commit \
    vllm \
    https://gitcode.com/gh_mirrors/vl/vllm.git \
    "${VLLM_COMMIT}"
clone_at_commit \
    vllm-ascend \
    https://gitcode.com/gh_mirrors/vl/vllm-ascend.git \
    "${VLLM_ASCEND_COMMIT}"
git -C "${SOURCES_DIR}/vllm-ascend" submodule update --init --recursive

# Apply source patches before building the framework packages.
apply_git_patch_once \
    "${SOURCES_DIR}/verl" \
    "${PATCH_DIR}/verl/0001-verl-feature-torchtitan_npu.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/torchtitan" \
    "${PATCH_DIR}/torchtitan/0001-torchtitan-bugfix-qwen3_npu_init.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/torchtitan" \
    "${PATCH_DIR}/torchtitan/0002-torchtitan-bugfix-qwen3_1_7b_seq_length.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/vllm" \
    "${PATCH_DIR}/vllm/0001-vllm-bugfix-ascend_dependencies.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/vllm-ascend" \
    "${PATCH_DIR}/vllm_ascend/0001-vllm_ascend-bugfix-torch_2_12_build.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/vllm-ascend" \
    "${PATCH_DIR}/vllm_ascend/0002-vllm_ascend-bugfix-build_python_torch_npu.patch"
apply_git_patch_once \
    "${SOURCES_DIR}/vllm-ascend" \
    "${PATCH_DIR}/vllm_ascend/0003-vllm_ascend-bugfix-cann_moe_headers.patch"

if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python3"
export PATH="${VENV_DIR}/bin:${PATH}"

"${UV_BIN}" pip install --python "${PYTHON}" \
    --upgrade pip wheel 'setuptools>=77,<81'
"${UV_BIN}" pip install --python "${PYTHON}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --default-index https://download.pytorch.org/whl/cpu \
    --allow-insecure-host download.pytorch.org \
    --allow-insecure-host download-r2.pytorch.org

# Install vLLM's Python dependencies while retaining the versions required by
# the matching vLLM-Ascend release.
sed \
    -e '/^opencv-python-headless/d' \
    -e '/^fastapi\[/d' \
    -e '/^xgrammar /d' \
    "${SOURCES_DIR}/vllm/requirements/common.txt" \
    | "${UV_BIN}" pip install --python "${PYTHON}" -r /dev/stdin
"${UV_BIN}" pip install --python "${PYTHON}" \
    'opencv-python-headless<=4.11.0.86' \
    'fastapi[standard]>=0.115,<0.124' \
    'xgrammar>=0.1.30' \
    'cmake>=3.26' \
    'setuptools-scm==10.2.1' \
    decorator scipy pandas-stubs msgpack quart numba

"${UV_BIN}" pip install --python "${PYTHON}" \
    -r "${BACKEND_DIR}/requirements.txt"

# triton-ascend overlays files from the generic triton package. Install it
# last, and reinstall it on reruns, so the Ascend backend is not overwritten.
"${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps \
    --reinstall-package triton-ascend \
    "triton-ascend==${TRITON_ASCEND_VERSION}" \
    --default-index https://triton-ascend.osinfra.cn/pypi/simple

VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" VLLM_TARGET_DEVICE=empty \
    "${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps --no-build-isolation -e "${SOURCES_DIR}/vllm"
"${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps -e "${SOURCES_DIR}/verl"
"${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps -e "${SOURCES_DIR}/torchtitan"
"${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps -e "${SOURCES_DIR}/torchtitan-npu"
SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_ASCEND_VERSION}" \
    "${UV_BIN}" pip install --python "${PYTHON}" \
    --no-deps --no-build-isolation -e "${SOURCES_DIR}/vllm-ascend"

torchair_dir=$(
    "${PYTHON}" -c '
import pathlib
import torch_npu

path = pathlib.Path(torch_npu.__file__).resolve().parent / "dynamo" / "torchair"
if not path.is_dir():
    raise SystemExit(f"TorchAir directory not found: {path}")
print(path)
'
)
# TorchAir is bundled with torch-npu and is available only after installation.
apply_directory_patch_once \
    "${torchair_dir}" \
    "${PATCH_DIR}/torchair/0001-torchair-bugfix-hint_int_import.patch"

"${PYTHON}" -c \
    "import nltk; nltk.download('words'); nltk.download('averaged_perceptron_tagger_eng')"
env -u PYTHONPATH "${UV_BIN}" pip check --python "${PYTHON}"
echo "TorchTitan-NPU backend is ready: ${VENV_DIR}"
