#!/bin/bash
set -e
set -o pipefail

# --- Colors ---
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

# Resolve root directory
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORKFLOW_DIR="${ROOT_DIR}/.gitcode/workflows"
PROJECT_ROOT="${ROOT_DIR}/llm_rl"

# Whitelisted projects to skip
WHITELIST=("qwen2_5" "deepseek")

echo -e "${CYAN}=== CI Wrapper: scanning llm_rl projects ===${RESET}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo -e "${RED}[ERROR] ${PROJECT_ROOT} does not exist${RESET}"
    exit 1
fi

for PROJECT in "${PROJECT_ROOT}"/*; do
    [ -d "$PROJECT" ] || continue

    PROJECT_NAME=$(basename "$PROJECT")

    # Skip whitelisted projects
    if [[ " ${WHITELIST[@]} " =~ " ${PROJECT_NAME} " ]]; then
        echo -e "${YELLOW}[SKIP] ${PROJECT_NAME} is whitelisted.${RESET}"
        continue
    fi

    echo -e "${CYAN}--- Running CI for project: ${PROJECT_NAME} ---${RESET}"

    # Check that the required project scripts exist
    for f in download_deps.sh build_project.sh apply_all_patches.sh check_patch_names.sh; do
        if [ ! -f "${PROJECT}/${f}" ]; then
            echo -e "${RED}[ERROR] Missing ${f} in project ${PROJECT_NAME}${RESET}"
            exit 1
        fi
    done

    # Switch into the project directory
    echo -e "${CYAN}Switching into ${PROJECT}${RESET}"
    pushd "${PROJECT}" >/dev/null

    # Run shared pipeline script that lives in workflow directory
    bash "${WORKFLOW_DIR}/validate_project.sh"
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] CI pipeline failed for ${PROJECT_NAME}${RESET}"
        popd >/dev/null
        exit 1
    fi

    echo -e "${GREEN}[OK] Project ${PROJECT_NAME} passed CI.${RESET}"

    popd >/dev/null
done

echo -e "${GREEN}=== All eligible projects passed CI ===${RESET}"
exit 0
