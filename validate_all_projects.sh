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
ROOT_DIR="pwd"
PROJECT_ROOT="${ROOT_DIR}/llm_rl"

# Whitelisted projects to skip
WHITELIST=("qwen2_5" "deepseek")

echo -e "${CYAN}=== CI Wrapper: scanning llm_rl projects ===${RESET}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo -e "${RED}[ERROR] ${PROJECT_ROOT} does not exist${RESET}"
    exit 1
fi

run_project_ci() {
    local PROJECT_NAME="$1"

    echo -e "${CYAN}=== Step 1: Checking patch naming ===${RESET}"
    bash check_patch_names.sh
    echo -e "${GREEN}[OK] Patch names are valid.${RESET}"

    echo -e "${CYAN}=== Step 2: Download dependencies ===${RESET}"
    bash download_deps.sh
    echo -e "${GREEN}[OK] Dependencies downloaded.${RESET}"

    echo -e "${CYAN}=== Step 3: Build project ===${RESET}"
    bash build_project.sh
    echo -e "${GREEN}[OK] Project built.${RESET}"

    echo -e "${CYAN}=== Step 4: Apply patches ===${RESET}"

    PATCH_LOG=$(bash apply_all_patches.sh 2>&1)
    PATCH_STATUS=$?

    echo "$PATCH_LOG"

    # Exit code failure
    if [ $PATCH_STATUS -ne 0 ]; then
        echo -e "${RED}[ERROR] apply_all_patches.sh failed.${RESET}"
        echo ""
        echo -e "${YELLOW}Failed patch details:${RESET}"

        echo "$PATCH_LOG" | grep -i "\[FAIL\]" | while IFS= read -r line; do
            PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
            echo "  $PATCH_NAME"
        done

        exit 1
    fi

    # Log-detected failures
    FAILED_PATCHES=$(echo "$PATCH_LOG" | grep -i "\[FAIL\]" || true)
    if [ -n "$FAILED_PATCHES" ]; then
        echo -e "${RED}[ERROR] A patch failed to apply.${RESET}"
        echo ""
        echo -e "${YELLOW}Failed patch details:${RESET}"

        echo "$FAILED_PATCHES" | while IFS= read -r line; do
            PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
            echo "  $PATCH_NAME"
        done

        exit 1
    fi

    # Skipped patches are not allowed
    SKIPPED_PATCHES=$(echo "$PATCH_LOG" | grep -i "skipped" || true)
    if [ -n "$SKIPPED_PATCHES" ]; then
        echo -e "${RED}[ERROR] One or more patches were skipped. Full application is required.${RESET}"
        echo ""
        echo -e "${YELLOW}Skipped patch details:${RESET}"

        echo "$SKIPPED_PATCHES" | while IFS= read -r line; do
            PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
            echo "  $PATCH_NAME"
        done

        exit 1
    fi

    echo -e "${GREEN}[OK] All patches applied successfully.${RESET}"
    echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
}

for PROJECT in "${PROJECT_ROOT}"/*; do
    [ -d "$PROJECT" ] || continue

    PROJECT_NAME=$(basename "$PROJECT")

    if [[ " ${WHITELIST[@]} " =~ " ${PROJECT_NAME} " ]]; then
        echo -e "${YELLOW}[SKIP] ${PROJECT_NAME} is whitelisted.${RESET}"
        continue
    fi

    echo -e "${CYAN}--- Running CI for project: ${PROJECT_NAME} ---${RESET}"

    for f in download_deps.sh build_project.sh apply_all_patches.sh check_patch_names.sh; do
        if [ ! -f "${PROJECT}/${f}" ]; then
            echo -e "${RED}[ERROR] Missing ${f} in project ${PROJECT_NAME}${RESET}"
            exit 1
        fi
    done

    echo -e "${CYAN}Switching into ${PROJECT}${RESET}"
    pushd "${PROJECT}" >/dev/null

    if ! run_project_ci "${PROJECT_NAME}"; then
        echo -e "${RED}[ERROR] CI pipeline failed for ${PROJECT_NAME}${RESET}"
        popd >/dev/null
        exit 1
    fi

    echo -e "${GREEN}[OK] Project ${PROJECT_NAME} passed CI.${RESET}"
    popd >/dev/null
done

echo -e "${GREEN}=== All eligible projects passed CI ===${RESET}"
exit 0
