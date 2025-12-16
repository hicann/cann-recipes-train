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
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "${ROOT_DIR}"
CI_DIR="${ROOT_DIR}/ci"

SCAN_LIST=(
    "llm_rl/qwen3"
    # Other paths that needed check...
)

echo -e "${CYAN}=== Scanning projects ===${RESET}"


validate_project() {

    # --- Step 1: Check patch naming ---
    echo -e "${CYAN}=== Step 1: Checking patch naming ===${RESET}"
    if ! bash "${CI_DIR}/check_patch_names.sh"; then
        echo -e "${RED}[ERROR] Patch naming validation failed.${RESET}" >&2
        exit 1
    fi
    echo -e "${GREEN}[OK] Patch names are valid.${RESET}"

    # --- Step 2: Download dependencies ---
    echo -e "${CYAN}=== Step 2: Download dependencies ===${RESET}"
    if ! bash download_deps.sh; then
        echo -e "${RED}[ERROR] Failed to download dependencies.${RESET}" >&2
        exit 1
    fi
    echo -e "${GREEN}[OK] Dependencies downloaded.${RESET}"

    # --- Step 3: Build project ---
    echo -e "${CYAN}=== Step 3: Build project ===${RESET}"
    if ! bash build_project.sh; then
        echo -e "${RED}[ERROR] Project build failed.${RESET}" >&2
        exit 1
    fi
    echo -e "${GREEN}[OK] Project built.${RESET}"

    echo -e "${CYAN}=== Step 4: Apply patches ===${RESET}"
    
    set +e
    PATCH_LOG=$(bash apply_all_patches.sh 2>&1)
    PATCH_STATUS=$?
    set -e

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

for PROJECT in "${SCAN_LIST[@]}"; do
    FULL_PATH="${ROOT_DIR}/${PROJECT}"

    [ -d "$FULL_PATH" ] || continue

    PROJECT_NAME=$(basename "$FULL_PATH")

    echo -e "${CYAN}--- Running CI for project: ${PROJECT_NAME} ---${RESET}"

    for f in download_deps.sh build_project.sh apply_all_patches.sh; do
        if [ ! -f "${FULL_PATH}/${f}" ]; then
            echo -e "${RED}[ERROR] Missing ${f} in project ${PROJECT_NAME}${RESET}"
            exit 1
        fi
    done

    echo -e "${CYAN}Validating project ${PROJECT}${RESET}"
    pushd "${FULL_PATH}" >/dev/null

    if ! validate_project; then
        echo -e "${RED}[ERROR] CI pipeline failed for ${PROJECT_NAME}${RESET}"
        popd >/dev/null
        exit 1
    fi

    echo -e "${GREEN}[OK] Project ${PROJECT_NAME} passed CI.${RESET}"
    popd >/dev/null
done

echo -e "${GREEN}=== All eligible projects passed CI ===${RESET}"
exit 0
