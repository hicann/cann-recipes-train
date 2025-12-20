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
ROOT_DIR=$(pwd)
CI_DIR="$(cd ci && pwd)"

SCAN_LIST=(
    "llm_rl/qwen3"
    # Other paths that needed check...
)

echo -e "${CYAN}=== CI Starts ===${RESET}"

apply_patches_without_git() {
    local PROJECT_DIR="$1"

    # Apply the patches from the root dir, where .git is supposed to be.
    cd "${ROOT_DIR}"
    PATCH_DIR="${ROOT_DIR}/patches"

    echo "Applying patches in numerical order..."

    find "${PATCH_DIR}" -type f -name "*.patch" | \
    sort -V | \

    ## apply patch
    while IFS= read -r PATCH_FILE; do
        # skip empty lines
        [[ -z "$PATCH_FILE" ]] && continue
        PATCH_REL_PATH=$(realpath --relative-to=PROJECT_PATH "$PATCH_FILE")
        
        echo -n "Applying $PATCH_REL_PATH ... "

        git apply -v --ignore-whitespace "$PATCH_REL_PATH"

        if [ $? -ne 0 ]; then
            echo "[FAIL]: $PATCH_REL_PATH" >&2
            exit 1
        fi
        echo "[SUCCESS]: $PATCH_REL_PATH"
    done

}

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
    ls -l
    echo -e "${GREEN}[OK] Project built.${RESET}"

    echo -e "${CYAN}=== Step 4: Apply patches ===${RESET}"
    
    set +e
    # Assume CI has no git availability.
    PATCH_LOG=$(apply_patches_without_git "${PROJECT}"  2>&1)
    PATCH_STATUS=$?
    set -e

    echo "$PATCH_LOG"

    # Patch application failed. Some patch failed during application.
    FAILED_PATCHES=$(echo "$PATCH_LOG" | grep -i "\[FAIL\]" || true)
    # Note: when applying a patch with wrong path WITH GIT AVAILABLE (the recommended way),
    # it will be silently skipped; However applying it without git (common in CI)
    # will raise an error, that's why there is no need to check for silent skips.

    if [ $PATCH_STATUS -ne 0 ] || [ -n "$FAILED_PATCHES" ]; then

        echo -e "${RED}[ERROR] Patch application failed.${RESET}"
        if [ -n "$FAILED_PATCHES" ]; then
            echo -e "${YELLOW}Failed patches:${RESET}"

            echo "$FAILED_PATCHES" | while IFS= read -r line; do
                PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
                echo "  $PATCH_NAME"
            done
            echo ""
        fi

        exit 1
    fi

    echo -e "${GREEN}[OK] All patches applied successfully.${RESET}"
    echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
}

for PROJECT in "${SCAN_LIST[@]}"; do
    FULL_PATH="${ROOT_DIR}/${PROJECT}"

    if [ ! -d "$FULL_PATH" ]; then
        echo -e "${YELLOW}[Warning] Project directory not found: ${FULL_PATH}${RESET}"
        continue
    fi

    PROJECT_BASENAME=$(basename "$FULL_PATH")

    echo -e "${CYAN}--- Running CI for project: ${PROJECT} ---${RESET}"

    for f in download_deps.sh build_project.sh apply_all_patches.sh; do
        if [ ! -f "${FULL_PATH}/${f}" ]; then
            echo -e "${RED}[ERROR] Missing ${f} in project ${PROJECT_BASENAME}${RESET}"
            exit 1
        fi
    done

    echo -e "${CYAN}Validating project ${PROJECT}${RESET}"
    pushd "${FULL_PATH}" >/dev/null

    if ! validate_project; then
        echo -e "${RED}[ERROR] CI pipeline failed for ${PROJECT_BASENAME}${RESET}"
        popd >/dev/null
        exit 1
    fi

    echo -e "${GREEN}[OK] Project ${PROJECT_BASENAME} passed CI.${RESET}"
    popd >/dev/null
done

echo -e "${GREEN}=== All projects passed CI ===${RESET}"
exit 0
