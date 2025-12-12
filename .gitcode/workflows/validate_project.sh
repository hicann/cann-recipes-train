#!/bin/bash
set -e
set -o pipefail

# --- Colors ---
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "${CYAN}=== Step 1: Checking patch naming ===${RESET}"

bash check_patch_names.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Patch naming validation failed.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Patch names are valid.${RESET}"


echo -e "${CYAN}=== Step 2: Download dependencies ===${RESET}"

bash download_deps.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] download_deps.sh failed.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Dependencies downloaded.${RESET}"


echo -e "${CYAN}=== Step 3: Build project ===${RESET}"

bash build_project.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] build_project.sh failed.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Project built.${RESET}"


echo -e "${CYAN}=== Step 4: Apply patches ===${RESET}"

PATCH_LOG=$(bash apply_all_patches.sh 2>&1)
PATCH_STATUS=$?

echo "$PATCH_LOG"

# Check exit code first
if [ $PATCH_STATUS -ne 0 ]; then
    echo -e "${RED}[ERROR] apply_all_patches.sh failed.${RESET}"
    echo ""

    echo -e "${YELLOW}Failed patch details (from exit code):${RESET}"

    # Try to extract failed patch lines from the log
    echo "$PATCH_LOG" | grep -i "\[FAIL\]" | while IFS= read -r line; do
        PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
        echo "  $PATCH_NAME"
    done

    echo ""
    exit 1
fi

# Detect failures in log output even if exit code is 0.
FAILED_PATCHES=$(echo "$PATCH_LOG" | grep -i "\[FAIL\]" || true)

if [ -n "$FAILED_PATCHES" ]; then
    echo -e "${RED}[ERROR] A patch failed to apply.${RESET}"
    echo ""

    echo -e "${YELLOW}Failed patch details:${RESET}"

    echo "$FAILED_PATCHES" | while IFS= read -r line; do
        # Try to extract the actual patch filename
        # Supports formats like:
        #   "Skipping patch patches/vllm/0003-example.patch"
        #   "patches/vllm/0003-example.patch: skipped"
        PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
        echo "  $PATCH_NAME"
    done

    echo ""
    exit 1
fi

# Detect skipped patches
SKIPPED_PATCHES=$(echo "$PATCH_LOG" | grep -i "skipped" || true)

if [ -n "$SKIPPED_PATCHES" ]; then
    echo -e "${RED}[ERROR] One or more patches were skipped. Full application is required.${RESET}"
    echo ""

    echo -e "${YELLOW}Skipped patch details:${RESET}"
    echo "$SKIPPED_PATCHES" | while IFS= read -r line; do
        PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')

        echo "  $PATCH_NAME"
    done

    echo ""
    exit 1
fi


echo -e "${GREEN}[OK] All patches applied successfully.${RESET}"
echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
