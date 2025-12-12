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

python3 check_patch_names.py
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

if [ $PATCH_STATUS -ne 0 ]; then
    echo -e "${RED}[ERROR] apply_all_patches.sh failed.${RESET}"
    exit 1
fi

if echo "$PATCH_LOG" | grep -qi "\[FAIL\]"; then
    echo -e "${RED}[ERROR] A patch failed to apply.${RESET}"
    exit 1
fi

if echo "$PATCH_LOG" | grep -qi "skipped"; then
    echo -e "${RED}[ERROR] A patch was skipped. CI requires full apply.${RESET}"
    exit 1
fi

echo -e "${GREEN}[OK] All patches applied successfully.${RESET}"
echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
