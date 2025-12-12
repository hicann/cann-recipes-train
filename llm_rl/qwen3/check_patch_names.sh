#!/bin/bash
set -e
set -o pipefail

ROOT="patches"
ERRORS=()

# -----------------------------------------
# Validate root-level patches (special rule)
#   NNNN-(bugfix|feature)-description.patch
# -----------------------------------------
check_root_patches() {
    local files=("$@")
    local nums=()

    for f in "${files[@]}"; do
        if [[ ! "$f" =~ ^([0-9]{4})\-(bugfix|feature)\-.+\.patch$ ]]; then
            ERRORS+=("[Root naming error] $ROOT/$f must follow: NNNN-(bugfix|feature)-description.patch")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$ROOT/$f")
    done

    # Sequence check
    IFS=$'\n' sorted=($(sort <<< "${nums[*]}"))
    unset IFS
    for ((i=1; i<${#sorted[@]}; i++)); do
        prev="${sorted[$((i-1))]%%:*}"
        curr="${sorted[$i]%%:*}"
        if (( curr <= prev )); then
            ERRORS+=("[Sequence error] ${sorted[$i]#*:} has a non-increasing sequence number")
        fi
    done
}

# -----------------------------------------
# Validate subdirectory patches
#   NNNN-module-(bugfix|feature)-description.patch
# -----------------------------------------
check_subdir_patches() {
    local subdir="$1"
    shift
    local files=("$@")

    local nums=()

    for f in "${files[@]}"; do
        local full="$subdir/$f"
        if [[ ! "$f" =~ ^([0-9]{4})\-[A-Za-z0-9_]+\-(bugfix|feature)\-.+\.patch$ ]]; then
            ERRORS+=("[Naming error] $full must follow: NNNN-module-(bugfix|feature)-description.patch")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$full")
    done

    # Sequence check
    IFS=$'\n' sorted=($(sort <<< "${nums[*]}"))
    unset IFS
    for ((i=1; i<${#sorted[@]}; i++)); do
        prev="${sorted[$((i-1))]%%:*}"
        curr="${sorted[$i]%%:*}"
        if (( 10#$curr <= 10#$prev )); then
            ERRORS+=("[Sequence error] ${sorted[$i]#*:} has a non-increasing sequence number")
        fi
    done
}

# -----------------------------------------
# Main logic
# -----------------------------------------
if [ ! -d "$ROOT" ]; then
    echo "[ERROR] Directory '$ROOT' does not exist"
    exit 1
fi

# Root-level patches
root_files=()
while IFS= read -r f; do
    root_files+=("$f")
done < <(find "$ROOT" -maxdepth 1 -type f -name "*.patch" -printf "%f\n")

check_root_patches "${root_files[@]}"

# Subdirectories (only level-1)
while IFS= read -r sub; do
    subname=$(basename "$sub")

    mod_files=()
    while IFS= read -r f; do
        mod_files+=("$f")
    done < <(find "$sub" -maxdepth 1 -type f -name "*.patch" -printf "%f\n")

    check_subdir_patches "$sub" "${mod_files[@]}"

done < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d)

# -----------------------------------------
# Final report
# -----------------------------------------
if (( ${#ERRORS[@]} > 0 )); then
    echo "Validation issues found:"
    for e in "${ERRORS[@]}"; do
        echo " - $e"
    done
    exit 1
fi

echo "All patches follow the naming rules."
exit 0
