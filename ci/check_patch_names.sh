#!/bin/bash 
set -e
set -o pipefail

ROOT="patches"
ERRORS=()

# -----------------------------------------
# Validate root-level patches
#   NNNN-(bugfix|feature)-description.patch
#   description must use underscores, not dashes
# -----------------------------------------
check_root_patches() {
    local files=("$@")
    local nums=()

    for f in "${files[@]}"; do
        if [[ ! "$f" =~ ^([0-9]{4})\-(bugfix|feature)\-[A-Za-z0-9_]+\.patch$ ]]; then
            ERRORS+=("[Naming error] $ROOT/$f must follow: NNNN-(bugfix|feature)-description.patch (description must use underscores)")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$ROOT/$f")
    done

    # Sequence check: patch numbers must strictly increase by 1 (0001, 0002, 0003, ...)
    if [[ ${#nums[@]} -gt 0 ]]; then
        IFS=$'\n'
        sorted=($(sort <<< "${nums[*]}"))
        unset IFS

        # Optional: ensure first patch starts at 0001
        first_seq="${sorted[0]%%:*}"
        if (( 10#$first_seq != 1 )); then
            ERRORS+=("[Number error] First patch must be 0001, but got $first_seq in ${sorted[0]#*:}")
        fi

        # Check continuity: each should be previous + 1
        for ((i=1; i < ${#sorted[@]}; i++)); do
            prev_seq="${sorted[$((i-1))]%%:*}"
            curr_seq="${sorted[$i]%%:*}"
            expected=$(( 10#$prev_seq + 1 ))
            actual=10#$curr_seq

            if (( actual != expected )); then
                ERRORS+=("[Number error] Expected sequence $expected, but got $curr_seq in ${sorted[$i]#*:}")
            fi
        done
    fi
}

# -----------------------------------------
# Validate subdirectory patches
#   NNNN-module-(bugfix|feature)-description.patch
#   description must use underscores, not dashes
# -----------------------------------------
check_subdir_patches() {
    local subdir="$1"
    shift
    local files=("$@")

    local nums=()

    for f in "${files[@]}"; do
        local full="$subdir/$f"

        if [[ ! "$f" =~ ^([0-9]{4})\-([A-Za-z0-9_]+)\-(bugfix|feature)\-([A-Za-z0-9_]+)\.patch$ ]]; then
            ERRORS+=("[Naming error] $full must follow: NNNN-module-(bugfix|feature)-description.patch (description must use underscores)")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$full")
        seq="${BASH_REMATCH[1]}"
        module="${BASH_REMATCH[2]}"
        expected_module="$(basename "$subdir")"
    
        if [[ "$module" != "$expected_module" ]]; then
            ERRORS+=("[Module mismatch] $full: module '$module' does not match directory '$expected_module'")
            continue
        fi

    done

    # Sequence check: patch numbers must strictly increase by 1 (0001, 0002, 0003, ...)
    if [[ ${#nums[@]} -gt 0 ]]; then
        IFS=$'\n'
        sorted=($(sort <<< "${nums[*]}"))
        unset IFS

        # Optional: ensure first patch starts at 0001
        first_seq="${sorted[0]%%:*}"
        if (( 10#$first_seq != 1 )); then
            ERRORS+=("[Number error] First patch must be 0001, but got $first_seq in ${sorted[0]#*:}")
        fi

        # Check continuity: each should be previous + 1
        for ((i=1; i < ${#sorted[@]}; i++)); do
            prev_seq="${sorted[$((i-1))]%%:*}"
            curr_seq="${sorted[$i]%%:*}"
            expected=$(( 10#$prev_seq + 1 ))
            actual=10#$curr_seq

            if (( actual != expected )); then
                ERRORS+=("[Number error] Expected sequence $expected, but got $curr_seq in ${sorted[$i]#*:}")
            fi
        done
    fi
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
