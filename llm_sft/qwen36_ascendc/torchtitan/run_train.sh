#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

EXAMPLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export_hf() {
  (( $# == 3 )) || {
    echo "usage: bash run_train.sh export-hf DCP SOURCE_HF OUTPUT" >&2
    return 2
  }
  python3 "$EXAMPLE_ROOT/cannbot_recipe.py" export-hf "$@"
}

if [[ ${1:-} == export-hf ]]; then
  shift
  export_hf "$@"
  exit
fi

: "${HF_ASSETS_PATH:?Set the pinned Qwen3.6-27B HF directory}"
: "${DUMP_FOLDER:?Set a new output directory}"
(( $# == 0 )) || { echo "This sample accepts only the export-hf subcommand" >&2; exit 2; }
mkdir -p "$DUMP_FOLDER"

IFS=',' read -r -a data_files <<< "${DATA_FILES:?Set comma-separated JSONL paths}"
export DATA_MANIFEST="$DUMP_FOLDER/data_manifest.json"
python3 "$EXAMPLE_ROOT/data_process.py" \
  --data "${data_files[@]}" --output "$DATA_MANIFEST"
export DATA_FILES="$DUMP_FOLDER/train.jsonl"
cat -- "${data_files[@]}" > "$DATA_FILES"

if [[ ${CANNBOT_SMOKE:-0} == 1 ]]; then
  export CANNBOT_TRAIN_STEPS=3
  export CANNBOT_CHECKPOINT_INTERVAL=3
else
  : "${CANNBOT_TRAIN_STEPS:?Set the number of optimizer steps}"
  : "${CANNBOT_CHECKPOINT_INTERVAL:?Set the checkpoint interval in optimizer steps}"
fi

export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}
export PYTHONPATH="$EXAMPLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-$DUMP_FOLDER/pycache}
mkdir -p "$PYTHONPYCACHEPREFIX"

python3 -m py_compile "$EXAMPLE_ROOT"/{cannbot_recipe,data_process}.py

exec torchrun --nnodes 1 --node-rank 0 --nproc-per-node 16 \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" \
  --master-port "${MASTER_PORT:-29851}" \
  --local-ranks-filter "${LOG_RANK:-0}" --role cannbot_qwen36_sft --tee 3 \
  -m torchtitan.train --module cannbot_recipe \
  --config qwen3_6_27b_sft_cp8_64k \
  --hf-assets-path "$HF_ASSETS_PATH" \
  --checkpoint.initial-load-path "$HF_ASSETS_PATH" \
  --checkpoint.initial-load-in-hf --checkpoint.initial-load-model-only \
  --checkpoint.folder checkpoint \
  --dump-folder "$DUMP_FOLDER" \
  --debug.print-config --debug.save-config-file resolved_config.json
