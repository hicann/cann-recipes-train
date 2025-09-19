# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import json
import argparse
import logging
from pathlib import Path
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def convert_parquet_to_json(parquet_path, output_json_path):
    """
    Convert a single Parquet file to JSON file
    The key format is: type content _ sequence number (sequence numbers increment for the same type)

    Parameters:
        parquet_path: Path to the input Parquet file
        output_json_path: Path to the output JSON file
    """
    result_dict = {}
    type_counters = {}
    record_count = 0

    df = pd.read_parquet(parquet_path)

    required_columns = {"problem", "level", "type", "solution"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"The file is missing required fields {missing}")
        return

    for _, row in df.iterrows():
        entry_type = row["type"]

        if entry_type not in type_counters:
            type_counters[entry_type] = 1
        key = f"{entry_type}_{type_counters[entry_type]}"

        entry = {
            "problem": row["problem"],
            "level": row["level"],
            "type": entry_type,
            "solution": row["solution"]
        }

        result_dict[key] = entry
        type_counters[entry_type] += 1
        record_count += 1

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    logging.info(
        f"Conversion completed. Processed {record_count} records, "
        f"involving {len(type_counters)} types. Saved to {output_json_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Parquet file to JSON file')
    parser.add_argument('--input',
                        required=True,
                        help='Path to the input Parquet file (e.g.: /path/to/input.parquet)')
    parser.add_argument('--output',
                        required=True,
                        help='Path to the output JSON file (e.g.: /path/to/output.json)')
    args = parser.parse_args()
    convert_parquet_to_json(args.input, args.output)
