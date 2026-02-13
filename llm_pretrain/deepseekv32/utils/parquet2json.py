# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import json
import pandas as pd
from pathlib import Path
import argparse

def parquet_to_json(input_path: str, output_path: str):
    parquet_files = []
    if os.path.isfile(input_path) and input_path.endswith((".parquet", ".pq")):
        parquet_files = [input_path]
    elif os.path.isdir(input_path):
        for file in Path(input_path).rglob("*.parquet"):
            parquet_files.append(str(file))
        for file in Path(input_path).rglob("*.pq"):
            parquet_files.append(str(file))
    
    all_text_data = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, columns=["text"])
            text_series = df["text"].dropna()
            text_list = text_series.tolist()
            all_text_data.extend(text_list)
        except Exception as e:
            print(f"error: {str(e)} in process {file}, skip")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_text_data:
            text_str = str(text).strip()
            if text_str:
                json.dump({"text": text_str}, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="output.json")
    args = parser.parse_args()
    parquet_to_json(args.input, args.output)