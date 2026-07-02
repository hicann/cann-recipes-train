# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import random
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error("line %d: invalid JSON - %s", i, e)
                return None
    return data


def write_split(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            record = {"question": item.get("question", ""), "answer": item.get("answer", "")}
            if "think" in item:
                record["think"] = item["think"]
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./assets/r1_data_example.jsonl")
    parser.add_argument("--output", default="./assets/medical_r1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error("file not found: %s", args.input)
        return 1

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)

    data = load_data(args.input)
    if data is None:
        return 1

    random.shuffle(data)
    split_idx = int(len(data) * args.split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_path = os.path.join(args.output, "train.jsonl")
    test_path = os.path.join(args.output, "test.jsonl")

    write_split(train_path, train_data)
    write_split(test_path, test_data)

    logger.info("[prepare] done:")
    logger.info("  input:  %s (%d records)", args.input, len(data))
    logger.info("  train:  %s (%d records)", train_path, len(train_data))
    logger.info("  test:   %s (%d records)", test_path, len(test_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
