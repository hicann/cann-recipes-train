# SPDX-License-Identifier: Apache-2.0
"""JSONL field mapping for the Qwen3.6 SFT sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def process_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Map input/output or prompt/response to a single-turn chat."""

    prompt = sample.get("input")
    response = sample.get("output")
    if prompt is None and response is None:
        prompt = sample.get("prompt")
        response = sample.get("response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input or prompt must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("output or response must be a non-empty string")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def build_manifest(sources: list[Path]) -> dict[str, Any]:
    """Validate source files and build one schema for their cat output."""

    from datasets import Features, load_dataset

    records = 0
    schemas: dict[str, int] = {}
    merged_features: dict[str, Any] = {}
    for source in sources:
        dataset = load_dataset("json", data_files=str(source), split="train")
        columns = set(dataset.column_names)
        if {"input", "output"} <= columns:
            schema = "instruction/input/output"
        elif {"prompt", "response"} <= columns:
            schema = "prompt/response"
        else:
            raise ValueError(f"unsupported fields in {source}")
        for sample in dataset:
            process_sample(sample)
        records += len(dataset)
        schemas[schema] = schemas.get(schema, 0) + len(dataset)
        for field, feature in dataset.features.items():
            if field in merged_features and merged_features[field] != feature:
                raise ValueError(f"incompatible field schema: {field}")
            merged_features[field] = feature

    return {
        "records": records,
        "schemas": schemas,
        "features": Features(merged_features).to_dict(),
        "template": "qwen3_6_nothink",
    }


def load_dataset_features(manifest_path: str):
    """Read the schema required for a cat-combined JSONL file."""

    from datasets import Features

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return Features.from_dict(manifest["features"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SFT JSONL files")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
