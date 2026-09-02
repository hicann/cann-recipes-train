# SPDX-License-Identifier: Apache-2.0
"""CANNBot Qwen3.6-27B CP8 SFT configuration registry."""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
import spmd_types as spmd  # torch must initialize its device backend first
from safetensors import safe_open
from safetensors.torch import save_file
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.trainer import Trainer
from torchtitan_npu.models.qwen3_5.config_registry import qwen35_27b_long_text_sft

from data_process import load_dataset_features, process_sample


# Training configuration

SEQUENCE_LENGTH = 65_536
NO_THINK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] "
    "+ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


class NoThinkTemplateTokenizer(HuggingFaceTokenizer):
    """Use the base tokenizer with an explicit no-thinking chat template."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def __init__(self, config=None, *, tokenizer_path: str) -> None:
        super().__init__(config, tokenizer_path=tokenizer_path)
        self.set_chat_template(NO_THINK_CHAT_TEMPLATE)


class _ActivationOffloadTrainer(Trainer):
    """Offload autograd-saved tensors and reclaim idle NPU blocks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def forward_backward_step(
        self,
        *,
        input_dict,
        labels,
        global_valid_tokens,
    ):
        if self.parallel_dims.pp_enabled:
            raise ValueError("activation offload requires PP=1")
        inputs, labels, kwargs = self.post_dataloading_process(
            input_dict,
            labels,
        )
        assert len(self.model_parts) == 1
        with self.train_context():
            with torch.autograd.graph.save_on_cpu(
                pin_memory=True,
                device_type="npu",
            ):
                prediction = self.model_parts[0](inputs, **kwargs)
            loss, _ = self.loss_fn(prediction, labels, global_valid_tokens)
            del prediction
            torch.npu.synchronize()
            torch.npu.empty_cache()
            with spmd.no_typecheck():
                loss.backward()
        return loss


def _offload_config(base: Trainer.Config) -> _ActivationOffloadTrainer.Config:
    values = {
        item.name: getattr(base, item.name)
        for item in fields(base)
        if item.init
    }
    return _ActivationOffloadTrainer.Config(**values)


def qwen3_6_27b_sft_cp8_64k() -> _ActivationOffloadTrainer.Config:
    """Build the CP8 x FSDP2 Ascend C operator-generation SFT configuration."""

    steps = int(os.environ["CANNBOT_TRAIN_STEPS"])
    checkpoint_interval = int(os.environ["CANNBOT_CHECKPOINT_INTERVAL"])
    config = _offload_config(qwen35_27b_long_text_sft())

    config.tokenizer = NoThinkTemplateTokenizer.Config()
    config.optimizer = default_adamw(
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    config.optimizer.implementation = "fused"
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=max(1, math.ceil(steps * 0.1)),
        decay_ratio=1.0,
        decay_type="cosine",
        min_lr_factor=0.0,
    )
    config.training = replace(
        config.training,
        local_batch_size=1,
        global_batch_size=2,
        seq_len=SEQUENCE_LENGTH,
        steps=steps,
        max_norm=math.inf,
        enable_cpu_offload=False,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )
    config.parallelism = replace(
        config.parallelism,
        context_parallel_degree=8,
        context_parallel_load_balancer=None,
        data_parallel_shard_degree=2,
        data_parallel_replicate_degree=1,
        fsdp_reshard_after_forward="always",
        tensor_parallel_degree=1,
    )
    config.activation_checkpoint = FullAC.Config()
    config.loss.num_chunks = SEQUENCE_LENGTH // 1_024
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={
            "data_files": os.environ["DATA_FILES"].split(","),
            "split": "train",
            "features": load_dataset_features(os.environ["DATA_MANIFEST"]),
        },
        sample_processor=process_sample,
        infinite=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=None,
    )
    config.debug.seed = 42
    config.debug.deterministic = False
    config.metrics.log_freq = 1
    config.comm = replace(
        config.comm,
        init_timeout_seconds=3_600,
        train_timeout_seconds=3_600,
    )
    config.checkpoint = replace(
        config.checkpoint,
        enable=True,
        load_only=os.environ.get("CANNBOT_SMOKE") == "1",
        interval=checkpoint_interval,
        keep_latest_k=3,
        async_mode="disabled",
        last_save_model_only=False,
        last_save_in_hf=False,
    )
    return config


# Hugging Face export

HF_INDEX_NAME = "model.safetensors.index.json"
HF_ASSET_NAMES = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
)


def _load_hf_index(root: Path) -> dict:
    return json.loads((root / HF_INDEX_NAME).read_text())


def _copy_hf_assets(source: Path, output: Path) -> None:
    for name in HF_ASSET_NAMES:
        source_path = source / name
        output_path = output / name
        if source_path.is_file() and not output_path.exists():
            shutil.copy2(source_path, output_path)


def _materialize_converter_placeholders(
    output: Path,
    source: Path,
    output_index: dict,
    source_index: dict,
) -> int:
    materialized = 0
    shards = set(output_index["weight_map"].values())
    for name in sorted(shards):
        shard = output / name
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))

        missing = sorted(
            key
            for key, record in header.items()
            if key != "__metadata__" and not record.get("dtype")
        )
        if any(key not in source_index["weight_map"] for key in missing):
            raise RuntimeError(f"unexpected converter placeholder in {shard}")
        if not missing:
            continue

        for key in missing:
            del header[key]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        if len(encoded) > header_size:
            raise RuntimeError(f"safetensors header grew in {shard}")
        with shard.open("r+b") as stream:
            stream.seek(8)
            stream.write(encoded)
            stream.write(b" " * (header_size - len(encoded)))
            stream.flush()
            os.fsync(stream.fileno())

        with safe_open(shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key in missing:
            source_shard = source / source_index["weight_map"][key]
            with safe_open(source_shard, framework="pt", device="cpu") as handle:
                tensors[key] = handle.get_tensor(key)

        temporary = shard.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, shard)
        materialized += len(missing)
    return materialized


def _validate_hf_checkpoint(
    output: Path,
    source: Path,
    output_index: dict,
    source_index: dict,
) -> tuple[int, int, int]:
    shards = set(output_index["weight_map"].values())
    actual_shards = {path.name for path in output.glob("model-*.safetensors")}
    if len(shards) != 15 or actual_shards != shards:
        raise RuntimeError("expected exactly 15 indexed HF shards")

    expected = set(output_index["weight_map"])
    actual: set[str] = set()
    total_size = 0
    for name in sorted(shards):
        shard = output / name
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual or output_index["weight_map"].get(key) != name:
                    raise RuntimeError(f"invalid shard mapping: {key}")
                tensor = handle.get_tensor(key)
                if handle.get_slice(key).get_dtype() != "BF16":
                    raise RuntimeError(f"non-BF16 tensor {key} in {shard}")
                if not tensor.isfinite().all().item():
                    raise RuntimeError(f"non-finite tensor {key} in {shard}")

                source_shard = source / source_index["weight_map"][key]
                with safe_open(source_shard, framework="pt", device="cpu") as base:
                    source_shape = tuple(base.get_slice(key).get_shape())
                if tuple(tensor.shape) != source_shape:
                    raise RuntimeError(f"shape mismatch: {key}")
                actual.add(key)
                total_size += tensor.numel() * tensor.element_size()

    if actual != expected:
        raise RuntimeError(f"HF tensor/index mismatch: {len(expected)} != {len(actual)}")
    return len(shards), len(actual), total_size


def export_hf_checkpoint(dcp: Path, source: Path, output: Path) -> None:
    """Export a complete, validated BF16 Hugging Face checkpoint."""

    dcp = dcp.resolve()
    source = source.resolve()
    output = output.resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if not (dcp / ".metadata").is_file():
        raise RuntimeError(f"incomplete DCP: {dcp}")
    if not (source / HF_INDEX_NAME).is_file() or output.exists() or staging.exists():
        raise RuntimeError("source must be sharded HF and output must not exist")

    converter = "/workspace/torchtitan/scripts/checkpoint_conversion/convert_to_hf.py"
    subprocess.run(
        [
            sys.executable,
            converter,
            str(dcp),
            str(staging),
            "--hf_assets_path",
            str(source),
            "--model_name",
            "qwen3_5",
            "--model_flavor",
            "27B",
            "--export_dtype",
            "bfloat16",
        ],
        check=True,
    )
    _copy_hf_assets(source, staging)
    output_index = _load_hf_index(staging)
    source_index = _load_hf_index(source)
    materialized = _materialize_converter_placeholders(
        staging,
        source,
        output_index,
        source_index,
    )
    shard_count, tensor_count, total_size = _validate_hf_checkpoint(
        staging,
        source,
        output_index,
        source_index,
    )

    output_index.setdefault("metadata", {})["total_size"] = total_size
    (staging / HF_INDEX_NAME).write_text(json.dumps(output_index, indent=2) + "\n")
    os.replace(staging, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "shards": shard_count,
                "tensors": tensor_count,
                "materialized_tensors": materialized,
                "total_size": total_size,
                "dtype": "bfloat16",
                "finite": True,
            }
        )
    )


# Command-line entry point

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-hf", help="export a DCP checkpoint")
    export.add_argument("dcp", type=Path)
    export.add_argument("source_hf", type=Path)
    export.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export-hf":
        export_hf_checkpoint(args.dcp, args.source_hf, args.output)


if __name__ == "__main__":
    main()
