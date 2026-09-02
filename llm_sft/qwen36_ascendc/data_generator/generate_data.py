#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
"""Generate the fixed prompt/output/JSONL dataset.

The fixture uses ``<op>/desc`` plus either a single ``<op>/src`` directory
or one or more ``<op>/src_N`` directories.  ``desc`` is shared by every
implementation, while each source directory produces one independent
response record.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


REQUIRED_DESC_FILES = ("cases.yaml", "desc.md", "golden.py", "proto.yaml")
OWNED_OUTPUT_DIRS = ("inputs", "outputs", "jsonl", "manifests", "intermediate")
PROMPT_KIND = "md-code-block-oneshot"
OUTPUT_KIND = "md-code-block"
JSONL_FILE = f"{PROMPT_KIND}.jsonl"
COPYRIGHT_MARKER_RE = re.compile(
    r"(?:copyright|\u00a9|\u7248\u6743\u6240\u6709|spdx[\s-]*license-identifier|"
    r"cann\s+open\s+software\s+license|all\s+rights\s+reserved)",
    re.IGNORECASE,
)
COPYRIGHT_COMMENT_BLOCK_RES = (
    re.compile(
        r"^[ \t]*/\*.*?\*/[ \t]*(?:\r?\n[ \t]*(?:\r?\n)?)?",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"^[ \t]*<!--.*?-->[ \t]*(?:\r?\n[ \t]*(?:\r?\n)?)?",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"^(?:[ \t]*(?:\#|//)[^\r\n]*(?:\r?\n|(?=\Z)))+"
        r"(?:[ \t]*\r?\n)?",
        re.MULTILINE,
    ),
)
SRC_DIR_RE = re.compile(r"^src(?:_([1-9][0-9]*))?$")


@dataclass(frozen=True)
class SampleEntry:
    sample_name: str
    source_sample: str
    op_name: str
    implementation_index: int
    implementation_count: int
    op_dir: Path
    desc_dir: Path
    src_dir: Path
    src_name: str
    cases_yaml: Path
    desc_md: Path
    golden_py: Path
    proto_yaml: Path
    kernel_src: Path
    launch_h_src: Path
    plugin_src: Path
    cmake_src: Path
    thinking_md: Path | None
    has_thinking: bool


@dataclass(frozen=True)
class SourceDirectory:
    path: Path
    priority: int
    sample_map: dict[tuple[str, str], str] = field(default_factory=dict)
    manifest_path: Path | None = None


@dataclass(frozen=True)
class OperatorDirectory:
    op_name: str
    op_dir: Path
    source_dir: SourceDirectory


@dataclass(frozen=True)
class SampleFilterDecision:
    entry: SampleEntry
    op_name: str
    normalized_op: str
    op_source: str
    removed: bool


@dataclass(frozen=True)
class ManifestContext:
    entries: list[SampleEntry]
    input_paths: dict[str, Path]
    output_paths: dict[str, Path]
    jsonl_path: Path
    output_dir: Path
    repo_root: Path
    source_root: Path
    source_dirs: list[SourceDirectory]
    selected_operator_dirs: list[OperatorDirectory]
    operator_candidates: list[OperatorDirectory]
    skipped: list[str]
    with_thinking: bool
    requested_filter_ops: list[str] | None
    filter_ops: set[str]
    filter_decisions: list[SampleFilterDecision]


@dataclass(frozen=True)
class GenerationPlan:
    args: argparse.Namespace
    repo_root: Path
    source_root: Path
    output_dir: Path
    examples_root: Path | None
    generator: object
    source_dirs: list[SourceDirectory]
    selected: list[OperatorDirectory]
    candidates: list[OperatorDirectory]
    entries: list[SampleEntry]
    skipped: list[str]
    filter_ops: set[str]
    filter_decisions: list[SampleFilterDecision]
    removed: list[SampleFilterDecision]
    missing_filter_ops: list[str]
    source_priority: str
    shadowed: int


def repo_relative(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = repo_root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_copyright_headers(text: str) -> str:
    """Remove recognized copyright/license comment blocks."""

    def remove_if_copyright(match: re.Match[str]) -> str:
        block = match.group(0)
        return "" if COPYRIGHT_MARKER_RE.search(block) else block

    for block_re in COPYRIGHT_COMMENT_BLOCK_RES:
        text = block_re.sub(remove_if_copyright, text)
    return text


def read_dataset_text(path: Path) -> str:
    return strip_copyright_headers(read_text(path))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def canonical_op_name(name: str) -> str:
    """Normalize an operator name in the same way as the augmenter/filter."""
    text = name.strip()
    text = re.sub(r"\bcann_bench\.", "", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_filter_ops(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    normalized: set[str] = set()
    for value in values:
        name = canonical_op_name(value)
        if name:
            normalized.add(name)
    if not normalized:
        raise ValueError("--filter-ops requires at least one valid operator name")
    return normalized


def source_op_matches_filter(normalized_op: str, filter_op: str) -> bool:
    if normalized_op == filter_op:
        return True
    prefix = f"{filter_op}_"
    return normalized_op.startswith(prefix) and normalized_op[len(prefix):].isdigit()


def matched_filter_op_names(
    decisions: Iterable[SampleFilterDecision], filter_ops: set[str]
) -> set[str]:
    matched: set[str] = set()
    for decision in decisions:
        if not decision.removed:
            continue
        for filter_op in filter_ops:
            if source_op_matches_filter(decision.normalized_op, filter_op):
                matched.add(filter_op)
    return matched


def load_prompt_generator(repo_root: Path):
    """Load prompt generator."""
    sys.dont_write_bytecode = True
    module_path = (
        Path(__file__).resolve().parents[1]
        / "prompt_generator"
        / "generate_prompts.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"prompt generator not found: {module_path}")
    spec = importlib.util.spec_from_file_location("prompt_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt generator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _looks_like_collection(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((child / "desc").is_dir() for child in path.iterdir() if child.is_dir())


def discover_default_source_dir(source_root: Path) -> Path:
    """Resolve a source root containing operator directories."""
    if _looks_like_collection(source_root):
        return source_root
    raise FileNotFoundError(
        f"operator data directory not found: {source_root}"
    )


def normalize_source_dir_values(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        normalized.extend(part.strip() for part in value.split(",") if part.strip())
    if not normalized:
        raise ValueError("--source-dirs requires at least one non-empty directory")
    return normalized


def _load_sample_manifest(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_sample", "destination_operator", "destination_src"}
        if not required.issubset(reader.fieldnames or ()):
            missing = ", ".join(sorted(required - set(reader.fieldnames or ())))
            raise ValueError(f"manifest {path} is missing column(s): {missing}")
        result: dict[tuple[str, str], str] = {}
        for row in reader:
            operator = (row.get("destination_operator") or "").strip()
            src_name = (row.get("destination_src") or "").strip()
            sample = (row.get("source_sample") or "").strip()
            if not operator or not src_name or not sample:
                raise ValueError(f"manifest {path} contains an incomplete row")
            if "/" in sample or "\\" in sample:
                raise ValueError(
                    f"manifest {path} contains an invalid source_sample: {sample}"
                )
            key = (operator, src_name)
            if key in result and result[key] != sample:
                raise ValueError(
                    f"manifest {path} contains duplicate mapping for {operator}/{src_name}"
                )
            result[key] = sample
        return result


def resolve_source_directories(
    source_root: Path, source_dir_values: Iterable[str] | None
) -> list[SourceDirectory]:
    if source_dir_values is None:
        paths = [discover_default_source_dir(source_root).resolve()]
    else:
        names = normalize_source_dir_values(source_dir_values)
        paths = [resolve_path(Path(name), source_root).resolve() for name in names]

    result: list[SourceDirectory] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_dir():
            raise FileNotFoundError(f"source directory not found: {path}")
        manifest_path = path / "manifest.csv"
        result.append(
            SourceDirectory(
                path=path,
                priority=len(result) + 1,
                sample_map=_load_sample_manifest(manifest_path),
                manifest_path=manifest_path if manifest_path.is_file() else None,
            )
        )
    return result


def discover_operator_directories(
    source_dirs: list[SourceDirectory], *, only_ops: set[str] | None = None
) -> tuple[list[OperatorDirectory], list[OperatorDirectory]]:
    selected_by_name: dict[str, OperatorDirectory] = {}
    candidates: list[OperatorDirectory] = []
    for source_dir in source_dirs:
        for op_dir in sorted(source_dir.path.iterdir()):
            if not op_dir.is_dir() or op_dir.name == "desc":
                continue
            if only_ops and op_dir.name not in only_ops:
                continue
            candidate = OperatorDirectory(op_dir.name, op_dir, source_dir)
            candidates.append(candidate)
            selected_by_name.setdefault(candidate.op_name, candidate)
    return sorted(selected_by_name.values(), key=lambda item: item.op_name), candidates


def choose_required_file(
    root: Path, pattern: str, preferred_name: str, label: str
) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"missing {label} directory: {root}")
    preferred = root / preferred_name
    if preferred.is_file():
        return preferred
    candidates = sorted(root.glob(pattern))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"missing {label} file matching {pattern}: {root}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"ambiguous {label} files under {root}: {names}")


def validate_desc(desc_dir: Path) -> dict[str, Path]:
    paths = {name: desc_dir / name for name in REQUIRED_DESC_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prompt input file(s):\n" + "\n".join(missing))
    return paths


def _src_index(path: Path) -> int:
    match = SRC_DIR_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"source directory must be named src or src_N: {path}")
    # ``src`` is the canonical name used by collections with one
    # implementation.  Treat it as implementation 1 for the common metadata
    # fields while keeping the sample name unnumbered when it is the only
    # implementation.
    return int(match.group(1) or 1)


def build_entry(
    op_dir: Path,
    src_dir: Path,
    *,
    source_sample: str | None = None,
    implementation_index: int | None = None,
    implementation_count: int | None = None,
) -> SampleEntry:
    op_name = op_dir.name
    index = implementation_index if implementation_index is not None else _src_index(src_dir)
    count = implementation_count if implementation_count is not None else 1
    desc_dir = op_dir / "desc"
    if not desc_dir.is_dir():
        raise FileNotFoundError(f"missing desc directory: {desc_dir}")
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing src directory: {src_dir}")
    desc_paths = validate_desc(desc_dir)
    kernel_src = choose_required_file(
        src_dir / "op_kernel", "*_kernel.cpp", f"{op_name}_kernel.cpp", "kernel source"
    )
    launch_h_src = choose_required_file(
        src_dir / "op_kernel", "*_launch.h", f"{op_name}_launch.h", "launch header"
    )
    plugin_src = choose_required_file(
        src_dir / "op_plugin", "*_plugin.cpp", f"{op_name}_plugin.cpp", "plugin source"
    )
    cmake_src = src_dir / "CMakeLists.txt"
    if not cmake_src.is_file():
        raise FileNotFoundError(f"missing cmake source: {cmake_src}")

    sample = source_sample or (op_name if count == 1 else f"{op_name}_{index}")
    if not sample or "/" in sample or "\\" in sample:
        raise ValueError(f"invalid sample name: {sample!r}")
    thinking_md = src_dir / "thinking.md"
    has_thinking = thinking_md.is_file() and bool(read_text(thinking_md).strip())
    return SampleEntry(
        sample_name=sample,
        source_sample=sample,
        op_name=op_name,
        implementation_index=index,
        implementation_count=count,
        op_dir=op_dir,
        desc_dir=desc_dir,
        src_dir=src_dir,
        src_name=src_dir.name,
        cases_yaml=desc_paths["cases.yaml"],
        desc_md=desc_paths["desc.md"],
        golden_py=desc_paths["golden.py"],
        proto_yaml=desc_paths["proto.yaml"],
        kernel_src=kernel_src,
        launch_h_src=launch_h_src,
        plugin_src=plugin_src,
        cmake_src=cmake_src,
        thinking_md=thinking_md if thinking_md.is_file() else None,
        has_thinking=has_thinking,
    )


def _find_source_dirs(op_dir: Path) -> tuple[list[Path], list[str]]:
    source_like_dirs = [
        path
        for path in sorted(op_dir.iterdir())
        if path.is_dir() and (path.name == "src" or path.name.startswith("src_"))
    ]
    errors = [
        f"{path}: source directory must be named src or src_N"
        for path in source_like_dirs
        if not SRC_DIR_RE.fullmatch(path.name)
    ]
    src_dirs = [path for path in source_like_dirs if SRC_DIR_RE.fullmatch(path.name)]
    return sorted(src_dirs, key=_src_index), errors


def _discover_operator_entries(
    operator: OperatorDirectory, *, skip_invalid: bool
) -> tuple[list[SampleEntry], list[str]]:
    op_dir = operator.op_dir
    src_dirs, errors = _find_source_dirs(op_dir)
    try:
        validate_desc(op_dir / "desc")
    except (FileNotFoundError, ValueError) as exc:
        message = f"{op_dir}: {exc}"
        if skip_invalid:
            return [], errors + [message]
        raise type(exc)(message) from None

    if not src_dirs:
        return [], errors + [f"{op_dir}: no src or src_N directories"]
    if len(src_dirs) > 1 and any(path.name == "src" for path in src_dirs):
        message = f"{op_dir}: cannot mix src and src_N directories"
        if skip_invalid:
            return [], errors + [message]
        raise ValueError(message)

    entries: list[SampleEntry] = []
    count = len(src_dirs)
    for src_dir in src_dirs:
        try:
            entries.append(
                build_entry(
                    op_dir,
                    src_dir,
                    source_sample=operator.source_dir.sample_map.get((op_dir.name, src_dir.name)),
                    implementation_index=_src_index(src_dir),
                    implementation_count=count,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            message = f"{src_dir}: {exc}"
            if skip_invalid:
                errors.append(message)
            else:
                raise type(exc)(message) from None
    return entries, errors


def discover_selected_entries(
    operator_dirs: list[OperatorDirectory], *, skip_invalid: bool = False
) -> tuple[list[SampleEntry], list[str]]:
    entries: list[SampleEntry] = []
    errors: list[str] = []
    for operator in operator_dirs:
        found, operator_errors = _discover_operator_entries(
            operator, skip_invalid=skip_invalid
        )
        entries.extend(found)
        errors.extend(operator_errors)
    if errors and not skip_invalid:
        raise ValueError("invalid source layout:\n" + "\n".join(errors))
    if not entries:
        location = ", ".join(str(item.op_dir) for item in operator_dirs)
        raise ValueError(
            f"no valid entries found under {location or 'selected source directories'}"
        )
    names = [entry.sample_name for entry in entries]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError("duplicate sample names: " + ", ".join(duplicates))
    return entries, errors


def discover_entries(
    source_root: Path | None = None,
    *,
    ops_root: Path | None = None,
    only_ops: set[str] | None = None,
    skip_invalid: bool = False,
) -> tuple[list[SampleEntry], list[str]]:
    """Discover entries from one source-data collection.

    ``ops_root`` is accepted as a compatibility alias for callers modeled on
    the ops augmenter API.
    """
    root = source_root if source_root is not None else ops_root
    if root is None:
        raise TypeError("discover_entries requires a source root")
    source_dirs = resolve_source_directories(root, None)
    selected, _ = discover_operator_directories(source_dirs, only_ops=only_ops)
    return discover_selected_entries(selected, skip_invalid=skip_invalid)


def filter_entries(
    entries: list[SampleEntry], filter_ops: set[str]
) -> tuple[list[SampleEntry], list[SampleFilterDecision]]:
    if not filter_ops:
        return entries, []
    kept: list[SampleEntry] = []
    decisions: list[SampleFilterDecision] = []
    for entry in entries:
        normalized = canonical_op_name(entry.op_name)
        removed = any(source_op_matches_filter(normalized, item) for item in filter_ops)
        decisions.append(
            SampleFilterDecision(entry, entry.op_name, normalized, "operator_directory", removed)
        )
        if not removed:
            kept.append(entry)
    return kept, decisions


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_DIRS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def has_owned_output_artifacts(output_dir: Path) -> bool:
    return any((output_dir / name).exists() for name in OWNED_OUTPUT_DIRS)


def load_prompt_inputs(generator, entry: SampleEntry):
    return generator.OpPromptInputs(
        name=entry.sample_name,
        op_dir=entry.desc_dir,
        cases_yaml=read_dataset_text(entry.cases_yaml),
        desc_md=read_dataset_text(entry.desc_md),
        golden_py=read_dataset_text(entry.golden_py),
        proto_yaml=read_dataset_text(entry.proto_yaml),
    )


def render_inputs(
    generator,
    entries: list[SampleEntry],
    output_dir: Path,
    *,
    example: str = "sqrt",
    one_shot_example: str | None = None,
    examples_root: Path | None = None,
) -> dict[str, Path]:
    if one_shot_example is not None:
        example = one_shot_example
    result: dict[str, Path] = {}
    input_dir = output_dir / "inputs" / PROMPT_KIND
    input_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        prompt = generator.render_prompt(
            load_prompt_inputs(generator, entry),
            example=example,
            examples_root=examples_root,
        )
        path = input_dir / f"{entry.sample_name}.md"
        write_text(path, strip_copyright_headers(prompt))
        result[entry.sample_name] = path
    return result


def extract_thinking(entry: SampleEntry) -> str:
    if entry.thinking_md is None or not entry.has_thinking:
        return ""
    text = read_dataset_text(entry.thinking_md).strip()
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    first_step = re.compile(r"^\s*##\s*第\s*1\s*步")
    start = next((index for index, line in enumerate(lines) if first_step.match(line)), None)
    if start is None:
        raise ValueError(f"thinking file has no '## 第 1 步' section: {entry.thinking_md}")
    return f"<think>\n{''.join(lines[start:]).strip()}\n</think>\n\n"


def load_output_parts(entry: SampleEntry) -> dict[str, str]:
    return {
        "kernel_src": read_dataset_text(entry.kernel_src).rstrip("\n"),
        "launch_h_src": read_dataset_text(entry.launch_h_src).rstrip("\n"),
        "plugin_src": read_dataset_text(entry.plugin_src).rstrip("\n"),
        "cmake_src": read_dataset_text(entry.cmake_src).rstrip("\n"),
    }


def format_md_output(entry: SampleEntry, *, with_thinking: bool = False) -> str:
    parts = load_output_parts(entry)
    for label, source in parts.items():
        if "```" in source:
            raise ValueError(f"{entry.sample_name} {label} contains markdown fence")
    output = (
        (extract_thinking(entry) if with_thinking else "")
        + "kernel_src\n```cpp\n"
        + parts["kernel_src"]
        + "\n```\n\nlaunch_h_src\n```cpp\n"
        + parts["launch_h_src"]
        + "\n```\n\nplugin_src\n```cpp\n"
        + parts["plugin_src"]
        + "\n```\n\ncmake_src\n```cmake\n"
        + parts["cmake_src"]
        + "\n```\n"
    )
    return strip_copyright_headers(output)


def render_outputs(
    entries: list[SampleEntry], output_dir: Path, *, with_thinking: bool = False
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    output_dir = output_dir / "outputs" / OUTPUT_KIND
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = output_dir / f"{entry.sample_name}.md"
        write_text(path, format_md_output(entry, with_thinking=with_thinking))
        result[entry.sample_name] = path
    return result


def write_jsonl(
    entries: list[SampleEntry],
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    output_dir: Path,
) -> Path:
    path = output_dir / "jsonl" / JSONL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            record = {
                "instruction": "",
                "input": read_text(input_paths[entry.sample_name]),
                "output": read_text(output_paths[entry.sample_name]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_intermediate_records(
    entries: list[SampleEntry], output_dir: Path, repo_root: Path
) -> None:
    for entry in entries:
        record = {
            "sample": entry.sample_name,
            "source_sample": entry.source_sample,
            "op": entry.op_name,
            "implementation_index": entry.implementation_index,
            "implementation_count": entry.implementation_count,
            "desc": repo_relative(entry.desc_dir, repo_root),
            "src": repo_relative(entry.src_dir, repo_root),
        }
        write_text(
            output_dir / "intermediate" / entry.sample_name / "source.json",
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        )


def _write_entry_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    selected = {item.op_name: item for item in context.selected_operator_dirs}
    fields = [
        "sample", "source_sample", "op", "implementation_index", "implementation_count",
        "source_dir", "source_priority", "op_dir", "desc_dir", "src_dir", "src_name",
        "has_thinking", "thinking_md", "kernel_src", "launch_h_src", "plugin_src", "cmake_src",
    ]
    rows = (
        {
            "sample": entry.sample_name,
            "source_sample": entry.source_sample,
            "op": entry.op_name,
            "implementation_index": entry.implementation_index,
            "implementation_count": entry.implementation_count,
            "source_dir": rel(selected[entry.op_name].source_dir.path),
            "source_priority": selected[entry.op_name].source_dir.priority,
            "op_dir": rel(entry.op_dir),
            "desc_dir": rel(entry.desc_dir),
            "src_dir": rel(entry.src_dir),
            "src_name": entry.src_name,
            "has_thinking": str(entry.has_thinking).lower(),
            "thinking_md": rel(entry.thinking_md) if entry.thinking_md else "",
            "kernel_src": rel(entry.kernel_src),
            "launch_h_src": rel(entry.launch_h_src),
            "plugin_src": rel(entry.plugin_src),
            "cmake_src": rel(entry.cmake_src),
        }
        for entry in context.entries
    )
    write_csv(manifest_dir / "entries.csv", fields, rows)


def _write_selection_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    selected = {item.op_name: item for item in context.selected_operator_dirs}
    fields = [
        "op", "source_dir", "source_priority", "op_dir", "selected",
        "selected_source_dir", "selected_op_dir",
    ]
    rows = (
        {
            "op": candidate.op_name,
            "source_dir": rel(candidate.source_dir.path),
            "source_priority": candidate.source_dir.priority,
            "op_dir": rel(candidate.op_dir),
            "selected": str(candidate.op_dir == selected[candidate.op_name].op_dir).lower(),
            "selected_source_dir": rel(selected[candidate.op_name].source_dir.path),
            "selected_op_dir": rel(selected[candidate.op_name].op_dir),
        }
        for candidate in context.operator_candidates
    )
    write_csv(manifest_dir / "source_selection.csv", fields, rows)


def _write_filter_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    path = manifest_dir / "filter_decisions.csv"
    if context.requested_filter_ops:
        fields = [
            "sample", "source_op", "op_name", "normalized_op", "op_source", "decision", "desc_md",
        ]
        rows = (
            {
                "sample": decision.entry.sample_name,
                "source_op": decision.entry.op_name,
                "op_name": decision.op_name,
                "normalized_op": decision.normalized_op,
                "op_source": decision.op_source,
                "decision": "removed" if decision.removed else "kept",
                "desc_md": rel(decision.entry.desc_md),
            }
            for decision in context.filter_decisions
        )
        write_csv(path, fields, rows)
    elif path.is_file():
        path.unlink()


def _write_prompt_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    prompt_fields = [
        "prompt_kind", "prompt_file", "sample", "source_desc_dir", "cases_yaml",
        "desc_md", "golden_py", "proto_yaml",
    ]
    prompt_rows = (
        {
            "prompt_kind": PROMPT_KIND,
            "prompt_file": rel(context.input_paths[entry.sample_name]),
            "sample": entry.sample_name,
            "source_desc_dir": rel(entry.desc_dir),
            "cases_yaml": rel(entry.cases_yaml),
            "desc_md": rel(entry.desc_md),
            "golden_py": rel(entry.golden_py),
            "proto_yaml": rel(entry.proto_yaml),
        }
        for entry in context.entries
    )
    write_csv(manifest_dir / "prompt_map.csv", prompt_fields, prompt_rows)


def _write_output_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    output_fields = [
        "output_kind", "output_file", "sample", "source_src_dir", "kernel_src",
        "launch_h_src", "plugin_src", "cmake_src", "has_thinking", "thinking_md",
    ]
    output_rows = (
        {
            "output_kind": OUTPUT_KIND,
            "output_file": rel(context.output_paths[entry.sample_name]),
            "sample": entry.sample_name,
            "source_src_dir": rel(entry.src_dir),
            "kernel_src": rel(entry.kernel_src),
            "launch_h_src": rel(entry.launch_h_src),
            "plugin_src": rel(entry.plugin_src),
            "cmake_src": rel(entry.cmake_src),
            "has_thinking": str(entry.has_thinking).lower(),
            "thinking_md": rel(entry.thinking_md) if entry.thinking_md else "",
        }
        for entry in context.entries
    )
    write_csv(manifest_dir / "output_map.csv", output_fields, output_rows)


def _write_jsonl_manifest(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    jsonl_fields = ["jsonl_kind", "jsonl_file", "sample", "prompt_file", "output_file"]
    jsonl_rows = (
        {
            "jsonl_kind": PROMPT_KIND,
            "jsonl_file": rel(context.jsonl_path),
            "sample": entry.sample_name,
            "prompt_file": rel(context.input_paths[entry.sample_name]),
            "output_file": rel(context.output_paths[entry.sample_name]),
        }
        for entry in context.entries
    )
    write_csv(manifest_dir / "jsonl_map.csv", jsonl_fields, jsonl_rows)


def _write_prompt_output_manifests(
    context: ManifestContext, manifest_dir: Path, rel: Callable[[Path], str]
) -> None:
    _write_prompt_manifest(context, manifest_dir, rel)
    _write_output_manifest(context, manifest_dir, rel)
    _write_jsonl_manifest(context, manifest_dir, rel)


def _write_optional_manifest_files(context: ManifestContext, manifest_dir: Path) -> None:
    path = manifest_dir / "skipped.txt"
    if context.skipped:
        write_text(path, "\n".join(context.skipped) + "\n")
    elif path.is_file():
        path.unlink()


def _manifest_summary(
    context: ManifestContext, rel: Callable[[Path], str]
) -> dict[str, object]:
    removed = [decision for decision in context.filter_decisions if decision.removed]
    filtered_by_op: dict[str, int] = {}
    for decision in removed:
        filtered_by_op[decision.normalized_op] = filtered_by_op.get(decision.normalized_op, 0) + 1
    before_filter = [decision.entry for decision in context.filter_decisions]
    if not before_filter:
        before_filter = context.entries
    matched = matched_filter_op_names(removed, context.filter_ops)
    selected_op_counts = {rel(source.path): 0 for source in context.source_dirs}
    for operator in context.selected_operator_dirs:
        selected_op_counts[rel(operator.source_dir.path)] += 1
    return {
        "source_root": rel(context.source_root),
        "source_dirs": [
            {"path": rel(source.path), "priority": source.priority}
            for source in context.source_dirs
        ],
        "selected_op_counts": selected_op_counts,
        "operator_candidate_count": len(context.operator_candidates),
        "shadowed_op_count": len(context.operator_candidates) - len(context.selected_operator_dirs),
        "requested_filter_ops": context.requested_filter_ops or [],
        "normalized_filter_ops": sorted(context.filter_ops),
        "sample_count_before_filter": len(before_filter),
        "op_count_before_filter": len({entry.op_name for entry in before_filter}),
        "filtered_sample_count": len(removed),
        "filtered_source_ops": sorted({decision.entry.op_name for decision in removed}),
        "filtered_by_op": dict(sorted(filtered_by_op.items())),
        "missing_filter_ops": sorted(context.filter_ops - matched),
        "output_dir": rel(context.output_dir),
        "prompt_kind": PROMPT_KIND,
        "output_kind": OUTPUT_KIND,
        "op_count": len({entry.op_name for entry in context.entries}),
        "sample_count": len(context.entries),
        "implementation_counts": {
            entry.op_name: entry.implementation_count
            for entry in context.entries
            if entry.implementation_count > 1
        },
        "with_thinking": context.with_thinking,
        "thinking_count": sum(1 for entry in context.entries if entry.has_thinking),
        "included_thinking_count": (
            sum(1 for entry in context.entries if entry.has_thinking)
            if context.with_thinking else 0
        ),
        "jsonl": {
            PROMPT_KIND: {"path": rel(context.jsonl_path), "records": len(context.entries)}
        },
        "skipped_count": len(context.skipped),
    }


def write_manifests(context: ManifestContext) -> None:
    manifest_dir = context.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def rel(path: Path) -> str:
        return repo_relative(path, context.repo_root)

    _write_entry_manifest(context, manifest_dir, rel)
    _write_selection_manifest(context, manifest_dir, rel)
    _write_filter_manifest(context, manifest_dir, rel)
    _write_prompt_output_manifests(context, manifest_dir, rel)
    _write_optional_manifest_files(context, manifest_dir)
    summary = _manifest_summary(context, rel)
    write_text(
        manifest_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed prompt, Markdown output, and JSONL dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Repository root (default: current directory).")
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="Source data root containing operator directories (default: current directory).")
    parser.add_argument("--source-dirs", nargs="+",
                        help="Ordered source-data collection paths under --source-root, highest priority first.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for generated artifacts.")
    parser.add_argument("--ops", nargs="+", help="Optional exact operator-directory include list.")
    parser.add_argument("--filter-ops", nargs="+",
                        help="Operator names to exclude; normalized names also match numeric replicas such as op_2.")
    parser.add_argument("--one-shot-example", default="sqrt",
                        help="Prompt one-shot example (default: sqrt).")
    parser.add_argument("--examples-root", type=Path, default=None,
                        help="Optional one-shot examples root.")
    parser.add_argument("--with-thinking", action="store_true",
                        help="Include per-source thinking.md when present.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove owned output directories before generation.")
    parser.add_argument("--skip-invalid", action="store_true",
                        help="Skip invalid src or src_N entries and record them in skipped.txt.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate source layout without writing artifacts.")
    return parser.parse_args()


def _requires_clean_filter(
    args: argparse.Namespace, filter_ops: set[str], output_dir: Path
) -> bool:
    return (
        bool(filter_ops)
        and not args.clean
        and not args.validate_only
        and has_owned_output_artifacts(output_dir)
    )


def _prepare_generation(args: argparse.Namespace) -> GenerationPlan:
    repo_root = args.repo_root.resolve()
    source_root = resolve_path(args.source_root, repo_root).resolve()
    output_dir = resolve_path(args.output_dir, repo_root).resolve()
    examples_root = (
        resolve_path(args.examples_root, repo_root).resolve()
        if args.examples_root else None
    )
    filter_ops = normalize_filter_ops(args.filter_ops)
    if _requires_clean_filter(args, filter_ops, output_dir):
        raise ValueError(
            "--filter-ops requires --clean when --output-dir already contains generated artifacts"
        )

    generator = load_prompt_generator(repo_root)
    generator.load_one_shot_example(args.one_shot_example, examples_root=examples_root)
    source_dirs = resolve_source_directories(source_root, args.source_dirs)
    selected, candidates = discover_operator_directories(
        source_dirs, only_ops=set(args.ops) if args.ops else None
    )
    discovered, skipped = discover_selected_entries(selected, skip_invalid=args.skip_invalid)
    entries, filter_decisions = filter_entries(discovered, filter_ops)
    removed = [decision for decision in filter_decisions if decision.removed]
    missing_filter_ops = sorted(filter_ops - matched_filter_op_names(removed, filter_ops))
    source_priority = " > ".join(repo_relative(source.path, repo_root) for source in source_dirs)
    return GenerationPlan(
        args=args, repo_root=repo_root, source_root=source_root, output_dir=output_dir,
        examples_root=examples_root, generator=generator, source_dirs=source_dirs,
        selected=selected, candidates=candidates, entries=entries, skipped=skipped,
        filter_ops=filter_ops, filter_decisions=filter_decisions, removed=removed,
        missing_filter_ops=missing_filter_ops, source_priority=source_priority,
        shadowed=len(candidates) - len(selected),
    )


def _log_validation(plan: GenerationPlan) -> None:
    LOGGER.info("[OK] source priority: %s", plan.source_priority)
    LOGGER.info("[OK] shadowed ops: %s", plan.shadowed)
    if plan.args.filter_ops:
        LOGGER.info("[OK] filtered samples: %s", len(plan.removed))
    LOGGER.info("[OK] validation samples: %s", len(plan.entries))
    LOGGER.info("[OK] validation ops: %s", len({entry.op_name for entry in plan.entries}))
    if plan.skipped:
        LOGGER.warning("[WARN] invalid entries: %s", len(plan.skipped))
    if plan.missing_filter_ops:
        LOGGER.warning("[WARN] filter ops not found: %s", ", ".join(plan.missing_filter_ops))


def _generate_artifacts(plan: GenerationPlan) -> Path:
    if plan.args.clean:
        clean_output_dir(plan.output_dir)
    else:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = render_inputs(
        plan.generator, plan.entries, plan.output_dir,
        example=plan.args.one_shot_example, examples_root=plan.examples_root,
    )
    output_paths = render_outputs(
        plan.entries, plan.output_dir, with_thinking=plan.args.with_thinking
    )
    jsonl_path = write_jsonl(plan.entries, input_paths, output_paths, plan.output_dir)
    write_intermediate_records(plan.entries, plan.output_dir, plan.repo_root)
    write_manifests(ManifestContext(
        entries=plan.entries, input_paths=input_paths, output_paths=output_paths,
        jsonl_path=jsonl_path, output_dir=plan.output_dir, repo_root=plan.repo_root,
        source_root=plan.source_root, source_dirs=plan.source_dirs,
        selected_operator_dirs=plan.selected, operator_candidates=plan.candidates,
        skipped=plan.skipped, with_thinking=plan.args.with_thinking,
        requested_filter_ops=plan.args.filter_ops, filter_ops=plan.filter_ops,
        filter_decisions=plan.filter_decisions,
    ))
    return jsonl_path


def _log_generation(plan: GenerationPlan, jsonl_path: Path) -> None:
    LOGGER.info("[OK] source priority: %s", plan.source_priority)
    LOGGER.info("[OK] shadowed ops: %s", plan.shadowed)
    if plan.args.filter_ops:
        LOGGER.info("[OK] filtered samples: %s", len(plan.removed))
    LOGGER.info("[OK] samples: %s", len(plan.entries))
    LOGGER.info("[OK] ops: %s", len({entry.op_name for entry in plan.entries}))
    LOGGER.info("[OK] jsonl: %s", repo_relative(jsonl_path, plan.repo_root))
    LOGGER.info("[OK] manifests: %s", repo_relative(plan.output_dir / "manifests", plan.repo_root))
    if plan.skipped:
        LOGGER.warning("[WARN] skipped invalid entries: %s", len(plan.skipped))
    if plan.missing_filter_ops:
        LOGGER.warning("[WARN] filter ops not found: %s", ", ".join(plan.missing_filter_ops))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    plan = _prepare_generation(parse_args())
    if plan.args.validate_only:
        _log_validation(plan)
        return
    _log_generation(plan, _generate_artifacts(plan))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None
