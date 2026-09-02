#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
"""Generate fixed md-code-block one-shot (post-oneshot) prompts.

This directory is a self-contained prompt generator. It intentionally exposes
no template, output-format, or one-shot placement switches: every generated
prompt uses the fixed md-code-block-oneshot variant.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


REQUIRED_OP_FILES = ("cases.yaml", "desc.md", "golden.py", "proto.yaml")
TEMPLATE_FILE = "main_assembly_v1.5.md"
OUTPUT_FORMAT_FILE = "output_format_code_block_md.md"
ONE_SHOT_TEMPLATE_FILE = "one_shot_code_block_md.md"
DEFAULT_EXAMPLE = "sqrt"
TASK_DELIVERABLES = "**kernel**、**launch_h**、**plugin**、**cmake** 4 个交付件"
_MD_HEADING_RE = re.compile(r"^(?P<prefix>\s{0,3}(?P<marks>#{1,6})\s+)(?P<body>.*?)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<prefix>\s{0,3}#{1,6}\s+)"
    r"(?P<number>\d+)"
    r"(?:(?P<sep>[.．、])(?!\d)\s*|\s+)"
    r"(?P<title>.*)$"
)
_FENCE_RE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")


@dataclass(frozen=True)
class OpPromptInputs:
    name: str
    op_dir: Path
    cases_yaml: str
    desc_md: str
    golden_py: str
    proto_yaml: str


@dataclass(frozen=True)
class OneShotExample:
    name: str
    kernel_src: str
    launch_h_src: str
    plugin_src: str
    cmake_src: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _local_file(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def load_template() -> str:
    return _read_text(_local_file(TEMPLATE_FILE))


def load_output_format() -> str:
    return _read_text(_local_file(OUTPUT_FORMAT_FILE)).rstrip("\n")


def load_one_shot_template() -> str:
    return _read_text(_local_file(ONE_SHOT_TEMPLATE_FILE))


def _validate_name(name: str, *, kind: str) -> None:
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid {kind} name {name!r}: pass a directory name, not a path")


def _examples_root(examples_root: Path | None = None) -> Path:
    return examples_root or _local_file("examples")


def list_examples(examples_root: Path | None = None) -> list[str]:
    root = _examples_root(examples_root)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def locate_example_dir(example: str, examples_root: Path | None = None) -> Path:
    _validate_name(example, kind="example")
    root = _examples_root(examples_root)
    example_dir = root / example
    if not example_dir.is_dir():
        available = list_examples(examples_root)
        raise FileNotFoundError(
            f"one-shot example {example!r} not found under {root}; "
            f"available examples: {available or '(none)'}"
        )
    return example_dir


def load_one_shot_example(
    example: str = DEFAULT_EXAMPLE,
    *,
    examples_root: Path | None = None,
) -> OneShotExample:
    example_dir = locate_example_dir(example, examples_root)

    def required(relative_path: str) -> str:
        path = example_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"one-shot example {example!r} is missing {relative_path}; expected {path}"
            )
        return _read_text(path)

    return OneShotExample(
        name=example,
        kernel_src=required(f"op_kernel/{example}_kernel.cpp"),
        launch_h_src=required(f"op_kernel/{example}_launch.h"),
        plugin_src=required(f"op_plugin/{example}_plugin.cpp"),
        cmake_src=required("CMakeLists.txt"),
    )


def render_one_shot(
    *,
    example: str = DEFAULT_EXAMPLE,
    examples_root: Path | None = None,
) -> str:
    template = load_one_shot_template()
    one_shot = load_one_shot_example(example, examples_root=examples_root)
    replacements = {
        "{{example_name}}": one_shot.name,
        "{{kernel_src}}": one_shot.kernel_src,
        "{{launch_h_src}}": one_shot.launch_h_src,
        "{{plugin_src}}": one_shot.plugin_src,
        "{{cmake_src}}": one_shot.cmake_src,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template.rstrip("\n")


def load_op_inputs(op_root: Path, op_name: str) -> OpPromptInputs:
    _validate_name(op_name, kind="op")
    op_dir = op_root / op_name
    if not op_dir.is_dir():
        raise FileNotFoundError(f"op directory not found: {op_dir}")

    missing = [name for name in REQUIRED_OP_FILES if not (op_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{op_dir} is missing required file(s): {', '.join(missing)}"
        )

    return OpPromptInputs(
        name=op_name,
        op_dir=op_dir,
        cases_yaml=_read_text(op_dir / "cases.yaml"),
        desc_md=_read_text(op_dir / "desc.md"),
        golden_py=_read_text(op_dir / "golden.py"),
        proto_yaml=_read_text(op_dir / "proto.yaml"),
    )


def _line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _numbered_heading(line: str) -> dict[str, object] | None:
    text, newline = _line_ending(line)
    heading = _MD_HEADING_RE.match(text)
    if not heading:
        return None
    numbered = _NUMBERED_HEADING_RE.match(text)
    number = int(numbered.group("number")) if numbered else None
    title = numbered.group("title") if numbered else heading.group("body")
    return {
        "level": len(heading.group("marks")),
        "number": number,
        "title": title,
        "newline": newline,
    }


def _is_standard_golden_code_heading(title: str) -> bool:
    normalized = re.sub(r"\s+", "", title).lower()
    return "标准" in normalized and "golden" in normalized and "代码" in normalized


def _renumber_heading(line: str, new_number: int) -> str:
    text, newline = _line_ending(line)
    numbered = _NUMBERED_HEADING_RE.match(text)
    if not numbered:
        return line
    return (
        text[: numbered.start("number")]
        + str(new_number)
        + text[numbered.end("number"):]
        + newline
    )


def _fence_marker(line: str) -> str | None:
    text, _ = _line_ending(line)
    match = _FENCE_RE.match(text)
    return match.group("fence")[0] if match else None


def _next_fence_state(fence_char: str | None, marker: str) -> str | None:
    if fence_char == marker:
        return None
    if fence_char is None:
        return marker
    return fence_char


def _find_golden_section(
    lines: list[str],
) -> tuple[int, int, int | None] | None:
    fence_char: str | None = None
    for index, line in enumerate(lines):
        marker = _fence_marker(line)
        if marker:
            fence_char = _next_fence_state(fence_char, marker)
            continue
        if fence_char is not None:
            continue
        heading = _numbered_heading(line)
        if heading is None:
            continue
        if not _is_standard_golden_code_heading(str(heading["title"])):
            continue
        number = heading["number"] if isinstance(heading["number"], int) else None
        return index, int(heading["level"]), number
    return None


def _find_section_end(lines: list[str], start: int, level: int) -> int:
    fence_char: str | None = None
    for index in range(start + 1, len(lines)):
        marker = _fence_marker(lines[index])
        if marker:
            fence_char = _next_fence_state(fence_char, marker)
            continue
        if fence_char is not None:
            continue
        heading = _numbered_heading(lines[index])
        if heading is not None and int(heading["level"]) <= level:
            return index
    return len(lines)


def _should_renumber(heading: dict[str, object] | None, level: int, removed: int) -> bool:
    if heading is None or int(heading["level"]) != level:
        return False
    number = heading["number"]
    return isinstance(number, int) and number > removed


def _renumber_sections(
    lines: list[str], start: int, level: int, removed: int
) -> None:
    for index in range(start, len(lines)):
        heading = _numbered_heading(lines[index])
        if _should_renumber(heading, level, removed):
            lines[index] = _renumber_heading(lines[index], int(heading["number"]) - 1)


def strip_desc_standard_golden_code(desc_md: str) -> str:
    """Remove desc.md's embedded standard Golden code section while rendering."""
    lines = desc_md.splitlines(keepends=True)
    section = _find_golden_section(lines)
    if section is None:
        return desc_md
    section_start, section_level, removed_number = section
    section_end = _find_section_end(lines, section_start, section_level)
    filtered = lines[:section_start] + lines[section_end:]
    if removed_number is not None:
        _renumber_sections(filtered, section_start, section_level, removed_number)
    return "".join(filtered)


def render_prompt(
    inputs: OpPromptInputs,
    *,
    example: str = DEFAULT_EXAMPLE,
    examples_root: Path | None = None,
) -> str:
    """Render the fixed Markdown code-block prompt with a post-oneshot example."""
    prompt = load_template()
    replacements = {
        "{{TASK_DELIVERABLES}}": TASK_DELIVERABLES,
        "{{DESC_MD}}": strip_desc_standard_golden_code(inputs.desc_md),
        "{{PROTO_YAML}}": inputs.proto_yaml,
        "{{CASES_YAML}}": inputs.cases_yaml,
        "{{GOLDE_PY}}": inputs.golden_py,
        "{{GOLDEN_PY}}": inputs.golden_py,
        "{{OUTPUT_FORMAT}}": load_output_format(),
        "{{ONE_SHOT}}": render_one_shot(
            example=example,
            examples_root=examples_root,
        ),
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def generate_prompts(
    op_root: Path,
    op_names: Iterable[str],
    out_dir: Path,
    *,
    example: str = DEFAULT_EXAMPLE,
    examples_root: Path | None = None,
) -> list[Path]:
    """Generate one fixed-format <op>.md prompt per operator."""
    load_one_shot_example(example, examples_root=examples_root)
    inputs = [load_op_inputs(op_root, op_name) for op_name in op_names]
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for item in inputs:
        out_path = out_dir / f"{item.name}.md"
        out_path.write_text(
            render_prompt(item, example=example, examples_root=examples_root),
            encoding="utf-8",
        )
        written.append(out_path)
    return written


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed md-code-block-oneshot (post-oneshot) cann-bench v1.5 prompts. "
            "Each op directory must contain cases.yaml, desc.md, golden.py, and proto.yaml."
        )
    )
    parser.add_argument(
        "--op-root",
        required=True,
        type=Path,
        help="Root directory containing operator subdirectories",
    )
    parser.add_argument(
        "--ops",
        required=True,
        nargs="+",
        help="One or more operator directory names under --op-root",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where <op>.md prompt files will be written",
    )
    parser.add_argument(
        "--example",
        default=DEFAULT_EXAMPLE,
        help=f"One-shot example name under --examples-root (default: {DEFAULT_EXAMPLE})",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=None,
        help="One-shot example root (default: examples/ next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()
    try:
        written = generate_prompts(
            args.op_root,
            _unique_preserve_order(args.ops),
            args.out_dir,
            example=args.example,
            examples_root=args.examples_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None

    for path in written:
        LOGGER.info("[OK] wrote %s (md-code-block-oneshot, post-oneshot=%s)", path, args.example)


if __name__ == "__main__":
    main()
