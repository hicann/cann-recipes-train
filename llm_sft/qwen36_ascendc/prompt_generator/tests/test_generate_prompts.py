#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "generate_prompts.py"


def load_module(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prompt_generator = load_module(
    "prompt_generator",
    SCRIPT_PATH,
)


class FixedPromptGeneratorTest(unittest.TestCase):
    def make_inputs(self, module, op_dir: Path):
        return module.OpPromptInputs(
            name="demo",
            op_dir=op_dir,
            cases_yaml="CASES_PAYLOAD",
            desc_md=(
                "## 1. 算子说明\nDESC_PAYLOAD\n\n"
                "## 2. 标准 Golden 代码\n```python\nEMBEDDED_GOLDEN\n```\n\n"
                "## 3. 约束\nTAIL_PAYLOAD\n"
            ),
            golden_py="GOLDEN_PAYLOAD",
            proto_yaml="PROTO_PAYLOAD",
        )

    def test_render_has_fixed_markdown_oneshot_content(self) -> None:
        prompt = prompt_generator.render_prompt(
            self.make_inputs(prompt_generator, Path("demo"))
        )

        self.assertIn("输出只需 **4 段** 代码块", prompt)
        self.assertIn("样例算子 `sqrt`", prompt)
        self.assertIn("DESC_PAYLOAD", prompt)
        self.assertIn("GOLDEN_PAYLOAD", prompt)
        self.assertNotIn("EMBEDDED_GOLDEN", prompt)

    def test_one_shot_is_after_operator_specification(self) -> None:
        prompt = prompt_generator.render_prompt(
            self.make_inputs(prompt_generator, Path("demo"))
        )

        op_specification = prompt.index("## 本任务算子规格等信息详述")
        output_format = prompt.index("## 输出格式")
        one_shot = prompt.index("## 示例代码")
        self.assertLess(op_specification, output_format)
        self.assertLess(output_format, one_shot)
        self.assertIn("kernel_src\n\n```cpp", prompt)
        self.assertNotIn("{{", prompt)

    def test_generate_multiple_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for op_name in ("first", "second"):
                op_dir = root / "ops" / op_name
                op_dir.mkdir(parents=True)
                for filename, content in {
                    "cases.yaml": f"{op_name}_CASES",
                    "desc.md": f"{op_name}_DESC",
                    "golden.py": f"{op_name}_GOLDEN",
                    "proto.yaml": f"{op_name}_PROTO",
                }.items():
                    (op_dir / filename).write_text(content, encoding="utf-8")

            written = prompt_generator.generate_prompts(
                root / "ops",
                ["first", "second"],
                root / "out",
            )

            self.assertEqual(
                written,
                [root / "out" / "first.md", root / "out" / "second.md"],
            )
            self.assertIn("first_DESC", written[0].read_text(encoding="utf-8"))
            self.assertIn("second_DESC", written[1].read_text(encoding="utf-8"))

    def test_missing_input_is_reported_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "ops" / "demo"
            op_dir.mkdir(parents=True)
            (op_dir / "desc.md").write_text("DESC", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "cases.yaml"):
                prompt_generator.generate_prompts(
                    root / "ops", ["demo"], root / "out"
                )
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
