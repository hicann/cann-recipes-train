# 固定格式 Prompt 生成器

该目录是可独立拷贝使用的 CANNBench v1.5 prompt 生成器。所有输出固定为
`md-code-block-oneshot` 形式：

- 交付件使用 Markdown fenced code block；
- 始终包含 one-shot 示例；
- one-shot 位于算子规格之后（post-oneshot）；
- 不提供模板、输出格式或 one-shot 位置切换参数。

## 使用方法

```bash
python prompt_generator/generate_prompts.py \
  --op-root path-to-op-root \
  --ops sigmoid exp \
  --out-dir generated_prompts
```

每个算子会生成 `<out-dir>/<op>.md`。`--ops` 中重复的算子会按首次出现的顺序去重。

## 参数

- `--op-root`：算子根目录，脚本按 `<op-root>/<op>` 读取输入。
- `--ops`：一个或多个算子目录名。
- `--out-dir`：prompt 输出目录。
- `--example`：one-shot 样例名，默认 `sqrt`。
- `--examples-root`：自定义样例根目录，默认使用脚本同级的 `examples/`。

每个算子目录必须包含 `cases.yaml`、`desc.md`、`golden.py` 和 `proto.yaml`。
每个样例目录必须包含：

```text
<example>/
├── CMakeLists.txt
├── op_kernel/<example>_kernel.cpp
├── op_kernel/<example>_launch.h
└── op_plugin/<example>_plugin.cpp
```

脚本只依赖 Python 3.10+ 标准库，运行时不会读取其他模板。
