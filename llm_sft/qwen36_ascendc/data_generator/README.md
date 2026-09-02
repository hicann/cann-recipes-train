# 数据生成器

`generate_data.py` 将精简的 `path-to-data` 数据样例转换为
prompt/output/JSONL 数据集。它使用自包含的 `prompt_generator`，并始终生成固定的
Markdown 代码块单轮格式。

```bash
python data_generator/generate_data.py \
  --source-root path-to-data \
  --output-dir generated-data \
  --clean
```

源数据支持单实现和多实现两种目录结构。单实现算子使用 `src`，多实现算子使用
`src_N`（同一算子下不要混用两种命名）：

```text
<source-root>/
└── <operator>/
    ├── desc/{cases.yaml,desc.md,golden.py,proto.yaml}
    └── src/{op_kernel,op_plugin,CMakeLists.txt}
```

`src` 对应单一代码实现；也可以使用 `src_1`、`src_2` 等目录表示多种代码实现，
每个目录对应一条独立的响应记录。比如 `opensourced_codes/ops_transcribed` 就是
单实现 `src` 格式的数据集。

生成的产物会写入指定输出目录下：

```text
inputs/md-code-block-oneshot/
outputs/md-code-block/
jsonl/md-code-block-oneshot.jsonl
manifests/{entries,source_selection,prompt_map,output_map,jsonl_map}.csv
manifests/summary.json
intermediate/<sample>/source.json
```

使用 `--filter-ops` 可排除完整的算子目录。名称会按照增强器的规则进行规范化，
并匹配数字副本：

```bash
python data_generator/generate_data.py \
  --source-root path-to-data \
  --output-dir generated-data \
  --filter-ops exp sigmoid \
  --clean
```

`--ops` 用于选择子集，`--validate-only` 执行只读校验，
`--skip-invalid` 会将无效的 `src` 或 `src_N` 条目记录到 `manifests/skipped.txt`。
