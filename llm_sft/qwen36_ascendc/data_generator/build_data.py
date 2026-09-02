#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
"""Compatibility entry point for :mod:`generate_data`."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from llm_sft.qwen36_ascendc.data_generator.generate_data import main
else:  # Direct execution has no package context.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_data import main


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None
