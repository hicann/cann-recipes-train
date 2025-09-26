# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import functools
import torch
from vllm.logger import logger


class CheckCounter:
    """
    A counter class to check if a specific number of calls has reached a threshold.
    """
    def __init__(self, threshold=100):
        self.total = 0  # Tracks the total number of times `check()` has been called since instantiation.
        self.counter = 0  # Tracks the count for the current cycle, resets after reaching the threshold.
        self.threshold = threshold  # The configured threshold for the counter to reset.

    def check(self):
        self.total += 1
        self.counter += 1
        if self.counter >= self.threshold:
            self.reset()
            return True
        return False

    def reset(self):
        self.counter = 0


class _Cache:
    global_rank: int = None


def rank_log_info(msg, force=False):
    rank = _Cache.global_rank
    if not rank:
        rank = _Cache.global_rank = torch.distributed.get_rank()
    if not force and rank != 0:  # 非强制模式下，默认只输出rank_0的打印
        return
    logger.info(f'[Rank={rank}]: {msg}')


def _patch_cls_method(original_method, before_callback=None, after_callback=None):
    @functools.wraps(original_method)
    def patched_method(self, *args, **kwargs):
        if before_callback and callable(before_callback):
            modified_params = before_callback(self, *args, **kwargs)
            if isinstance(modified_params, tuple) and len(modified_params) == 2:
                args, kwargs = modified_params

        ret = original_method(self, *args, **kwargs)

        if after_callback and callable(after_callback):
            final_ret = after_callback(self, ret, *args, **kwargs)
            if final_ret is not None:
                return final_ret

        return ret

    return patched_method


class _HookUtils:
    """
    A decorator-based toolkit for hooking into class methods.
    """
    @staticmethod
    def before(cls, method_name):
        original_method = getattr(cls, method_name)

        def decorator(callback):
            setattr(cls, method_name, _patch_cls_method(original_method, before_callback=callback))
            return callback

        return decorator

    @staticmethod
    def after(cls, method_name):
        original_method = getattr(cls, method_name)

        def decorator(callback):
            setattr(cls, method_name, _patch_cls_method(original_method, after_callback=callback))
            return callback

        return decorator


# Provides namespaced decorators like `@hook.before` and `@hook.after`
# to add logic without modifying the original class source code.
hook = _HookUtils()
