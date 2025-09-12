# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

import sys
import importlib


def apply_bert_padding():
    from verl_patches import bert_padding
    sys.modules['flash_attn.bert_padding'] = bert_padding


def apply_optimizer_patch():
    if 'apex.optimizers' in sys.modules:
        apex_opt = sys.modules['apex.optimizers']
    else:
        apex_opt = importlib.import_module('apex.optimizers')
    from mindspeed.optimizer.adamw import AdamW
    apex_opt.FusedAdam = AdamW
    if not hasattr(apex_opt, '__path__'):
        apex_opt.__path__ = []

    sys.modules['apex.optimizers'] = apex_opt
    sys.modules['apex.optimizers.FusedAdam'] = AdamW


# apply patches
apply_bert_padding()
apply_optimizer_patch()