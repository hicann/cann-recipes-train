# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
