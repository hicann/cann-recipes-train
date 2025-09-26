# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mathruler.grader import extract_boxed_content

def extract_solution(solution_str, ground_truth):
    extract_output = extract_boxed_content(solution_str)
    if ground_truth in solution_str or extract_output == ground_truth:
        return 1.0
    return 0.0

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for deepscaler.

    Reference: Trung, Luong
    Args:
        solution_str (str): the solution text
        ground_truth (str): the ground truth
        method: the method to use to compute the score. Defaults to "strict".
        format_score: the format of the score. Defaults to 0.0.
        score: the score. Defaults to 1.0.
    """
    score = extract_solution(solution_str=solution_str, ground_truth=ground_truth)
    return score
