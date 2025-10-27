# coding=utf-8
# Adapted from Bytedance Ltd. and/or its affiliates' implementation at:
# https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math_reward.py
# which is derived from EleutherAI's original work available at:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
# Additional references are made to OpenCompass Authors' work at:
# https://github.com/open-compass/opencompass/blob/main/opencompass/evaluator/math_evaluator.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2020 OpenCompass Authors. All rights reserved.
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


import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    accuracy_score = compute_accuracy_score(solution_str, ground_truth)

    # If the answer is incorrect, there will be no Format Score or Chain of Thought Score.
    if accuracy_score < 0.7:
        return 0.0

    format_score = compute_format_score(solution_str)
    cot_score = compute_cot_score(solution_str)

    final_score = accuracy_score + format_score + cot_score
    return final_score


def compute_accuracy_score(solution_str, ground_truth) -> float:
    # 1. Accuracy Score (0.7 Points)
    ref_with_env = f"${ground_truth}$"
    gold_parsed = parse(
        ref_with_env,
        extraction_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
    )

    if len(gold_parsed) != 0:
        answer_parsed = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        math_verify_score = float(verify(answer_parsed, gold_parsed))
    else:
        math_verify_score = 0.0

    if math_verify_score < 0.99:
        return 0.0
    else:
        return 0.7


def compute_format_score(solution_str) -> float:
    # 2. Format Score (0.2 Points)
    format_score = _calculate_basic_format_score(solution_str)

    if not _check_tag_order_and_content(solution_str, format_score):
        return 0.0

    return format_score


def _calculate_basic_format_score(solution_str: str) -> float:
    format_score = 0.0

    think_open_count = solution_str.count("<think>\n")
    think_close_count = solution_str.count("\n</think>\n")
    answer_open_count = solution_str.count("\n<answer>\n")
    answer_close_count = solution_str.count("\n</answer>")

    if think_open_count == 1:
        format_score += 0.05
    if think_close_count == 1:
        format_score += 0.05
    if answer_open_count == 1:
        format_score += 0.05
    if answer_close_count == 1:
        format_score += 0.05

    think_start = solution_str.find("<think>")
    answer_end = solution_str.find("</answer>")

    if think_start > 0:
        text_before_think = solution_str[:think_start].strip()
        if text_before_think:
            penalty = min(len(text_before_think) * 0.001, 0.1)
            format_score = max(format_score - penalty, 0)

    if answer_end != -1:
        text_after_answer = solution_str[answer_end + len("</answer>"):].strip()
        if text_after_answer:
            penalty = min(len(text_after_answer) * 0.001, 0.1)
            format_score = max(format_score - penalty, 0)

    return format_score


def _check_tag_order_and_content(solution_str: str, format_score: float) -> bool:
    think_start = solution_str.find("<think>")
    think_end = solution_str.find("</think>")
    answer_start = solution_str.find("<answer>")
    answer_end = solution_str.find("</answer>")

    invalid_order = (
        (think_start != -1 and think_end != -1 and think_end < think_start) or
        (think_end != -1 and answer_start != -1 and answer_start < think_end) or
        (answer_start != -1 and answer_end != -1 and answer_end < answer_start)
    )
    if invalid_order:
        return False

    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    if re.match(pattern, solution_str, re.DOTALL):
        format_score = min(format_score + 0.05, 0.2)

    all_tags_present = (
        think_start != -1 and
        think_end != -1 and
        answer_start != -1 and
        answer_end != -1
    )

    if all_tags_present:
        think_content = solution_str[think_start + len("<think>"):think_end].strip()
        answer_content = solution_str[answer_start + len("<answer>"):answer_end].strip()

        content_empty = not think_content or not answer_content
        if content_empty:
            format_score *= 0.5

        think_shorter_than_answer = len(think_content) < len(answer_content)
        if think_shorter_than_answer:
            format_score = 0.0

    return True


def compute_cot_score(solution_str) -> float:
    # 3. Chain of Thought Score (0.1 Points)
    cot_score = 0.0
    think_start = solution_str.find("<think>")
    think_end = solution_str.find("</think>")

    if think_start == -1 or think_end == -1:
        return cot_score

    think_content = solution_str[think_start + len("<think>"):think_end].strip()

    step_patterns = [
        r"^(?:step\s*|step|Step\s*|Step)\s*([1-9]\d*)\s*[\:\.]\s*(.+)$",
        r"^([1-9]\d*)\s*[\.\)]\s*(.+)$",
        r"^[\(\[]?\s*([1-9]\d*)\s*[\)\]]\s*(.+)$"
    ]

    step_numbers = []
    lines = think_content.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pattern in step_patterns:
            matches = re.findall(pattern, line)
            if not matches:
                continue
            step_num = int(matches[0][0])
            step_text = matches[0][1].strip()
            if len(step_text) > 3:
                step_numbers.append((step_num, step_text))
            break

    valid_sequential_steps = 0
    expected_step = 1
    for step_num, _ in step_numbers:
        if step_num == expected_step:
            valid_sequential_steps += 1
            expected_step += 1

    cot_keywords = ["solve", "equation", "calculate", "substitute", "therefore", "because", "thus",
                    "sum", "product", "derivative", "integral"]
    keyword_count = 0
    for _, step_text in step_numbers:
        if any(keyword in step_text.lower() for keyword in cot_keywords):
            keyword_count += 1

    if valid_sequential_steps >= 3 and keyword_count >= 2:
        cot_score = 0.1
    elif valid_sequential_steps >= 2 and keyword_count >= 1:
        cot_score = 0.05

    return cot_score
