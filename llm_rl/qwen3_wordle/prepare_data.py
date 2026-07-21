# coding=utf-8
# Adapted from Prime Intellect's Verifiers Wordle environment:
# https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/wordle/wordle.py
# Game prompt text sourced from TextArena Wordle-v0:
# https://github.com/LeonGuertler/TextArena/blob/main/textarena/envs/Wordle/env.py
# Copyright (c) 2025 Leon Guertler
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Generate Wordle training/eval data in verl parquet format.

Creates ``wordle_train.parquet`` and ``wordle_test.parquet`` compatible
with verl's RLHFDataset. Word list is sourced from TextArena Wordle-v0
(same as prime-rl). Output schema:

  - prompt: List[dict] in OpenAI chat format (system + user)
  - answer: str — the secret 5-letter word
  - index: str — unique example identifier
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_wordle_words() -> list[str]:
    """Load 5-letter words from TextArena Wordle-v0 (same as prime-rl)."""
    import textarena as ta
    env = ta.make("Wordle-v0")
    env.reset(num_players=1)
    words = env.word_list
    if isinstance(words, dict):
        words = [w for values in words.values()
                 for w in (values if isinstance(values, (list, tuple)) else [values])]
    return sorted(set(w.lower() for w in words if len(w) == 5 and w.isalpha()))


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a competitive game player. Make sure you read the game instructions "
        "carefully, and always follow the required format.\n\n"
        "In each turn, think step-by-step, then output your guess exactly as "
        "<guess>[word]</guess>."
    ),
}

GAME_PROMPT = {
    "role": "user",
    "content": (
        "\n[GAME] You are Playing Wordle.\n"
        "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
        "For each guess, wrap your word in square brackets (e.g., '[apple]').\n"
        "Feedback for each letter will be given as follows:\n"
        "  - G (green): correct letter in the correct position\n"
        "  - Y (yellow): letter exists in the word but in the wrong position\n"
        "  - X (wrong): letter is not in the word\n"
        "Enter your guess to begin.\n"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)

    words = _load_wordle_words()
    if len(words) < args.num_test:
        raise ValueError(
            f"Not enough 5-letter words in TextArena: {len(words)} available, "
            f"{args.num_test} needed for test set"
        )

    random.shuffle(words)
    test_words = words[: args.num_test]  # stable order and distinct eval words
    test_word_set = set(test_words)
    train_pool = [w for w in words if w not in test_word_set]  # exclude test words
    # Train: sample with replacement (same as prime-rl), 2000 examples from train_pool
    train_words = [random.choice(train_pool) for _ in range(args.num_train)]

    for name, word_list in [("wordle_train", train_words), ("wordle_test", test_words)]:
        rows = []
        for row_index, w in enumerate(word_list):
            messages = [SYSTEM_PROMPT, GAME_PROMPT]
            rows.append(
                {
                    "prompt": messages,
                    "raw_prompt": messages,
                    "answer": w,
                    "index": f"{name}_{row_index:04d}_{w}",
                    "data_source": "wordle",
                    "reward_model": {"ground_truth": w},
                }
            )
        df = pd.DataFrame(rows)
        output_path = Path(args.output_dir) / f"{name}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved %d examples to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
