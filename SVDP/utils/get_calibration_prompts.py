# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import random

from datasets import load_dataset

from utils import save_prompts


def build_test_dataset() -> list[str]:
    prompts = []

    n_prompts_per_domain = 16

    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test").to_list()
    gsm8k_prompts = random.sample(gsm8k_ds, n_prompts_per_domain)
    fewshots = random.sample(gsm8k_ds, n_prompts_per_domain)
    prompts.extend(
        [
            f"Question: {fewshot['question']}\nAnswer: {fewshot['answer']}\nQuestion: {prompt['question']}\nAnswer: {prompt['answer']}"
            for fewshot, prompt in zip(fewshots, gsm8k_prompts, strict=False)
        ]
    )

    n_prompts_per_domain = 8

    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    mapping_n = {"1": 0, "2": 1, "3": 2, "4": 3}
    train_data = random.sample(
        load_dataset("allenai/openbookqa", "main", split="test").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question_stem']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    prompts.extend(random.sample(load_dataset("allenai/qasper", split="test")["abstract"], n_prompts_per_domain))

    # ARC-C TRAIN
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    train_data = random.sample(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    train_data = random.sample(
        load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    # PYTHON CODE
    n_prompts_per_domain = 64
    prompts.extend(
        random.sample(
            load_dataset("CM/codexglue_code2text_python", "default", split="test")["code"],
            n_prompts_per_domain,
        )
    )

    random.shuffle(prompts)
    return prompts


def build_dataset() -> list[str]:
    prompts = []

    n_prompts_per_domain = 16

    # FEWSHOT GSM8K TRAIN
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train").to_list()
    gsm8k_prompts = random.sample(gsm8k_ds, n_prompts_per_domain)
    fewshots = random.sample(gsm8k_ds, n_prompts_per_domain)
    prompts.extend(
        [
            f"Question: {fewshot['question']}\nAnswer: {fewshot['answer']}\nQuestion: {prompt['question']}\nAnswer: {prompt['answer']}"
            for fewshot, prompt in zip(fewshots, gsm8k_prompts, strict=False)
        ]
    )

    n_prompts_per_domain = 8

    # OPENBOOKQA TRAIN
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    mapping_n = {"1": 0, "2": 1, "3": 2, "4": 3}
    train_data = random.sample(
        load_dataset("allenai/openbookqa", "main", split="train").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question_stem']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    # QASPER TRAIN
    prompts.extend(random.sample(load_dataset("allenai/qasper", split="train")["abstract"], n_prompts_per_domain))

    # ARC-C TRAIN
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    train_data = random.sample(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    # ARC-E TRAIN
    train_data = random.sample(
        load_dataset("allenai/ai2_arc", "ARC-Easy", split="train").to_list(),
        n_prompts_per_domain,
    )
    for _i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"], strict=False):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)

    # PYTHON CODE
    n_prompts_per_domain = 64
    prompts.extend(
        random.sample(
            load_dataset("CM/codexglue_code2text_python", "default", split="train")["code"],
            n_prompts_per_domain,
        )
    )

    random.shuffle(prompts)
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect prompts for calibration")
    parser.add_argument("--output_path", help="json", type=str, required=True)
    parser.add_argument("--test", action="store_true", help="prompts for test")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible prompt sampling. "
        "If omitted, sampling is non-deterministic (legacy behavior).",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.test:
        prompts = build_test_dataset()
    else:
        prompts = build_dataset()
    save_prompts(prompts, args.output_path)


if __name__ == "__main__":
    main()
