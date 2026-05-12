import random
import argparse

from datasets import load_dataset

from utils import fix_ssl, save_prompts


def build_dataset_old() -> list[str]:

    random.seed(42)

    prompts = []
    
    n_prompts_per_domain = 64

    # FEWSHOT GSM8K TRAIN
    gsm8k_ds = load_dataset("gsm8k", "main", split="train").to_list()
    gsm8k_prompts = random.sample(gsm8k_ds, n_prompts_per_domain)
    fewshots = random.sample(gsm8k_ds, n_prompts_per_domain)
    prompts.extend(
        [
            f"Question: {fewshot['question']}\nAnswer: {fewshot['answer']}\nQuestion: {prompt['question']}\nAnswer: {prompt['answer']}"
            for fewshot, prompt in zip(fewshots, gsm8k_prompts)
        ]
    )

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

def build_test_dataset() -> list[str]:
    from datasets import load_dataset
    prompts = []
    
    n_prompts_per_domain = 16

    gsm8k_ds = load_dataset("gsm8k", "main", split="test").to_list()
    gsm8k_prompts = random.sample(gsm8k_ds, n_prompts_per_domain)
    fewshots = random.sample(gsm8k_ds, n_prompts_per_domain)
    prompts.extend(
        [
            f"Question: {fewshot['question']}\nAnswer: {fewshot['answer']}\nQuestion: {prompt['question']}\nAnswer: {prompt['answer']}"
            for fewshot, prompt in zip(fewshots, gsm8k_prompts)
        ]
    )

    n_prompts_per_domain = 8
    
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    mapping_n = {"1": 0, "2": 1, "3": 2, "4": 3}
    train_data = random.sample(
        load_dataset("allenai/openbookqa", "main", split="test").to_list(),
        n_prompts_per_domain,
    )
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question_stem']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)
    
    prompts.extend(
        random.sample(
            load_dataset("allenai/qasper", split="test")["abstract"], n_prompts_per_domain
        )
    )
    
    # ARC-C TRAIN
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    train_data = random.sample(
        load_dataset("ai2_arc", "ARC-Challenge", split="test").to_list(),
        n_prompts_per_domain,
    )
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)
        
    train_data = random.sample(
        load_dataset("ai2_arc", "ARC-Easy", split="test").to_list(),
        n_prompts_per_domain,
    )
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
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
    # save_prompts(prompts, "test_prompts.json")
    return prompts

def build_dataset() -> list[str]:
    from datasets import load_dataset
    prompts = []
    
    n_prompts_per_domain = 16
    
    # WIKI
    # prompts.extend(
    #     random.sample(
    #         load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")[
    #             "text"
    #         ],
    #         n_prompts_per_domain,
    #     )
    # )

    # FEWSHOT GSM8K TRAIN
    gsm8k_ds = load_dataset("gsm8k", "main", split="train").to_list()
    gsm8k_prompts = random.sample(gsm8k_ds, n_prompts_per_domain)
    fewshots = random.sample(gsm8k_ds, n_prompts_per_domain)
    prompts.extend(
        [
            f"Question: {fewshot['question']}\nAnswer: {fewshot['answer']}\nQuestion: {prompt['question']}\nAnswer: {prompt['answer']}"
            for fewshot, prompt in zip(fewshots, gsm8k_prompts)
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
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question_stem']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)
    
    # QASPER TRAIN
    prompts.extend(
        random.sample(
            load_dataset("allenai/qasper", split="train")["abstract"], n_prompts_per_domain
        )
    )
    
    # ARC-C TRAIN
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    train_data = random.sample(
        load_dataset("ai2_arc", "ARC-Challenge", split="train").to_list(),
        n_prompts_per_domain,
    )
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
            prompt = prompt + "(" + label + ") " + text + "\n"
        answer = sample["answerKey"]
        prompt = (
            prompt
            + f"Answer:\n({answer}) {choices['text'][mapping[answer] if answer in mapping else mapping_n[answer]]}"
        )
        prompts.append(prompt)
        
    # ARC-E TRAIN
    train_data = random.sample(
        load_dataset("ai2_arc", "ARC-Easy", split="train").to_list(),
        n_prompts_per_domain,
    )
    for i, sample in enumerate(train_data):
        prompt = f"Question:\n{sample['question']}\nOptions:\n"
        choices = sample["choices"]
        for text, label in zip(choices["text"], choices["label"]):
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
    # save_prompts(prompts, "calibration_prompts.json")
    return prompts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect prompts for calibration")
    parser.add_argument("--output_path", help="json", type=str, required=True)  
    parser.add_argument('--test', action='store_true', help="prompts for test")
    args = parser.parse_args()
    return args


def main():
    fix_ssl()
    args = parse_args()
    if args.test:
        prompts = build_test_dataset()
    else:
        prompts = build_dataset()
    save_prompts(prompts, args.output_path)

if __name__ == "__main__":
    main()
