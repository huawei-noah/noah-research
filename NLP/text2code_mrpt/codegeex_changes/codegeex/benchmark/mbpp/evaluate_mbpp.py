# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
# This code is based on CodeGeeX's original codegeex/benchmark/humaneval-x/evaluate_humaneval_x.py and has been adapted for the MBPP benchmark

import os
import sys
import fire
import json
import gzip
import regex
import numpy as np

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.benchmark.metric import estimate_pass_at_k
from codegeex.benchmark.execution import check_correctness

LANGUAGE_NAME = {
    "cpp"   : "CPP",
    "go"    : "Go",
    "java"  : "Java",
    "js"    : "JavaScript",
    "python": "Python",
}


def process_mbpp_test(sample, problems, example_test=False):
    task_id = sample["task_id"]
    if isinstance(task_id, int):
        pass
    else:
        if '_' in task_id:
            task_id = task_id[:task_id.find('_')]
    task_id = int(task_id)

    prompt = sample["prompt"]
    tests = "\n".join([f'    {tl}' for tl in problems[task_id]["test_list"]])
    signature = (problems[task_id]["code"][problems[task_id]["code"].find('def ') + 4:problems[task_id]["code"].find('(')])
    test = f'def check({signature}):\n{tests}\n\ncheck({signature})'

    code = sample["generation"]
    code_ = []
    for line in code.split("\n"):
        if (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
            break
        code_.append(line)
    code = "\n".join(code_)
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    test_string = test_setup + prompt + code + "\n\n" + test + "\n"

    return test_string


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(
        input_file: str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: float = 500.0,
        problem_file: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,
):
    if example_test:
        print("Example test...")

    problems = read_dataset(problem_file,
                            dataset_type="humaneval")
    sample_jsonl = stream_jsonl_all(input_file)

    if example_test:
        suffix = "_example_test.jsonl"
    else:
        suffix = "_results.jsonl"
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, input_file.split('/')[-1].replace(".jsonl", suffix))
    else:
        out_file = os.path.join(input_file.replace(".jsonl", suffix))

    if "/codegeex/benchmark/humaneval-x/" in input_file:
        test_groundtruth = True

    if "-to-" in input_file:
        translation_mode = True
    else:
        translation_mode = False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            print("Testing ground truth...")
            for sample in tqdm(problems.values()):
                task_id = int(sample["task_id"])
                lang = "python"
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["generation"] = sample["canonical_solution"]
                sample["test_code"] = process_mbpp_test(sample, problems, example_test)
                if sample["test_code"] is None:
                    continue
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            print("Reading samples...")
            for sample in tqdm(sample_jsonl):
                task_id = sample["task_id"]
                lang = 'python'
                if translation_mode:
                    task_id = sample["task_id"].split("/")[-1]
                    lang = regex.findall("-to-.*-", input_file)[0].split("-to-")[-1].rstrip("-")
                    for l in LANGUAGE_NAME:
                        if l in lang:
                            lang = l
                            break
                    task_id = f"{LANGUAGE_NAME[lang]}/{task_id}"
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["task_id"] = task_id
                sample["test_code"] = process_mbpp_test(sample, problems, example_test)
                if sample["test_code"] is None:
                    continue
                if "completion_id" in sample:
                    completion_id_ = sample["completion_id"]
                else:
                    completion_id_ = completion_id[task_id]

                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        print(completion_id)
        print(len(completion_id), len(problems))
        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
    evaluate_pass_at_k = True
    # Calculate pass@k.
    total, correct = [], []
    total_micro, correct_micro = {}, {}
    for result_key in results:
        result = results[result_key]
        passed = [r[1]["passed"] for r in result]
        micro_result_key = result_key
        if isinstance(micro_result_key, int):
            pass
        else:
            if '_' in micro_result_key:
                micro_result_key = micro_result_key[:micro_result_key.find('_')]
        total.append(len(passed))
        correct.append(sum(passed))
        if micro_result_key not in total_micro: total_micro[micro_result_key] = 0
        if micro_result_key not in correct_micro: correct_micro[micro_result_key] = 0
        total_micro[micro_result_key] += len(passed)
        correct_micro[micro_result_key] += sum(passed)
    total = np.array(total)
    correct = np.array(correct)
    total_micro = np.array([t for t in total_micro.values()])
    correct_micro = np.array([c for c in correct_micro.values()])
    if evaluate_pass_at_k:
        print("Total (macro):", np.sum(total))
        print("Correct (macro):", np.sum(correct))
        print("Total (micro):", np.sum(total_micro))
        print("Correct (micro):", np.sum(correct_micro))
        ks = k
        pass_at_k = {f"macropass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        print(pass_at_k)
        pass_at_k_micro = {f"micropass@{k}": estimate_pass_at_k(total_micro, correct_micro, k).mean()
                     for k in ks if (total >= k).all()}
        print(pass_at_k_micro)
    else:
        print("Total (macro):", np.sum(total))
        print("Correct (macro):", np.sum(correct))
        print("Total (micro):", np.sum(total_micro))
        print("Correct (micro):", np.sum(correct_micro))

    print("Writing to: ", out_file)
    if out_file.endswith(".gz"):
        fp = gzip.GzipFile(fileobj=open(out_file, "wb"), mode="wb")
        for res in results.values():
            for r in res:
                fp.write((json.dumps(r[1]) + "\n").encode("utf-8"))
    else:
        fp = open(out_file, 'w')
        for res in results.values():
            for r in res:
                fp.write(json.dumps(r[1]) + "\n")
    fp.close()

    print("Evaluation finished.")


def main():
    fire.Fire(evaluate_functional_correctness)


if __name__ == "__main__":
    sys.exit(main())
