import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    inp = "题目："
    inp += data["question"] + "\n"

    options = data["target_scores"].keys()

    temp = "{}. "

    for i, opt in enumerate(options):
        inp += temp.format(chr(i + 65)) + opt + "\n"

    inp += "答案是："

    index_of_correct_answer = list(data["target_scores"].values()).index(1)
    processed_correct_answer = correct_answer = f"{chr(65 + index_of_correct_answer)}"

    return {
        "input": inp,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
