import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    inp = ""
    inp += data["question"] + "\n"

    options = [j for j in data["target_scores"]]

    temp = "{}. "
    ans = 0
    for i, opt in enumerate(options):
        inp += temp.format(chr(i + 65)) + opt + "\n"
        if data["target_scores"][opt] == 1:
            ans = i

    inp += "答案："

    out = chr(ans + 65)

    return {"input": inp, "output": out, "processed_output": out}
