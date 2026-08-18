import random


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    description = data["text"]
    tests = "\n".join(data["test_list"])

    return {
        "input": f'"""{description}\n{tests}"""',
        "output": data["code"],
        "processed_output": data["code"],
    }
