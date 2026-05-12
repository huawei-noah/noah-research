import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    return {"input": data["prompt"].strip(), "output": "", "processed_output": ""}
