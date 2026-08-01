import random

from UltraEval.tasks.postprocess import ExactMatchPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQuestion: {data['question']}\nAnswer: "
    correct_answer = data["answer"]
    emp = ExactMatchPost()
    _, processed_correct_answer = emp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
