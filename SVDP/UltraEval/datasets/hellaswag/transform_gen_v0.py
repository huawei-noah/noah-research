import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    options = list(data["target_scores"].keys())
    A, B, C, D = options
    text = f"{data['question']}\nQuestion: Which ending makes the most sense?\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nYou may choose from 'A', 'B', 'C', 'D'.\nAnswer:\n"
    index_of_correct_answer = list(data["target_scores"].values()).index(1)
    answers = ["A", "B", "C", "D"]
    correct_answer = answers[index_of_correct_answer]

    return {"input": text, "output": correct_answer, "processed_output": correct_answer}
