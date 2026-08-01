import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    context = f"Context:\n {data['activity_label']}: {data['question']}\n"
    question = f"Question:\nWhich ending makes the most sense?\n"
    instruction = f"Requirement:\nChoose and respond with the letter of the correct answer, including the parentheses.\n"
    options = "Options:\n"
    for idx, item in enumerate(data["target_scores"].keys()):
        options += f"({chr(65 + idx)}) {item}\n"
    answer_prompt = f"Answer:\n"
    text = context + question + instruction + options + answer_prompt
    index_of_correct_answer = list(data["target_scores"].values()).index(1)
    processed_correct_answer = correct_answer = f"({chr(65 + index_of_correct_answer)})"

    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
