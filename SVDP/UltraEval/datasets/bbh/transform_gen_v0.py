import random

from UltraEval.tasks.postprocess import ExactMatchPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    gen_cases = [
        "dyck_languages",
        "object_counting",
        "multistep_arithmetic_two",
        "word_sorting",
    ]

    question = f"Question:\n{data['question']}\n"
    answer_prompt = f"Answer:\n"
    true_dataset_name = dataset_name.split("_")[1].replace("-", "_")
    if true_dataset_name in gen_cases:
        text = question + answer_prompt
        correct_answer = data["answer"]
        # if true_dataset_name == "dyck_languages":
        #     processed_correct_answer = correct_answer.replace(" ", "")
        # else:
        processed_correct_answer = correct_answer
        # emp = ExactMatchPost()
        # _, processed_correct_answer = emp([], correct_answer)
    else:
        instruction = f"Requirement:\nChoose and respond with the letter of the correct answer, including the parentheses.\n"
        options = "Options:\n"
        for idx, item in enumerate(data["target_scores"].keys()):
            options += f"({chr(65 + idx)}) {item}\n"
        text = question + instruction + options + answer_prompt
        index_of_correct_answer = list(data["target_scores"].values()).index(1)
        processed_correct_answer = (
            correct_answer
        ) = f"({chr(65 + index_of_correct_answer)})"

    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
