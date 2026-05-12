import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    choices = []
    answer = None
    for i, (kw, val) in enumerate(data["target_scores"].items()):
        if val > 0:
            answer = i
        choices.append(kw)
    if answer is None:
        raise ValueError("Invalid data `{}`".format(data))

    choice_style = "({})"

    sep_style = " "

    idx_style = "ABCD"

    prompt = "Question\n{question}\n\nOptions\n{options}\n\nAnswer\n"

    options = []
    for i, choice in enumerate(choices):
        options.append(choice_style.format(idx_style[i]) + sep_style + choice)

    answer = choice_style.format(idx_style[answer]) + "\n"

    prompt = prompt.format(question=data["question"], options="\n".join(options))

    return {"input": prompt, "output": answer, "processed_output": answer}
