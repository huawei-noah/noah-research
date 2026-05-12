import sys

sys.path.append("..")
import argparse
import importlib.util
import json
import os
import random

from tasks.instance import Instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)  # 配置文件路径
    return parser.parse_args()


def fewshot_examples(k, rnd, dataset):
    return rnd.sample(dataset, k + 1)


def construct_input(
    doc,
    num_fewshot,
    rnd,
    trans_func,
    dataset,
    dataset_name,
    description,
):
    """Returns a fewshot context string that is made up of a prepended description
    (if provided), the `num_fewshot` number of examples, and an appended prompt example.
    """
    assert (
        rnd is not None
    ), "A `random.Random` generator argument must be provided to `rnd`"

    if num_fewshot == 0:
        labeled_examples = ""
    else:
        fewshotex = fewshot_examples(k=num_fewshot + 1, rnd=rnd, dataset=dataset)

        fewshotex = [x.data for x in fewshotex if x.data != doc][:num_fewshot]
        rnd_state = rnd.getstate()

        transformed_docs = []
        for d in fewshotex:
            data = trans_func(d, num_fewshot, rnd, dataset_name)
            if isinstance(data["output"], list):
                transformed_docs.append(data["input"] + data["output"][0])
            elif isinstance(data["output"], str):
                transformed_docs.append(data["input"] + data["output"])
            else:
                raise ValueError(
                    "Invalid output data type. Please notice the realted transform.py"
                )
            rnd.setstate(rnd_state)

        labeled_examples = "\n\n".join(transformed_docs) + "\n\n"

    example = trans_func(doc, num_fewshot, rnd, dataset_name)

    return {
        "input": description + labeled_examples + example["input"],
        "output": example["processed_output"],
    }


def main():
    args = parse_args()
    random.seed(42)

    with open(args.config, "r", encoding="utf-8") as f:
        f_config = json.load(f)

    rnd = random.Random()
    print("Selected UltraEval Dataset Path: ", f_config["path"])

    num_fewshot = f_config["fewshot"]
    path_parts = args.config.split("/")
    dataset_part = path_parts[-3]

    filename_part = os.path.splitext(path_parts[-1])[0]
    filename_parts = filename_part.split("_")
    config_part = filename_parts[-1]

    dataset_name = f_config["task_name"]

    dataset_name = f"{dataset_part}_{dataset_name}_{config_part}"
    description = f_config["description"] + "\n\n" if f_config["description"] else ""
    dataset = []
    with open(f_config["path"], "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            dataset.append(Instance(data))

    spec = importlib.util.spec_from_file_location("transform", f_config["transform"])
    transform_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform_module)

    trans_func = transform_module.transform

    raw_input = []
    for ins in dataset:
        raw_input.extend([ins])

    prompt_input = [
        construct_input(
            item.data, num_fewshot, rnd, trans_func, dataset, dataset_name, description
        )
        for item in raw_input
    ]

    for i, prompt in enumerate(prompt_input):
        print("\033[92m" + ("=" * 8), "example {}".format(i + 1), ("=" * 8) + "\033[0m")
        print(prompt["input"], end="")
        print("\033[91m{}\033[0m".format(prompt["output"]))
        if input("\033[92m" + ("=" * 8) + "\033[0m").strip() in ["q", "quit", "exit"]:
            break


if __name__ == "__main__":
    main()
