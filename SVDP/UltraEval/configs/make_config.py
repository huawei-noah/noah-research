import argparse
import json
import os
import re


def get_task_path(base_dir: str = "datasets"):
    config_dict = {}

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            config_dir = os.path.join(dir_path, "config")
            if os.path.isdir(config_dir):
                for file_name in os.listdir(config_dir):
                    file_path = os.path.join(config_dir, file_name)
                    rel_path = os.path.relpath(file_path, base_dir)

                    dataset, task = (
                        rel_path.replace(".json", "").lower().split("/")[::2]
                    )
                    task = "-".join(re.split("[-_]", task))
                    if task.endswith("-gen"):
                        task = task[:-4] + "_gen"
                    elif task.endswith("-ppl"):
                        task = task[:-4] + "_ppl"

                    if not config_dict.get(dataset):
                        config_dict[dataset] = {}
                    if not config_dict[dataset].get(task):
                        config_dict[dataset][task] = file_path
                    else:
                        exit(f"error: {dataset}-{task}")

    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Name of datasets you select",
        required=True,
    )
    parser.add_argument(
        "--tasks", type=str, default="", help="name of tasks you select", required=False
    )
    parser.add_argument("--method", type=str, default="gen", required=False)
    parser.add_argument("--save", type=str, default="eval_config.json", required=False)
    args = parser.parse_args()

    config_dict_path = get_task_path("datasets")
    print(f"The num of datasets is: {len(config_dict_path)}")

    datasets = [
        item.strip() for item in args.datasets.lower().split(",") if item.strip()
    ]
    tasks = [item.strip() for item in args.tasks.lower().split(",") if item.strip()]
    method = args.method.lower()

    config_dic = dict()
    if not tasks:
        if "all" in datasets:
            for key, value in config_dict_path.items():
                config_dic[key] = list(value.keys())
        else:
            for key in datasets:
                if not config_dict_path.get(key):
                    exit(f"error: {key} not in dataset list!")
                config_dic[key] = list(config_dict_path[key].keys())
    else:
        if len(datasets) != 1:
            exit("error: datasets must be one!")
        else:
            if not config_dict_path.get(datasets[0]):
                exit(f"error: {datasets[0]} not in dataset list!")
            key = datasets[0]
            values = list(config_dict_path[key].keys())
            config_dic[key] = [
                f"{t}_{method}" for t in tasks if f"{t}_{method}" in values
            ]  # 需要确保每个模型都添加了 gen/ppl
            if config_dic[key] == [] or len(config_dic[key]) != len(tasks):
                exit(f"error: {tasks} not in {datasets}!")

    if len(tasks):
        print(f"Tasks for evaluation: {','.join(tasks)}")
    else:
        print(f"Datasets for evaluation: {','.join(config_dic.keys())}")

    eval_config = []
    for key, value in config_dic.items():
        for task in value:
            path = config_dict_path[key][task]
            with open(path, "r", encoding="utf-8") as f:
                tdict = json.load(f)
            name1 = key
            name2 = "-".join(re.split("[-_]", tdict["task_name"]))
            name3 = "gen" if tdict["generate"]["method"] == "generate" else "ppl"
            tdict["task_name"] = name1 + "_" + name2 + "_" + name3
            if name3 != args.method:
                continue
            eval_config.append(tdict)

    print(
        f"The number of selected datasets is {len(config_dic.keys())}; the number of selected tasks is {len(eval_config)}."
    )
    with open(f"configs/{args.save}", "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=4, ensure_ascii=False)
    print("Results have been saved！")
