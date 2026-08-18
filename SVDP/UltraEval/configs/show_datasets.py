import os
import re
import shutil


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

                    if not config_dict.get(dataset):
                        config_dict[dataset] = {}
                    if not config_dict[dataset].get(task):
                        config_dict[dataset][task] = file_path
                    else:
                        exit(f"error: {dataset}_{task}")

    return config_dict


if __name__ == "__main__":
    config_dict_path = get_task_path("datasets")
    print(f"The number of datasets is: {len(config_dict_path)}")

    max_key_length = max(len(key) for key in config_dict_path.keys())

    print("The following datasets and corresponding tasks are currently supported")
    for key, value in sorted(config_dict_path.items()):
        temp = {
            item.rsplit("-", 1)[0] if item.endswith(("-gen", "-ppl")) else item
            for item in value
        }
        tasks = ", ".join(sorted(temp))

        terminal_width = shutil.get_terminal_size().columns
        max_line_length = terminal_width - max_key_length - 2
        indent = " " * (max_key_length + 2)
        line = ""
        for task in tasks.split(", "):
            if len(line) + len(task) >= max_line_length:
                print(f"{key:<{max_key_length}}: {line.rstrip()}")
                line = task + ", "
                key = " " * max_key_length
            else:
                line += task + ", "

        print(f"{key:<{max_key_length}}: {line.rstrip(', ')}")
