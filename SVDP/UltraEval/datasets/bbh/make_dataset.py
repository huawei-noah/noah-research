import glob
import json
import os


def extract_options(input_str):
    options = {}
    option_order = []
    for line in input_str.split("\n"):
        if line.startswith("(") and ")" in line:
            key, _, value = line.partition(") ")
            options[value.strip()] = 0
            option_order.append(value.strip())
    return options, option_order


def transform_entry(example, file_name):
    target_scores, option_order = extract_options(example["input"])
    special_cases = [
        "boolean_expressions.json",
        "causal_judgement.json",
        "navigate.json",
        "sports_understanding.json",
        "web_of_lies.json",
        "formal_fallacies.json",
    ]
    gen_cases = [
        "dyck_languages.json",
        "object_counting.json",
        "multistep_arithmetic_two.json",
        "word_sorting.json",
    ]

    if file_name in special_cases:
        if file_name == "formal_fallacies.json":
            target_scores = {"valid": 0, "invalid": 0}
        elif file_name == "boolean_expressions.json":
            target_scores = {"True": 0, "False": 0}
        else:
            target_scores = {"Yes": 0, "No": 0}
        target_scores[example["target"]] = 1
    elif file_name not in gen_cases:
        target_letter = example["target"][1:-1]
        # fix dataset error in movie_recommendaiton.json
        if example["target"] == "Monsters, Inc":
            target_letter = "A"
        # fix dataset error in ruin_names.json
        if (
            example["target"] == "dearth, wind, & fire"
            or example["target"] == "rita, sue and bob poo"
        ):
            return None
        target_index = ord(target_letter) - ord("A")
        correct_option = option_order[target_index]
        target_scores[correct_option] = 1
    else:
        target_scores = {}

    return {
        "passage": "",
        "question": example["input"],
        "target_scores": target_scores,
        "answer": example["target"]
        if file_name
        in [
            "dyck_languages.json",
            "object_counting.json",
            "multistep_arithmetic_two.json",
            "word_sorting.json",
        ]
        else "",
    }


def convert(input_file_path, output_file_path, file_name):
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for example in data["examples"]:
                new_entry = transform_entry(example, file_name)
                if new_entry:
                    outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_dir = os.path.join(script_dir, "../../RawData/bbh")
    output_dir = os.path.join(script_dir, "./data")

    os.makedirs(output_dir, exist_ok=True)

    for input_file in glob.glob(input_dir + "/*.json"):
        file_name = os.path.basename(input_file)
        output_file = os.path.join(
            output_dir, file_name.replace(".json", ".jsonl").replace("_", "-")
        )
        convert(input_file, output_file, file_name)


if __name__ == "__main__":
    main()
