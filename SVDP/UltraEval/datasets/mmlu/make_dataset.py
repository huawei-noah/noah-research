import csv
import glob
import json
import os


def transform_entry(row):
    question, *choices, answer = row
    target_scores = {
        choice: int((ord(answer) - ord("A")) == idx)
        for idx, choice in enumerate(choices)
    }

    return {
        "passage": "",
        "question": question,
        "target_scores": target_scores,
        "answer": "",
    }


def convert(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        for row in reader:
            transformed_entry = transform_entry(row)
            if len(transformed_entry["target_scores"].keys()) != 4:
                continue
            outfile.write(json.dumps(transformed_entry, ensure_ascii=False) + "\n")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = "../../RawData/mmlu/test/"
    output_path = "./data/"
    input_dir = os.path.join(script_dir, input_path)
    output_dir = os.path.join(script_dir, output_path)

    os.makedirs(output_dir, exist_ok=True)

    for input_file in glob.glob(os.path.join(input_dir, "*_test.csv")):
        base_name = os.path.basename(input_file).rsplit("*_test.csv", 1)[0]
        output_file_path = os.path.join(
            output_dir, f"{base_name.replace('_test.csv', '').replace('_', '-')}.jsonl"
        )
        convert(input_file, output_file_path)


if __name__ == "__main__":
    main()
