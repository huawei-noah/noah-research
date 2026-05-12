import json
import os


def transform_entry(data_entry):
    question_stem = data_entry["question"]["stem"]
    answer_key = data_entry["answerKey"]
    target_scores = {
        choice["text"]: int(choice["label"] == answer_key)
        for choice in data_entry["question"]["choices"]
    }

    return {
        "passage": "",
        "question": question_stem,
        "target_scores": target_scores,
        "answer": "",
    }


def convert(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data_entry = json.loads(line.strip())
            transformed_data = transform_entry(data_entry)
            outfile.write(json.dumps(transformed_data, ensure_ascii=False) + "\n")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = "../../RawData/arc-c/ARC-Challenge-Test.jsonl"
    output_path = "./data/arc-c.jsonl"
    input_file_path = os.path.join(script_dir, input_path)
    output_file_path = os.path.join(script_dir, output_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    convert(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
