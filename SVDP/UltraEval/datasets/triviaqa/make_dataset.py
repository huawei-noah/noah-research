import json
import os


def transform_entry(data_entry):
    return {
        "passage": "",
        "question": data_entry["Question"],
        "target_scores": {},
        "answer": data_entry["Answer"]["NormalizedAliases"],
    }


def convert(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        data = json.load(file)
        for entry in data["Data"]:
            new_entry = transform_entry(entry)
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_paths = [
        "../../RawData/triviaqa/verified-web-dev.json",
        "../../RawData/triviaqa/verified-wikipedia-dev.json",
    ]
    output_paths = ["./data/web.jsonl", "./data/wikipedia.jsonl"]

    for input_path, output_path in zip(input_paths, output_paths):
        input_file_path = os.path.join(script_dir, input_path)
        output_file_path = os.path.join(script_dir, output_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        convert(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
