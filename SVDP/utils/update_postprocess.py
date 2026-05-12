import json
import sys

FILE_PATH = "UltraEval/datasets/humaneval/config/humaneval_gen.json"
NEW_VALUE = sys.argv[1]
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    data["postprocess"] = NEW_VALUE
with open(FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)