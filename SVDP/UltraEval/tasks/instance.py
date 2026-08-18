import json
import os
from collections import defaultdict


class Instance:
    """all information of a test instance.
    Include:
    1.  the original data
    2.  the full input to model
    3.  all the output for the instance
    4.  all the tested result of every instance
    """

    def __init__(self, data):
        self.data = data
        self.ground_truth = ""
        self.prompt_inputs = []
        self.raw_outputs = []
        self.processed_outputs = []
        self.eval_results = defaultdict(list)
        self.metrics = defaultdict(None)

    def dump(self):
        instance_data = {
            "data": self.data,
            "ground_truth": self.ground_truth,
            "prompt_inputs": self.prompt_inputs,
            "raw_outputs": self.raw_outputs,
            "processed_outputs": self.processed_outputs,
            "eval_results": self.eval_results,
            "metrics": self.metrics,
        }
        
        return json.dumps(instance_data, ensure_ascii=False)

        # with open(
        #     os.path.join(file_path, "instance.jsonl"), "a", encoding="utf-8"
        # ) as jsonl_file:
        #     jsonl_file.write(json.dumps(instance_data, ensure_ascii=False) + "\n")
