import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

from metrics import get_metric
from metrics.aggregator import get_aggregator
from tasks.instance import Instance
from tasks.postprocess import get_postprocess
from utils import utils
from utils.request import Request


class EvalTask:
    def __init__(
        self,
        task_name: str,
        task_path: str,
        description: str,
        transform_script_path: str,
        num_fewshot: int,
        metric_config: Dict[str, Any],
        sample_config: Dict[str, Any],
        model_postprocess: str,
        task_postprocess: str,
        log_dir: str = "",
        params: str = "",
        limit: int = 1,
        batch_size: int = 1,
    ):
        self.task_name = task_name
        self._task_path = task_path

        self.dataset = []
        with open(task_path, "r", encoding="utf-8") as file:
            for line in file:
                self.dataset.append(Instance(json.loads(line.strip())))

        self.description = description
        self._transform_script_path = transform_script_path
        self._transform_func = utils.import_function_from_path(
            transform_script_path, "transform"
        )
        self.num_fewshot = num_fewshot
        self._metric_config = deepcopy(metric_config)
        self.construct_metrics(metric_config)

        self.sample_config = sample_config  # sampling的配置
        _params = sample_config["params"] if sample_config["params"] else params
        with open(_params, "r", encoding="utf-8") as f:
            self.sample_args = json.load(f)
        self.sample_config["args"] = self.sample_args

        self._model_postprocess = model_postprocess
        self._task_postprocess = task_postprocess

        try:
            self.model_postprocess = get_postprocess(
                model_postprocess
            )()  # 对输出结果进行处理，比如截断等
        except:
            exit(f"{model_postprocess} not exists!")
        self.task_postprocess = (
            get_postprocess(task_postprocess)() if task_postprocess else ""
        )

        self.log_dir = log_dir
        self.limit = limit if limit is not None else len(self.dataset)
        self.batch_size = (
            self.sample_args["batch_size"]
            if self.sample_args.get("batch_size")
            else batch_size
        )

    def construct_metrics(self, metrics_cfg):
        for metric_name, metric_cfg in metrics_cfg.items():
            eval_metric = metric_cfg["evaluation"]
            em_name = eval_metric["type"].lower()
            eval_metric.pop("type")
            metric_cfg["evaluation"] = get_metric(em_name)(**eval_metric)
            if "aggregation" in metric_cfg:
                aggregator_metric = metric_cfg["aggregation"]
                am_name = aggregator_metric["type"].lower()
                aggregator_metric.pop("type")
                metric_cfg["aggregation"] = get_aggregator(am_name)(**aggregator_metric)
            else:
                metric_cfg["aggregation"] = None
        self.metrics = metrics_cfg

    def run(self, model):
        if model.concurrency < 2:
            for request in self.yield_batch_requests():
                generate = getattr(model, request.request_type)
                result = generate(request)

                raw_outputs, processed_outputs = self.model_postprocess(result, request)
                if self.task_postprocess:
                    raw_outputs, processed_outputs = self.task_postprocess(
                        raw_outputs, processed_outputs
                    )

                for i in range(len(request.raw_example)):
                    request.raw_example[i].raw_outputs.append(raw_outputs[i])
                    request.raw_example[i].processed_outputs.append(
                        processed_outputs[i]
                    )
                    request.raw_example[i].ground_truth = request.instances[i][
                        "processed_output"
                    ]
                    request.raw_example[i].prompt_inputs.append(
                        request.instances[i]["input"]
                    )

        else:
            requests = [request for request in self.yield_batch_requests()]

            generate = getattr(model, requests[0].request_type)
            result = generate(requests)
            if len(requests) != len(result):
                result = [item for sublist in result for item in sublist]
                result = [
                    result[i : i + self.batch_size]
                    for i in range(0, len(result), self.batch_size)
                ]
            assert len(requests) == len(result)
            for j in range(len(requests)):
                req = requests[j]
                res = result[j]
                raw_outputs, processed_outputs = self.model_postprocess(res, req)
                if self.task_postprocess:
                    raw_outputs, processed_outputs = self.task_postprocess(
                        raw_outputs, processed_outputs
                    )

                for i in range(len(req.raw_example)):
                    req.raw_example[i].raw_outputs.append(raw_outputs[i])
                    req.raw_example[i].processed_outputs.append(processed_outputs[i])
                    req.raw_example[i].ground_truth = req.instances[i][
                        "processed_output"
                    ]
                    req.raw_example[i].prompt_inputs.append(req.instances[i]["input"])

        self.evaluate()
        self.finish()

    def yield_batch_requests(self):
        rnd = random.Random(42)
        sampling_num = self.sample_args.get("sampling_num", 1)
        raw_input = []
        for ins in self.dataset[: self.limit]:
            raw_input.extend([ins for i in range(sampling_num)])
        prompt_input = [
            self.construct_input(item.data, self.num_fewshot, rnd, self.description)
            for item in raw_input
        ]

        batch_size = self.batch_size

        for i in tqdm(range(0, len(prompt_input), batch_size)):
            request = Request(
                request_type=self.sample_config["method"],
                instances=prompt_input[i : i + batch_size],
                params=self.sample_args,
                raw_example=raw_input[i : i + batch_size],
            )
            yield request

    def evaluate(self):
        for ins in self.dataset[: self.limit]:
            for metric_name, metric in self.metrics.items():
                ins.eval_results[metric_name] = metric["evaluation"](
                    ins.data, ins.ground_truth, ins.processed_outputs
                )
                if metric["aggregation"] is not None:
                    temp = metric["aggregation"](ins.eval_results[metric_name])
                    if isinstance(temp, dict):
                        ins.metrics.update(temp)
                    else:
                        ins.metrics[metric_name] = temp
                else:
                    ins.metrics[metric_name] = ins.eval_results[metric_name]

    def finish(self):
        save_task_path = os.path.join(self.log_dir, self.task_name)
        os.makedirs(save_task_path, exist_ok=True)

        self.gathered_metrics = defaultdict(list)
        for ins in self.dataset[: self.limit]:
            for metric in ins.metrics:
                self.gathered_metrics[metric].append(ins.metrics[metric])
        print(
            "<<{}>> Gathered metrics are: {}".format(
                self.task_name, self.gathered_metrics
            )
        )

        self.final_metrics = {}
        for metric in self.gathered_metrics:
            self.final_metrics[metric] = np.array(self.gathered_metrics[metric]).mean()

        print("<<{}>> Final Metric is: {}".format(self.task_name, self.final_metrics))

        dump_data = {
            "task_name": self.task_name,
            "instance_result": self.gathered_metrics,
            "overall_result": self.final_metrics,
        }

        with open(
            os.path.join(save_task_path, "final_metrics.json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(dump_data, fout, indent=4, ensure_ascii=False)

        config_data = {
            "task_name": self.task_name,
            "path": self._task_path,
            "description": self.description,
            "transform": self._transform_script_path,
            "fewshot": self.num_fewshot,
            "batch_size": self.batch_size,
            "generate": self.sample_config,
            "model_postprocess": self._model_postprocess,
            "task_postprocess": self._task_postprocess,
            "metric": self._metric_config,
            "log_dir": self.log_dir,
        }

        with open(
            os.path.join(save_task_path, "config.json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(config_data, fout, indent=4, ensure_ascii=False)

    def construct_input(
        self,
        doc: Dict[str, Any],
        num_fewshot: int,
        rnd: Optional[random.Random] = None,
        description: Optional[str] = None,
    ):
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = rnd.sample(self.dataset, num_fewshot + 1)

            fewshotex = [x.data for x in fewshotex if x.data != doc][:num_fewshot]

            rnd_state = rnd.getstate()

            transformed_docs = []
            for d in fewshotex:
                data = self._transform_func(d, num_fewshot, rnd, self.task_name)
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

        example = self._transform_func(doc, num_fewshot, rnd, self.task_name)
        return {
            "input": description + labeled_examples + example["input"],
            "output": example["output"],
            "processed_output": example["processed_output"],
        }
