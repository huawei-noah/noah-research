import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

import requests
from tqdm import tqdm

from utils import utils

data_prompt = {
    "params": {},
    "instances": [],
}


def run_thread_pool_sub(target, url: str, data, max_work_count: int):
    with tqdm(total=len(data)) as pbar:
        with ThreadPoolExecutor(max_workers=max_work_count) as t:
            futures = [t.submit(target, url, i, data[i]) for i in range(len(data))]
            for future in as_completed(futures):
                pbar.update(1)
                yield future.result()


def _post_request(url, data):
    data_prompt["instances"] = data
    s = json.dumps(data_prompt)
    headers = {"Content-Type": "application/json"}

    backoff_time = 2
    backoff_count = 50
    i = 0
    while i < backoff_count:
        try:
            return requests.post(url, data=s, headers=headers).json()
        except Exception:
            time.sleep(backoff_time)
            backoff_time *= 1.5
            i += 1
    return "request time out"


def thread_function(url: str, idx: int, chk: List[Any]):
    lp = _post_request(url, chk)
    return idx, lp


class GeneralModel:
    url = ""
    concurrency = 1

    def __init__(self, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        self.url = args.get("url", "")
        self.concurrency = args.get("concurrency", 1)

    def make_request_instance(self, input: str, output: str) -> Any:
        return input + output

    def loglikelihood(self, request):
        if self.concurrency < 2:
            data_prompt["params"].update(request.params)
            chunk_req = []
            res = []
            for instance, raw_example in zip(request.instances, request.raw_example):
                input_string = instance["input"]
                options = raw_example.data["target_scores"].keys()
                combined_inputs = [
                    self.make_request_instance(input_string, option)
                    for option in options
                ]
                chunk_req.append(combined_inputs)
            for i in range(len(chunk_req)):
                result = _post_request(self.url, chunk_req[i])
                res.append(result)
        else:
            data_prompt["params"].update(request[0].params)
            chunk_req = []
            res = []
            for single_request in request:
                # single_chunk_req = []
                for instance, raw_example in zip(
                    single_request.instances, single_request.raw_example
                ):
                    input_string = instance["input"]
                    options = raw_example.data["target_scores"].keys()
                    combined_inputs = [
                        self.make_request_instance(input_string, option)
                        for option in options
                    ]

                    chunk_req.append(combined_inputs)

            result_map = {}
            for i, result_list in run_thread_pool_sub(
                thread_function, self.url, chunk_req, self.concurrency
            ):
                if i not in result_map:
                    result_map[i] = []
                result_map[i].append(result_list)
            for i in range(len(chunk_req)):
                res.append(result_map[i])
        return res

    def generate(self, request):
        if self.concurrency < 2:
            data_prompt["params"].update(request.params)

            data = [
                self.make_request_instance(req["input"], "")
                for req in request.instances
            ]
            result = _post_request(self.url, data)
        else:
            data_prompt["params"].update(request[0].params)

            data = []
            for ins in request:
                data.append(
                    [
                        self.make_request_instance(req["input"], "")
                        for req in ins.instances
                    ]
                )
            result_map = {}
            for i, result_list in run_thread_pool_sub(
                thread_function, self.url, data, self.concurrency
            ):
                result_map[i] = result_list
            result = []
            for i in range(len(data)):
                result.append(result_map[i])

        return result
