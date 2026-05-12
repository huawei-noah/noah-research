import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from openai import OpenAI
from tqdm import tqdm

from utils import utils
from utils.request import Request

client = OpenAI(api_key="your key", organization="your org")


def run_thread_pool_sub(post_method, url: str, reqs, max_work_count: int):
    with ThreadPoolExecutor(max_workers=max_work_count) as t, tqdm(
        total=len(reqs)
    ) as pbar:
        futures = [t.submit(post_method, url, i, reqs[i]) for i in range(len(reqs))]
        for future in as_completed(futures):
            pbar.update(1)
            yield future.result()


def _post_request(sys_prompt, content, backoff_time=20, backoff_count=50, **kwargs):
    req_json = {
        "model": kwargs["model"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ],
    }

    resp = "### RESPONSE NOT GOT ###"
    i = 0
    while i < backoff_count:
        try:
            resp = client.chat.completions.create(**req_json)
            content = resp.choices[0].message.content
            return content
        except Exception as e:
            time.sleep(backoff_time)
            i += 1
    return "### MAX RETRIES EXCEEED ###"


class OPENAI_API_MODEL:
    """https://platform.openai.com/docs/models/gpt-3-5 (updated on 20231030)"""

    MAX_TOKENS_MAP = {
        "gpt-3.5-turbo": 4097,
        "gpt-4": 8192,
    }

    DEFAULT_PROMPT = "you are a helpful AI assistant. please answer questions."

    def __init__(
        self,
        model,
        concurrency=1,
        max_retries=10,
        backoff_time=1.0,
        sys_prompt=None,
        api_key_path=None,
    ):
        MODEL_LIST = self.MAX_TOKENS_MAP.keys()
        assert model in MODEL_LIST, f"invalid model name: {model}"
        self._model = model
        self.concurrency = concurrency
        self._max_retries = max_retries
        self._backoff_time = backoff_time
        self._sys_prompt = sys_prompt if sys_prompt else self.DEFAULT_PROMPT
        self._api_key_path = api_key_path
        print(f"model: {self._model}")

    def make_request_instance(self, input: str, output: str) -> Any:
        return input + output

    def generate(self, request: Request):
        data = {"params": {"model": self._model}}
        data.update(request.params)
        data["instances"] = [
            self.make_request_instance(ins["input"], "") for ins in request.instances
        ]

        results: List[str] = []
        for content in data["instances"]:
            print(
                "Request:\n" + json.dumps(content, indent=4, ensure_ascii=False) + "\n"
            )
            output = _post_request(self._sys_prompt, content, **data["params"])
            from pprint import pprint

            pprint(output)

            print("Result:\n" + json.dumps(output, indent=4, ensure_ascii=False) + "\n")

            logging.info("Output:\n" + output)
            logging.info("*-*-" * 20)

            results.append(output)

        return results

    def loglikelihood(self, request):
        raise NotImplementedError


class GPT3_5(OPENAI_API_MODEL):
    def __init__(self, kwargs):
        args = utils.simple_parse_args_string(kwargs)
        args["model"] = "gpt-3.5-turbo"
        super().__init__(**args)


class GPT4(OPENAI_API_MODEL):
    def __init__(self, kwargs):
        args = utils.simple_parse_args_string(kwargs)
        args["model"] = "gpt-4"
        super().__init__(**args)
