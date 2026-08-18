# This file is based on vllm profiling file.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import os
import sys
from argparse import RawTextHelpFormatter
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tqdm
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.profiler.layerwise_profile import LayerwiseProfileResults  # layerwise_profile
from vllm.utils import FlexibleArgumentParser

from utils import apply_chat_template

PROMPT_LEN_DEFAULT = 256


class layerwise_profile(profile):
    def __init__(self, num_running_seqs: int | None = None):
        """
        layerwise profile constructor.

        Args:
            num_running_seqs (Optional[int], optional): When given,
            num_running_seqs will be passed to LayerProfileResults for metadata
            update. Defaults to None.
        """
        super().__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=True,
            with_modules=True,
            experimental_config=_ExperimentalConfig(verbose=True),
        )

        self.num_running_seqs = num_running_seqs

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.results = LayerwiseProfileResults(self.profiler.kineto_results, num_running_seqs=self.num_running_seqs)


@dataclass
class ProfileContext:
    engine_args: EngineArgs
    generation_length: int
    vllm_model_modulename: str | None = None  # name of module with model code to profile
    num_prompts: int | None = None

    # The profiler can run in 2 modes:
    # 1. Profiling on random prompts of fixed length
    prompt_len: int | None = None
    # 2. Profiling on specified dataset (.json)
    dataset_path: str | None = None


def report_metrics(prefill_results, decode_results_list: list, output_file: str, metadata: dict):
    prefill_stats = prefill_results.convert_stats_to_dict()["summary_stats"]
    TTFT = sum(stage["entry"]["cuda_time_us"] for stage in prefill_stats)
    TPOT = []
    TPOT_MLP = []
    for decode_result in decode_results_list:
        decode_stats = decode_result.convert_stats_to_dict()["summary_stats"]
        TPOT.append(sum(stage["entry"]["cuda_time_us"] for stage in decode_stats))
        try:
            mlp_entry = decode_stats[0]["children"][1]["children"][2]["entry"]
            if "MLP" not in mlp_entry["name"]:
                mlp_entry = decode_stats[0]["children"][1]["children"][4]["entry"]
        except Exception as e:
            raise RuntimeError(
                "Could not locate the MLP entry in the profiler tree. The expected tree structure may have changed."
            ) from e
        assert "MLP" in mlp_entry["name"]
        TPOT_MLP.append(mlp_entry["cuda_time_us"])

    try:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(
        {
            "TTFT": TTFT,
            "TPOT": np.mean(TPOT),
            "TPOT_MLP": np.mean(TPOT_MLP),
            "E2E": TTFT + np.sum(TPOT),
        }
        | metadata
    )
    dir_path = os.path.dirname(output_file)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_promt_token_ids(llm, context):
    from_dataset: bool = context.dataset_path is not None
    if from_dataset:
        with open(context.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]

        tokenizer = llm.llm_engine.tokenizer
        if llm.get_tokenizer().chat_template is not None:
            prompts = apply_chat_template(prompts, llm.get_tokenizer())
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts[: context.num_prompts]]
    else:
        prompt_token_ids = [
            torch.randint(llm.get_tokenizer().vocab_size, size=(context.prompt_len,)).tolist()
            for _ in range(context.num_prompts)
        ]
    return prompt_token_ids


def validate_params(llm, context):
    prompt_len = context.prompt_len
    max_model_len = llm.llm_engine.model_config.max_model_len

    max_output_len = context.generation_length

    print(
        "llm.llm_engine.model_config.max_model_len: ",
        llm.llm_engine.model_config.max_model_len,
    )
    if prompt_len is not None and prompt_len + max_output_len > llm.llm_engine.model_config.max_model_len:
        print(
            f"ERROR: chosen prompt_len + max_output_len ({prompt_len} + "
            f"{max_output_len} = {prompt_len + max_output_len}) is larger "
            f"than the model's max_model_len ({max_model_len}), please "
            f"choose a smaller prompt_len or max_output_len, or increase "
            f"--max-model-len"
        )
        sys.exit(-1)


def add_request(llm, sampling_params, prompt_token_ids, request_id):
    llm.llm_engine.add_request(
        request_id=request_id,
        prompt={"prompt_token_ids": prompt_token_ids},
        params=sampling_params,
    )


def abort_request(llm, request_id):
    llm.llm_engine.abort_request(request_id)


def run_profile(context: ProfileContext, metrics_output: str | None, json_output: str | None):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=context.generation_length + 1,  # including prefill step token
        min_tokens=context.generation_length + 1,  # including prefill step token
        ignore_eos=True,
    )

    def register(vllm_module_path: str, model_path: str) -> None:
        import importlib
        import sys

        from vllm import ModelRegistry

        module_path = vllm_module_path
        module_name = os.path.splitext(os.path.basename(module_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        import json

        with open(model_path + "/config.json", encoding="utf-8") as file:
            data = json.load(file)
            class_name = data["architectures"][0]
        ModelRegistry.register_model(class_name, getattr(module, class_name))

    if context.vllm_model_modulename is not None:
        register(context.vllm_model_modulename, context.engine_args.model)

    # Create LLM
    llm = LLM(**asdict(context.engine_args))
    prompt_len = context.prompt_len

    validate_params(llm, context)

    prompts_token_ids = get_promt_token_ids(llm, context)

    # Warm up run
    request_name = "42"
    print("Warm up run ...")
    add_request(llm, sampling_params, prompts_token_ids[0], request_name)
    llm.llm_engine.step()  # Prefill
    llm.llm_engine.step()  # Decode
    abort_request(llm, request_name)

    if metrics_output and os.path.isfile(metrics_output):
        directory = os.path.dirname(metrics_output)
        os.makedirs(directory, exist_ok=True)
        # Clear file contents
        with open(metrics_output, "w"):
            pass

    for i in range(len(prompts_token_ids)):
        print("Profile run ...")
        add_request(llm, sampling_params, prompts_token_ids[i], request_name)
        with layerwise_profile() as prefill_prof:
            llm.llm_engine.step()  # First step is prefill

        decode_profs = []
        for _ in tqdm.tqdm(range(context.generation_length)):
            num_running_seqs = llm.llm_engine.scheduler[0].get_num_unfinished_seq_groups()
            with layerwise_profile(num_running_seqs=num_running_seqs) as decode_prof:
                llm.llm_engine.step()
            decode_profs.append(decode_prof)

        decode_results_list = [prof.results for prof in decode_profs]
        prefill_results = prefill_prof.results

        LINE_WIDTH = 80
        print()
        print("=" * LINE_WIDTH)
        print(f"= Prefill Summary Table (prompt_len={prompt_len})")
        print("=" * LINE_WIDTH)
        print()
        prefill_results.print_summary_table()

        print()
        print("=" * LINE_WIDTH)
        print(f"= First Decode Step Summary Table (prompt_len={prompt_len})")
        print("=" * LINE_WIDTH)
        print()
        decode_results_list[0].print_summary_table()
        if metrics_output:
            metadata = {}
            report_metrics(
                prefill_results=prefill_results,
                decode_results_list=decode_results_list,
                output_file=metrics_output,
                metadata=metadata,
            )

        if json_output:
            cuda_devices = [torch.cuda.get_device_properties(dev_idx) for dev_idx in range(torch.cuda.device_count())]

            json_dict = {
                "context": {
                    "python_version": f"{sys.version}",
                    "torch_version": f"{torch.__version__}",
                    "torch_cuda_version": f"{torch.version.cuda}",
                    "cuda_devices": f"{cuda_devices}",
                    **asdict(context),
                },
                "prefill": prefill_results.convert_stats_to_dict(),
            }

            for idx, dr in enumerate(decode_results_list):
                json_dict[f"decode_{idx + 1}"] = dr.convert_stats_to_dict()

            # Add .json to json_output filename if it doesn't exist already.
            json_output_file = json_output if json_output.endswith(".json") else json_output + ".json"
            with open(json_output_file, "w+") as f:
                json.dump(json_dict, f, indent=2)

        abort_request(llm, request_name)


def parse_args():
    parser = FlexibleArgumentParser(
        description="""
Profile a model
""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Export metrics as a json file. This should be the filename with '.json'",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export profile stats as a json file. This should be the filename with '.json'",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="input prompt file",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=PROMPT_LEN_DEFAULT,
        help=f"Length of the random prompt to use when profiling, all batched "
        f"requests use the same prompt_len, default={PROMPT_LEN_DEFAULT}",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts, default=1",
    )
    parser.add_argument(
        "--vllm-model-modulename",
        type=str,
        default=None,
        help="module with vllm model code to profile",
    )
    parser.add_argument(
        "--generation-length",
        type=int,
        default=1,
        help="Number of prompts, default=1",
    )

    EngineArgs.add_cli_args(parser)

    return parser.parse_args()


def main(args):
    context = ProfileContext(
        engine_args=EngineArgs.from_cli_args(args),
        **{k: v for k, v in vars(args).items() if k in inspect.signature(ProfileContext).parameters},
    )
    run_profile(context, metrics_output=args.metrics, json_output=args.json)


if __name__ == "__main__":
    args = parse_args()
    main(args)
