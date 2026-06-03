# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging

# basicConfig must run before vllm is imported so vllm's import-time logging
# uses our configuration; hence the deliberate non-top-level imports below.
logging.basicConfig(level=logging.INFO)

import vllm  # noqa: E402

logging.info(f"{vllm.__version__=}")

from utils import apply_chat_template, register_vllm_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for implementations")

    parser.add_argument("--vllm_module_path", help="Path to vllm implementation module", required=True)
    parser.add_argument("--model_path", help="Path to model", required=True)
    parser.add_argument("--start_prompt", help="Prompt", required=True)
    parser.add_argument("--temperature", type=float, required=False, default=0.0)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    register_vllm_model(args.vllm_module_path, args.model_path)

    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
    )
    model = vllm.LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="half",
        max_model_len=1000,
        device="cuda:0",
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        max_num_batched_tokens=1000,
        max_num_seqs=1,
        enforce_eager=True,
    )
    prompts = [args.start_prompt]

    tokenizer = model.get_tokenizer()
    prompts = apply_chat_template(prompts, tokenizer)

    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(prompt)
        print(100 * "=")
        print(f"{generated_text!r}")


if __name__ == "__main__":
    main()
