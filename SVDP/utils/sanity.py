import argparse
import logging
logging.basicConfig(level=logging.INFO)

import vllm
logging.info(f"{vllm.__version__=}")

from utils import register_vllm_model, apply_chat_template


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
        # min_tokens=200,
        # max_tokens=200,
        # stop=['\n\n"""', "\n\n\n\n"],
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
        print(100*"=")
        print(f"{generated_text!r}")


if __name__ == "__main__":
    main()
