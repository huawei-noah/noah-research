import argparse
import os

from flask import Flask, jsonify, request
from vllm import LLM, SamplingParams

"""
reference:https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Model name on hugginface")
parser.add_argument("--module_path", type=str, help="Module Path")
parser.add_argument("--port", type=int, default=5002, help="the port")
parser.add_argument("--use_chat_template", action='store_true', help='Enable verbose output')
args = parser.parse_args()



def register():
    from vllm import ModelRegistry
    import importlib
    import sys

    module_path = args.module_path 
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    import json
    with open(args.model_name+"/config.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        class_name = data["architectures"][0]
    ModelRegistry.register_model(class_name, getattr(module, class_name))
    # ModelRegistry.register_model("SparseLlamaForCausalLM", module.SparseLlamaForCausalLM)

register()

TRUNCATION_LENGTH=2000

llm = LLM(
    model=args.model_name,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="half",
    gpu_memory_utilization=0.92,
    max_model_len=2100,
    max_num_batched_tokens=2100, 
    max_num_seqs=1,
    enforce_eager=True
)

# 模型的模型参数
params_dict = {
    "n": 1,
    "best_of": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
    "use_beam_search": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 3000,
}

app = Flask(__name__)


def truncate_prompts(prompts: list[str], tokenizer, max_length: int):
    trimmed_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
            print(f"will truncate seq to {max_length} tokens")
        trimmed_prompt = tokenizer.decode(tokens)
        trimmed_prompts.append(trimmed_prompt)
    return trimmed_prompts

def apply_chat_template(prompts: list[str], tokenizer):
    edited_prompts = []
    for prompt in prompts:
        messages = [
                {"role" : "user", "content" : prompt}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        edited_prompts.append(prompt)
    return edited_prompts

@app.route("/infer", methods=["POST"])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompts = datas["instances"]

    for key, value in params.items():
        if key in params_dict:
            params_dict[key] = value

    use_beam_search = params_dict.pop("use_beam_search", None)
    if use_beam_search == True:
        raise ValueError("Some issues with beam currently")
       # outputs = llm.beam_search(prompts, SamplingParams(**params_dict))
    tokenizer = llm.get_tokenizer()
    prompts = truncate_prompts(prompts, tokenizer, TRUNCATION_LENGTH)
    if args.use_chat_template:
        import json
        try:
            with open(args.model_name+"/tokenizer_config.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                if "chat_template" in data:
                    prompts = apply_chat_template(prompts, tokenizer)
                    print("Applied chat template")
                else:
                    print("no chat template")
        except FileNotFoundError:
            print("no chat template")
            
    outputs = llm.generate(prompts, SamplingParams(**params_dict))

    res = []
    if "prompt_logprobs" in params and params["prompt_logprobs"] is not None:
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs
            logp_list = [list(d.values())[0] for d in prompt_logprobs[1:]]
            res.append(logp_list)
        return jsonify(res)
    else:
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return jsonify(res)


if __name__ == "__main__":
    app.run(port=args.port, debug=False)
