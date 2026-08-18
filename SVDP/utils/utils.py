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

import json
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from configure_logger import configure_logger

logger = logging.getLogger(__name__)
configure_logger(logger, None)


def load_prompts(filename: str) -> list[str]:
    with open(filename, encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def save_prompts(prompts: list[str], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)


def apply_chat_template(prompts: list[str], tokenizer):
    if tokenizer.chat_template is None:
        logger.info("The tokenizer does not have any chat template, will skip chat formatting!")
        return prompts
    edited_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        edited_prompts.append(prompt)
    logger.info("Chat template applied!")
    return edited_prompts


def load_model_and_tokenizer(model_path: str, attn_implementation: str, torch_dtype, device_map: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model = model.eval()
    return model, tokenizer


def identify_is_drelu(model) -> bool:
    import inspect

    source = inspect.getsource(model.model.layers[0].mlp.forward)
    if "self.act_fn(self.up_proj(x))" in source:
        is_drelu = True
        logger.warning("Identified as dReLU type!")
    else:
        is_drelu = False
        logger.warning("Identified as vanilla ReLU type!")
    return is_drelu


def register_vllm_model(vllm_module_path: str, model_path: str) -> None:
    import importlib
    import json
    import sys

    from vllm import ModelRegistry

    module_path = vllm_module_path
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    with open(model_path + "/config.json", encoding="utf-8") as file:
        data = json.load(file)
        class_name = data["architectures"][0]
    ModelRegistry.register_model(class_name, getattr(module, class_name))
