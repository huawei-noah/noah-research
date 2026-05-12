from typing import Tuple
import os
import requests
import urllib3
import random
import logging

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from configure_logger import configure_logger
logger = logging.getLogger(__name__)
configure_logger(logger, None)

def load_prompts(filename: str) -> list[str]:
    with open(filename, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts

def save_prompts(prompts: list[str], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)

def fix_ssl():

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    old_merge_environment_settings = requests.Session.merge_environment_settings
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False
        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

def apply_chat_template(prompts: list[str], tokenizer):
    if tokenizer.chat_template is None:
        logger.info("The tokenizer does not have any chat template, will skip chat formatting!")
        return prompts
    edited_prompts = []
    for prompt in prompts:
        messages = [
                {"role" : "user", "content" : prompt}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        edited_prompts.append(prompt)
    logger.info("Chat template applied!")
    return edited_prompts

def load_model_and_tokenizer(model_path: str, attn_implementation:str, torch_dtype, device_map:str) -> Tuple:
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
    from vllm import ModelRegistry
    import importlib
    import sys
    import json

    module_path = vllm_module_path
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    with open(model_path+"/config.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        class_name = data["architectures"][0]
    ModelRegistry.register_model(class_name, getattr(module, class_name))