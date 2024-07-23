from . import auto_hf
from . import mindspore

MODEL_REGISTRY = {
    "auto_hf": auto_hf.AutoCausalLM,
    "mindspore": mindspore.MindSporeLM
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
