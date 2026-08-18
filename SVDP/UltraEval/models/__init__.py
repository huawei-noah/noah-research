from . import general_model, openai_model

MODEL_REGISTRY = {
    "gpt3_5": openai_model.GPT3_5,
    "gpt-3.5-turbo": openai_model.GPT3_5,
    "gpt4": openai_model.GPT4,
    "gpt-4": openai_model.GPT4,
    "general": general_model.GeneralModel,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
