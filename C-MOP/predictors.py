# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod


class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt


class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']
    user_prompt = "Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\nAnswer Yes or No as labels\n\nText: {{ text }} Label:"

    task_instruct = "Answer Yes (lie) or No (not a lie) as labels\n\nText: {{ text }} Label:"


class EthosBinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']
    user_prompt = "Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\nAnswer Yes or No as labels\n\nText: {{ text }} Label:"

    task_instruct = "Is the following text hate speech? Answer Yes or No as labels\n\nText: {{ text }} Label:"


class Gsm8kPredictor(GPT4Predictor):
    task_instruct = "{{ text }}\n\nPlease put your final answer within \\boxed{}."


class BBHPredictor(GPT4Predictor):
    task_instruct = "Question: {{ text }}\n You must give your final answer by starting with 'So the answer is' "


class CfinBenchPredictor(GPT4Predictor):
    task_instruct = "你是一个财经金融专家，下面可能会有一些单选、多选或者判断的题目，请根据你自身渊博的知识进行作答。题目如下：{{text}}\n答案："
    single_user_prompt = task_instruct
    multi_user_prompt = task_instruct
    judgement_user_prompt = task_instruct
