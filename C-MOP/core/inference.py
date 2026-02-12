# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import requests
import time
from liquid import Template
from openai import AsyncOpenAI, OpenAI
from sklearn.metrics import f1_score
import predictors
from utils.constants import TASK_MODEL_URL, TASK_MODEL_NAME, TASK_MODEL_APIKEY
from utils.constants import OPTIMIZER_MODEL_URL, OPTIMIZER_MODEL_NAME, OPTIMIZER_MODEL_APIKEY


async def run_test(system_prompt, ex, predictor, custom_instruct=None):
    """ Executes task model inference test."""
    instruct = custom_instruct if custom_instruct else predictor.task_instruct
    user_query = f"{system_prompt}\n\n{Template(instruct).render(text=ex['text'])}"
    messages = [{"role": "user", "content": user_query}]

    client = AsyncOpenAI(api_key=TASK_MODEL_APIKEY,
                         base_url=TASK_MODEL_URL)

    retries, MAX_RETRIES = 0, 16
    while retries < MAX_RETRIES:
        try:
            r = await client.chat.completions.create(
                messages=messages,
                model=TASK_MODEL_NAME,
                temperature=0,
                max_tokens=8192
            )
            break
        except Exception as e:
            retries += 1
            await asyncio.sleep(1)
    else:
        return "Inference failed multiple times."

    tmp = r.choices[0].message.content
    return parse_results(tmp)


def parse_results(tmp):
    res = tmp.split("</think>")[-1].strip().split("[unused17]")[-1].strip()
    return res


async def process_task(task, candidate, predictor, test_task):
    """ Process the task and inference."""
    labels = []
    preds = []
    texts = []
    exs = []

    tasks_tmp = []
    for ex in test_task:
        if hasattr(task, "cfinbench_postprocess"):
            question_type = ex.get("question_type", None)
            if question_type == "multi_choice":
                custom_instruct = predictor.multi_user_prompt
            elif question_type == "single_choice":
                custom_instruct = predictor.single_user_prompt
            elif question_type == "judgment":
                custom_instruct = predictor.judgement_user_prompt
            else:
                custom_instruct = None
        else:
            custom_instruct = None

        tasks_tmp.append(run_test(candidate, ex, predictor, custom_instruct=custom_instruct))

    results = await asyncio.gather(*tasks_tmp)

    for ex, pred in zip(test_task, results):
        texts.append(ex['text'])
        labels.append(ex['label'])
        is_special_task = isinstance(predictor,
                                     (
                                         predictors.Gsm8kPredictor, predictors.CfinBenchPredictor,
                                         predictors.BBHPredictor))
        if is_special_task:
            pred = pred
        else:
            pred = 1 if pred.upper().startswith('YES') else 0
        preds.append(pred)
        exs.append(ex)

    return labels, preds, texts, exs


async def multi_inference(predictor, candidate, test_exs, task):
    """ Process the task and get the task score."""
    labels, preds, texts, exs = await process_task(task, candidate, predictor, test_exs)

    if len(preds) != len(labels):
        raise ValueError(f"Error: Preds({len(preds)}) and Labels({len(labels)}) length mismatch!")

    score = calculate_score(task, labels, preds, exs=exs)

    return score, texts, labels, preds, exs


def calculate_score(task, labels, preds, exs=None):
    """ Calculates the task score."""
    if not preds or len(preds) == 0:
        return 0

    if hasattr(task, "gsm8k_postprocess"):
        correct = 0
        for p, l in zip(preds, labels):
            ref_ans = l.split('#### ')[1].replace(',', '').strip() if '####' in str(l) else str(l)
            pred_ans = task.gsm8k_postprocess(p)
            if ref_ans == pred_ans:
                correct += 1
        return 100 * correct / len(labels)

    elif hasattr(task, "cfinbench_postprocess"):
        correct_score_multi, count_multi = 0, 0
        correct_score, count = 0, 0

        for p, l, ex in zip(preds, labels, exs):
            q_type = ex.get("question_type")
            pred = task.cfinbench_postprocess(p, q_type)
            ref = task.cfinbench_postprocess(l, q_type)

            if q_type == "multi_choice":
                count_multi += 2
                if pred == ref:
                    correct_score_multi += 2
                elif all(item in ref for item in pred) and pred:
                    correct_score_multi += 1
            else:
                count += 1
                if pred == ref:
                    correct_score += 1

        score = 0
        if count_multi > 0 and count > 0:
            score = (correct_score_multi / count_multi) * 50 + (correct_score / count) * 50
        elif count_multi > 0:
            score = (correct_score_multi / count_multi) * 100
        elif count > 0:
            score = (correct_score / count) * 100
        return score

    elif hasattr(task, "bbh_mcq_postprocess"):
        correct = 0
        for p, l, ex in zip(preds, labels, exs):
            q_type = ex.get("question_type")
            if q_type == "bbh_multiple_choice_sets":
                is_correct = task.bbh_mcq_postprocess(p) == task.bbh_mcq_postprocess(l)
            else:
                is_correct = task.bbh_freeform_postprocess(p) == task.bbh_freeform_postprocess(l)
            if is_correct:
                correct += 1
        return (correct / len(labels)) * 100

    else:
        return f1_score(labels, preds, average='micro') * 100


def optimizer_model_infer(prompt, system_prompt=False, temperature=0.0, n=1, top_p=1, stop=None, max_tokens=8192,
                          presence_penalty=0, frequency_penalty=0, logit_bias=None, timeout=300):
    """ Executes optimizer model inference."""
    if system_prompt and system_prompt != "":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    client = OpenAI(
        api_key=OPTIMIZER_MODEL_APIKEY,
        base_url=OPTIMIZER_MODEL_URL
    )

    while True:
        try:
            completion = client.chat.completions.create(
                model=OPTIMIZER_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                n=n,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                timeout=timeout
            )
            content = completion.choices[0].message.content
            return parse_results(content)

        except Exception as e:
            print(f"Inference failed: {e}. Retrying in 1s...")
            time.sleep(1)
