from pprint import pprint
from typing import List, Union
import lm_eval.base
from . import human_rank_eval


########################################
# All tasks
########################################

TASK_REGISTRY = {
    "human_rank_eval_python": human_rank_eval.HumanRankEvalPython,
    "human_rank_eval_java": human_rank_eval.HumanRankEvalJava,
    "human_rank_eval_unix": human_rank_eval.HumanRankEvalUnix,
    "human_rank_eval_cpp": human_rank_eval.HumanRankEvalCPP,
    "human_rank_eval_html": human_rank_eval.HumanRankEvalHTML,
    "human_rank_eval_english": human_rank_eval.HumanRankEvalEnglish,
    "human_rank_eval_physics": human_rank_eval.HumanRankEvalPhysics,
    "human_rank_eval_latex": human_rank_eval.HumanRankEvalLaTeX,
    "human_rank_eval_soft_eng": human_rank_eval.HumanRankEvalSoftEng,
    "human_rank_eval_stats": human_rank_eval.HumanRankEvalStats,
    "human_rank_eval_cs_db": human_rank_eval.HumanRankEvalCSDB,
    "human_rank_eval_languages_sciences": human_rank_eval.HumanRankEvalLanguagesSciences,
    "human_rank_eval_apple_android": human_rank_eval.HumanRankEvalAppleAndroid,
    "human_rank_eval_math": human_rank_eval.HumanRankEvalMath,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
