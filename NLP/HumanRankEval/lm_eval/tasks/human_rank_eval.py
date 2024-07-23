# Copyright (C) 2020 EleutherAI
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from numpy import corrcoef
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


class HumanRankEval(Task):

    VERSION = 1
    DATASET_NAME = None
    DATASET_CACHE_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return [
            {
                "passage": item['question'],
                "query": f"Question: {item['question']}\nAnswer:",
                "choices": [a['text'] for a in item['answers']],
                "gold": [int(a['votes']) for a in item['answers']]
            }
            for item in self.dataset
        ]

    def doc_to_target(self, doc):
        raise NotImplementedError

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls

    def process_results(self, doc, results):
        gold = np.array(doc["gold"])
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        pearson_corr = corrcoef(gold, results / completion_len)[0][1]

        return {
            "pearson_corr": pearson_corr,
        }

    def higher_is_better(self):
        return {
            "pearson_corr": True,
        }

    def aggregation(self):
        return {
            "pearson_corr": mean,
        }

class HumanRankEvalPython(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_python"
    DATASET_CACHE_NAME = "HumanRankEvalPython"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalJava(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_java"
    DATASET_CACHE_NAME = "HumanRankEvalJava"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalUnix(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_unix"
    DATASET_CACHE_NAME = "HumanRankEvalUnix"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalCPP(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_cpp"
    DATASET_CACHE_NAME = "HumanRankEvalCPP"

    def doc_to_target(self, doc):
        raise NotImplementedError

class HumanRankEvalHTML(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_html"
    DATASET_CACHE_NAME = "HumanRankEvalHTML"

    def doc_to_target(self, doc):
        raise NotImplementedError

class HumanRankEvalEnglish(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_english"
    DATASET_CACHE_NAME = "HumanRankEvalEnglish"

    def doc_to_target(self, doc):
        raise NotImplementedError

class HumanRankEvalPhysics(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_physics"
    DATASET_CACHE_NAME = "HumanRankEvalPhysics"

    def doc_to_target(self, doc):
        raise NotImplementedError

class HumanRankEvalLaTeX(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_latex"
    DATASET_CACHE_NAME = "HumanRankEvalLaTeX"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalSoftEng(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_soft_eng"
    DATASET_CACHE_NAME = "HumanRankEvalSoftEng"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalStats(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_stats"
    DATASET_CACHE_NAME = "HumanRankEvalStats"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalCSDB(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_cs_db"
    DATASET_CACHE_NAME = "HumanRankEvalCSDB"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalLanguagesSciences(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_languages_sciences"
    DATASET_CACHE_NAME = "HumanRankEvalLanguagesSciences"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalAppleAndroid(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_apple_android"
    DATASET_CACHE_NAME = "HumanRankEvalAppleAndroid"

    def doc_to_target(self, doc):
        raise NotImplementedError


class HumanRankEvalMath(HumanRankEval):

    VERSION = 1
    DATASET_NAME = "human_rank_eval_math"
    DATASET_CACHE_NAME = "HumanRankEvalMath"

    def doc_to_target(self, doc):
        raise NotImplementedError
