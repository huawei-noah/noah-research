# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os, re, sys
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--languages", type=str, default="en,es,de,hi,fr,pt,zh,ja,tr")
parser.add_argument("--model_dir", type=str)
parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 42, 55])
args = parser.parse_args()


seed_results = []
seed_results_intent = []
eval_overall_performance = []
for seed in args.seeds:
    if "seed" in args.model_dir:
        current_seed = args.model_dir.split("_seed")[-1].split("_")[0]
        model_dir = args.model_dir.replace(f"_seed{current_seed}", f"_seed{seed}")
    else:
        current_seed = args.model_dir.split("/")[-1].split("_")[-1]
        model_dir = args.model_dir.replace(f"_{current_seed}", f"_{seed}")

    with open(os.path.join(f"{model_dir}", f"eval_results.json")) as infile:
        data = json.load(infile)

        eval_overall_performance.append(data["eval_overall_performance"])

    res = {}
    intent = {}
    for lang in args.languages.split(","):
        with open(os.path.join(f"{model_dir}", f"predict-{lang}.json")) as infile:
            data = json.load(infile)

            res[lang] = data["overall_f1"]
            intent[lang] = data["intent_accuracy"]

    avg = np.mean([res[lang] for lang in res])
    avg_intent = np.mean([intent[lang] for lang in intent])

    seed_results.append([res[lang] for lang in res])
    seed_results_intent.append([intent[lang] for lang in intent])

    print("\t".join(["AVG"] + [f"{lang}" for lang in res]))
    print("\t".join([f"{avg:.4f}"] + [f"{res[lang]:.4f}" for lang in res]))
    print("\t".join([f"{avg_intent:.4f}"] + [f"{intent[lang]:.4f}" for lang in intent]))
    print()


# Collect all
seed_results = np.array(seed_results)  # seeds X langs
seed_results_mean = np.mean(seed_results, axis=0)
seed_results_std = np.std(seed_results, axis=0)

avg_across_langs_mean = np.mean(
    np.mean(seed_results, axis=1), axis=0
)  # mean across langs, mean across seeds
avg_across_langs_std = np.std(
    np.mean(seed_results, axis=1), axis=0
)  # mean across langs, std across seeds

seed_results_intent = np.array(seed_results_intent)  # seeds X langs
seed_results_mean_intent = np.mean(seed_results_intent, axis=0)
seed_results_std_intent = np.std(seed_results_intent, axis=0)

avg_across_langs_mean_intent = np.mean(
    np.mean(seed_results_intent, axis=1), axis=0
)  # mean across langs, mean across seeds
avg_across_langs_std_intent = np.std(
    np.mean(seed_results_intent, axis=1), axis=0
)  # mean across langs, std across seeds

print(f"\n********** {args.model_dir} **********")
print("\t".join(["    "] + ["AVG"] + [f"{lang}" for lang in res]))
print(
    "\t".join(
        ["mean"]
        + [f"{avg_across_langs_mean:.4f}"]
        + [f"{i:.4f}" for i in seed_results_mean]
    )
)
print(
    "\t".join(
        ["std"]
        + [f"{avg_across_langs_std:.4f}"]
        + [f"{i:.4f}" for i in seed_results_std]
    )
)
print(
    "\t".join(
        ["mean"]
        + [f"{avg_across_langs_mean_intent:.4f}"]
        + [f"{i:.4f}" for i in seed_results_mean_intent]
    )
)
print(
    "\t".join(
        ["std"]
        + [f"{avg_across_langs_std_intent:.4f}"]
        + [f"{i:.4f}" for i in seed_results_std_intent]
    )
)


avg_eval_overall_performance = np.mean(np.mean(eval_overall_performance))
print(f"avg_eval_overall_performance on EN: {avg_eval_overall_performance}")
