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
parser.add_argument(
    "--languages",
    type=str,
    default="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu",
)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 42, 55])
args = parser.parse_args()


seed_results = []
eval_overall_f1 = []
for seed in args.seeds:
    if "seed" in args.model_dir:
        current_seed = args.model_dir.split("_seed")[-1].split("_")[0]
        model_dir = args.model_dir.replace(f"_seed{current_seed}", f"_seed{seed}")
    else:
        current_seed = args.model_dir.split("/")[-1].split("_")[2]
        model_dir = args.model_dir.replace(f"_{current_seed}", f"_{seed}")

    res = {}
    with open(os.path.join(f"{model_dir}", f"eval_results.json")) as infile:
        data = json.load(infile)

        eval_overall_f1.append(data["eval_overall_f1"])

    for lang in args.languages.split(","):
        with open(
            os.path.join(f"{model_dir}", f"predict-{lang}_results.json")
        ) as infile:
            data = json.load(infile)

            res[lang] = data["predict_overall_f1"]

    avg = np.mean([res[lang] for lang in res])

    seed_results.append([res[lang] for lang in res])

    print("\t".join(["AVG"] + [f"{lang}" for lang in res]))
    print("\t".join([f"{avg:.4f}"] + [f"{res[lang]:.4f}" for lang in res]))
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

avg_eval_overall_f1 = np.mean(np.mean(eval_overall_f1))
print(f"avg_eval_overall_f1 on EN: {avg_eval_overall_f1:.4f}")
