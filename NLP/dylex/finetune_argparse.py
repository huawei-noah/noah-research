# coding=utf-8
# Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

def str2bool(flag):
    if flag.lower() in ('yes', 'true', 'y'):
        return True
    elif flag.lower() in ('no', 'false', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("unsopport argument")

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", default="cws_cityu", help="用来标识在哪个数据集上测"
    )
    parser.add_argument("--root_dir", default="/home/ma-user/work/path/to/project", type=str, help="在华为云平台运行时的路径前缀")
    parser.add_argument(
        "--data_dir", default="data/cws_cityu_1000", type=Path, help="数据集路径"
    ) #data_ner_public_en/conll2003_tokenize_cased
    parser.add_argument(
        "--model_name_or_path", default="model/roberta-wwm-base", type=Path, help="预训练模型存放路径"
    ) #  ../google_model/pretrain_cased_en
    parser.add_argument(
        "--output_dir", default="outputs/", type=Path, help="输出目录"
    )
    parser.add_argument("--model_save_dir", default="out_models/", type=Path)
    parser.add_argument(
        "--overwrite_output", default="no", type=str2bool, help="清空目录"
    )
    parser.add_argument(
        "--predict_model_path", default="", type=Path, help="infer时候加载的模型"
    )
    parser.add_argument(
        "--cache_dir", default="", type=Path, help="用来缓存数据生成的feature文件"
    )
    #-----------------------------------------------------------------------------------
    parser.add_argument(
        "--do_train", default="yes", type=str2bool, help="用来标识做训练"
    )
    parser.add_argument(
        "--do_predict", default="no", type=str2bool, help="用来标识只做predict"
    )
    parser.add_argument(
        "--do_lower_case", default="no", type=str2bool, help="分词之前是否先统一转小写"
    )
    parser.add_argument(
        "--on_huashan", default="no", type=str2bool, help="是否在华山平台上运行"
    )# simple code写的no
    parser.add_argument(
        "--on_yundao", default="no", type=str2bool, help="是否在云道平台上运行"
    )# simple code写的no
    #-----------------------------------------------------------------------------------
    parser.add_argument(
        "--train_bs_per_gpu", default=8, type=int, help="train batch size"
    )
    parser.add_argument(
        "--eval_bs_per_gpu", default=8, type=int, help="eval batch size",
    )
    parser.add_argument(
        "--num_train_epochs", default=20, type=int, help="设置训练轮数"
    )
    parser.add_argument(
        "--train_max_seq_length", default=128, type=int, help="句子最大长度"
    )
    parser.add_argument(
        "--lr", default=5e-5, type=float, help="设置学习率"
    )
    parser.add_argument(
        "--other_lr", default=5e-5, type=float, help="设置其他学习率"
    )
    parser.add_argument(
        "--LAMBDA", default=1.0, type=float, help="设置平衡系数"
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float
    )
    parser.add_argument(
        "--warmup_proportion", default=0.1, type=float
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--eval_per_n", default=100, type=int, help="多少步eval dev，并记录"
    )
    parser.add_argument(
        "--log_per_n", default=100, type=int, help="多少步打印一次loss"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="set seed"
    ) 
    #-----------------------------------------------------------------------------------
    parser.add_argument(
        "--max_dict_num", default=16, type=int, help="词典最大匹配数目"
    )
    parser.add_argument(
        "--match_num", default=1, type=int, help="每个位置最大匹配数目"
    )
    parser.add_argument(
        "--dict_path", default="data/cws_cityu_1000/dict/dict.txt", type=Path, help="加载词典路径"
    )
    parser.add_argument(
        "--dict_label_path", default="data/cws_cityu_1000/dict/itos_dict.txt", type=Path, help="词典标签路径"
    )
    parser.add_argument(
        "--intent_label_path", default="data/cws_cityu_1000/label.txt", type=Path, help="词典标签路径"
    )
    parser.add_argument(
        "--use_subword", default="yes", type=str2bool, help="序列标注的时候是否wordpiece的subword参与标注"
    )

    return parser
