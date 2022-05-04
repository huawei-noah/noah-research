# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import os
import sys
import argparse
import logging
import subprocess
import glob


FORMAT = "%(asctime)-15s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_NAME_MAP = {
    "bert-small": "bert-small-cased",
    }

parser = argparse.ArgumentParser()
parser.add_argument("--res_path_pat", type=str, default="./cache/train_s-cls_cloud/*", help="")
parser.add_argument("--task", type=str, help="")
parser.add_argument("--task_type", type=str, help="s-cls, seq-tag")
parser.add_argument("--run_date", type=str, help="")
parser.add_argument("--langs", type=str, help="test languages")
parser.add_argument("--out_dir", type=str, help="")
parser.add_argument("--data_dir", type=str, help="")
parser.add_argument("--model_name", type=str, help="")
args = parser.parse_args()

def main():
  res_paths = glob.glob(args.res_path_pat)
  task = args.task 

  run_date = args.run_date
  exec_script = f"scripts/test_{args.task_type}_cloud.sh"
  exec_script_name = os.path.basename(exec_script)
  
  out_dir = os.path.join(args.out_dir, exec_script_name)
  os.makedirs(out_dir, exist_ok=True)
  
  for i, path in enumerate(res_paths):
    fields = path.split('/')
    model_name, langs, spec_dir = fields[-5], fields[-3], fields[-1]
    tgt_langs = args.langs
    if args.model_name:
      model_name = args.model_name
    else:
      model_name = MODEL_NAME_MAP[model_name] if model_name in MODEL_NAME_MAP else model_name
    # print(model_name, tgt_lang, spec_dir)
    log_prefix = os.path.join(out_dir, f"{run_date}.none.{model_name}.{tgt_langs}")
    cmd = f"{exec_script} --test_langs={tgt_langs} --model={model_name} --data_dir={args.data_dir} --load_dir={path}/checkpoint-best/ --task={task}"
    logger.info(" ###### running")
    logger.info(cmd)
    with open(log_prefix + ".info", 'w') as finfo, open(log_prefix + ".err", 'w') as ferr:
      results = subprocess.run([sys.executable] + cmd.split(' '), stdout=finfo, stderr=ferr) # py3.7+: , capture_output=True, text=True)
    logger.info(f" ###### results: {results}")

if __name__ == "__main__":
  main()
