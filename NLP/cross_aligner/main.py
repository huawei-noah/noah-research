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

import argparse, os
import logging
import sys
from alignment import generate_alignment_pairs
from train import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, Tasks
from data_loader import load_and_cache_examples

logger = logging.getLogger(__name__)


def main(args):
    if os.path.exists(args.model_dir) and len(os.listdir(args.model_dir)) > 0:
        print("The model output path '%s' already exists and is not empty." % args.model_dir)
        return

    init_logger(args)
    set_seed(args)
    tokenizer = load_tokenizer(args.model_name_or_path)
    logger.info("******* Running with the following arguments *********")
    for a in vars(args):
        logger.info(a + " = " + str(getattr(args, a)))
    logger.info("***********************************************")
    train_dataset, train_examples = load_and_cache_examples(args, tokenizer, mode="train")
    train_examples = dict([(example.guid, example) for example in train_examples])
    dev_dataset, dev_examples = load_and_cache_examples(args, tokenizer, mode="dev")
    dev_examples = dict([(example.guid, example) for example in dev_examples])
    test_dataset, test_examples = load_and_cache_examples(args, tokenizer, mode="test")
    test_examples = dict([(example.guid, example) for example in test_examples])

    if args.align_languages:
        alignment_dataset = generate_alignment_pairs(args=args)
    else:
        alignment_dataset = None

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset,
                      train_examples, dev_examples, test_examples, tokenizer, alignment_dataset)

    if args.do_train:
        trainer.load_model(final_eval=False)
        logger.info(trainer.model)
        trainer.train()
        if args.save_model:
            trainer.save_model()

    if args.do_eval:
        if not args.do_train:
            trainer.load_model(final_eval=True)
        trainer.evaluate("dev", exp_name=args.model_dir)
        trainer.evaluate("test", exp_name=args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Task Specification
    parser.add_argument("--task", default=None, required=True, type=str,
                        help="The name of the task to run from %s" % ", ".join([t.value for t in Tasks]))
    parser.add_argument("--train_languages", required=True, type=str,
                        help="Language(s) to train on, comma separated, e.g. 'en', 'es', 'th'")
    parser.add_argument("--dev_languages", required=True, type=str,
                        help="Language(s) to use for model selection, comma separated, e.g. 'en', 'es', 'th'")
    parser.add_argument("--test_languages", required=True, type=str,
                        help="Language(s) to test on, comma separated, e.g. 'en', 'es', 'th'")
    parser.add_argument("--align_languages", default=None, type=str,
                        help="Run language alignment during training for each language, i.e. zh,ko")
    # Model, evaluation and data
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Path to save the models, logs, data and other project files")
    parser.add_argument("--data_dir", default="data", type=str,
                        help="The input data directory (should live outside the project dir)")
    parser.add_argument("--model_type", required=True, type=str,
                        help="Model type selected from the following list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--load_train_model", default=None, type=str,
                        help="Load a specific model for **training**, instead of from: %s" % str(MODEL_PATH_MAP))
    parser.add_argument("--load_eval_model", default=None, type=str,
                        help="Load a specific model for **evaluation**, instead of from: %s" % str(MODEL_PATH_MAP))
    parser.add_argument("--save_model", action="store_true",
                        help="Store the model in --model_dir (not saving trained models by default to save space)")
    parser.add_argument("--debug", action="store_true", help="Use this flag to print a detailed error report.")
    # Trainer arguments
    parser.add_argument('--seed', type=int, default=123456789,
                        help="Random seed for model initialization")
    parser.add_argument("--train_batch_size", required='--do_train' in sys.argv, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", required=True, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", required=True, type=int,
                        help="The maximum total input sequence length after tokenisation.")
    parser.add_argument("--learning_rate", required='--do_train' in sys.argv, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", required='--do_train' in sys.argv, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # Main commands
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--cuda_device", default="cpu", type=str, help="Which CUDA device to use or 'cpu' for CPU")
    # Loss function args
    parser.add_argument("--ignore_index", default=-100, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    # Auxiliary Loss Configurations
    parser.add_argument("--use_aux_losses", type=str, required="--align_languages" in sys.argv, default=None, nargs='+',
                        action='append', help="Specifies which auxiliary losses to use in alignment.",
                        choices=['XA', 'CA', 'CTR', 'TI'])
    # Auxiliary Loss Weighting Schemes
    parser.add_argument("--use_weighting", type=str, default=None,
                        help="Specifies one loss weighting scheme to use for multiple auxiliary losses [COV, None]")
    # Go!
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
