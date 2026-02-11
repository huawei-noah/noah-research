# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse

from dotenv import load_dotenv

from evofabric.app.rethinker import LLMConfig
from evofabric.app.rethinker.evaluation import (
    GaiaEvaluator, HLEEvaluator, XBenchEvaluator
)


def get_args():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")

    # benchmark & data
    parser.add_argument(
        "--benchmark",
        default="hle",
        choices=["hle", "gaia", "xbench"],
        help="benchmark name"
    )
    parser.add_argument(
        "--dataset",
        default="data/HLE_all.json",
        help="path to dataset json file"
    )
    parser.add_argument(
        "--save-root",
        default="output/pangu_hle_all",
        help="output directory"
    )

    parser.add_argument(
        "--model-name",
        default="o3-mini-2025-01-31",
        help="LLM model name"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for LLM (can also be set via env)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for LLM API"
    )
    parser.add_argument(
        "--disable-ssl-verify",
        action="store_true",
        help="Disable SSL verification"
    )

    parser.add_argument(
        "--csb-token",
        default=None,
        help="csb-token for request header"
    )

    # result saving
    parser.add_argument(
        "--save-result",
        default=None,
        help="result json filename (e.g. evaluate.json); if not set, results will not be saved"
    )

    return parser.parse_args()


def main():
    BENCHMAKRS = {
        "hle": HLEEvaluator,
        "gaia": GaiaEvaluator,
        "xbench": XBenchEvaluator,
    }
    args = get_args()

    llm_config = LLMConfig(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        http_client_kwargs={
            "verify": not args.disable_ssl_verify
        },
        csb_token=args.csb_token,
    )
    evaluator = BENCHMAKRS[args.benchmark](
        data_file=args.dataset,
        result_root=args.save_root,
        eval_llm=llm_config,
        save_path=args.save_result,
    )
    evaluator.run()


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
