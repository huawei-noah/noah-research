# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import argparse
import asyncio
import json
import os
import sys
import traceback

import rich.pretty
from dotenv import load_dotenv

from evofabric.app.rethinker import build_rethinker_graph, config, run_rethinker_graph
from evofabric.logger import get_logger, set_logger

logger = get_logger()


def get_args():
    """Parse command-line arguments and return the args object."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
        help="Path to configuration YAML file, must match GraphConfig definition"
    )
    parser.add_argument(
        "--graph-visual-path",
        default="graph.html",
        help="Path to save graph visualization HTML"
    )
    return parser.parse_args()


def logger_setup():
    """Initialize logging and filter internal module logs."""
    from loguru import logger
    logger.remove()
    logger.add(
        sys.stderr,
        filter=lambda r: r["name"] != "evofabric.core.graph._node"
    )
    set_logger(logger)


def config_setup():
    """Load configuration from YAML and print the graph config."""
    args = get_args()
    config.load(args.config)
    rich.pretty.pprint(config.graph.model_dump())


def load_dataset():
    """Load dataset from a JSON file."""
    data_file_path = config.exp.input_file_path
    with open(data_file_path, "r", encoding="utf-8") as data_file:
        return json.load(data_file)


def save_graph_visualization():
    """Generate graph visualization HTML if path is provided."""
    args = get_args()
    if args.graph_visual_path:
        graph = build_rethinker_graph()
        graph.draw_graph(args.graph_visual_path)


async def main():
    """Main function: setup config, logging, visualization, and run queries."""
    config_setup()
    logger_setup()
    save_graph_visualization()
    data_list = load_dataset()

    # debug filter
    sem = asyncio.Semaphore(config.exp.max_question_thread_limit)
    tasks = [run_rethinker_graph(data["question"], data["id"], sem) for data in data_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            traceback_info = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            logger.error(f"Rethinker graph error:\n{traceback_info}")
    return results

if __name__ == '__main__':
    load_dotenv(override=True)
    asyncio.run(main())
