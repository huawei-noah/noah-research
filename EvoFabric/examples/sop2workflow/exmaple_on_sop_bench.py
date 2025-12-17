# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This demo illustrates how to transform an SOP into a fully executable workflow using the Sop2workflow feature.
# The sample data used in this example is derived from SOP-Bench (https://arxiv.org/abs/2506.08119).

import argparse
import asyncio
import os

from dotenv import load_dotenv

from evofabric.app.sop2workflow import WorkflowGenerator
from evofabric.core.agent import UserNode
from evofabric.core.clients import OpenAIChatClient
from evofabric.core.tool import McpToolManager
from evofabric.core.typing import StdioLink


def get_args():
    parser = argparse.ArgumentParser()

    # LLM setting
    parser.add_argument("--graph_llm", type=str, default="glm-4.5-air")
    parser.add_argument("--node_llm", type=str, default="glm-4.5-air")
    parser.add_argument("--run_llm", type=str, default="glm-4.5-air")
    parser.add_argument("--no_http_verify", action='store_true', default=False)
    parser.add_argument("--env", type=str, default=".env",
        help=".env file path, must contain OPENAI_API_KEY and OPENAI_BASE_URL")

    # exp setting
    parser.add_argument("--sop", type=str, default="customer_service_sop/sop.txt", help="sop file path")
    parser.add_argument("--tool_file", type=str, default="customer_service_sop/tool_mcp.py",
        help="Python file path of tools")
    parser.add_argument("--class_name", type=str, default="ServiceAccountManager",
        help="Class name storing all python file")
    parser.add_argument("--save_dir", type=str, default="output/customer_service_sop/",
        help="graph desp file save path")

    return parser.parse_args()


def load_sop(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


async def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    load_dotenv(args.env, override=True)

    tool_manager = McpToolManager(
        server_links={
            "tools": StdioLink(
                command="python",
                args=[args.tool_file]
            )
        }
    )

    generator = WorkflowGenerator(
        sop=load_sop(args.sop),
        graph_generation_client=OpenAIChatClient(
            model=args.graph_llm,
            client_kwargs={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "max_retries": 5,
                "timeout": 3600
            },
            http_client_kwargs={"verify": not args.no_http_verify},
            inference_kwargs={"temperature": 0.0, "timeout": 3600}
        ),
        graph_node_complete_client=OpenAIChatClient(
            model=args.node_llm,
            client_kwargs={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "max_retries": 5,
                "timeout": 3600
            },
            http_client_kwargs={"verify": not args.no_http_verify},
            inference_kwargs={"temperature": 0.0, "timeout": 3600}
        ),
        graph_run_client=OpenAIChatClient(
            model=args.run_llm,
            client_kwargs={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "max_retries": 5,
                "timeout": 3600
            },
            http_client_kwargs={"verify": not args.no_http_verify},
            inference_kwargs={"temperature": 0.0, "timeout": 3600}
        ),
        output_dir=args.save_dir,
        tools=[tool_manager],
        memories={},
        state_schema=None,
        addition_global_instruction="",
        user_node=UserNode(),
        fallback_node="end",
        auto_self_loop=True,
        tool_list_mode="select",
        memory_list_mode="select",
        exit_function_name=None,
        build_kwargs={"max_turn": 20},
    )

    graph = await generator.generate()
    graph.draw_graph()

    result = await graph.run({"messages": [{"role": "user", "content": "hello"}]})
    print(result)


if __name__ == '__main__':
    asyncio.run(main())
