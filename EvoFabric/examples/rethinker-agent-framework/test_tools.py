# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio

from dotenv import load_dotenv

from evofabric.app.rethinker import config
from evofabric.app.rethinker.tools import web_parse, web_search


def test_web_search():
    keywords = "Statistics"
    result = asyncio.run(web_search(keywords))
    print(result)


def test_web_parser_pdf():
    pdf_url = "https://arxiv.org/pdf/1512.03385"
    query = "What is the performance of ResNet"
    result = asyncio.run(web_parse(pdf_url, query))
    print(result)


def test_web_parser_jina():
    pdf_url = "https://en.wikipedia.org/wiki/Statistics"
    query = "What is 'Statistics'"
    result = asyncio.run(web_parse(pdf_url, query))
    print(result)


def test_web_parser_html():
    _jina_key = config.web_parser.jina_api_key
    _use_jina = config.web_parser.use_jina
    config.web_parser.jina_api_key = None
    config.web_parser.use_jina = False

    pdf_url = "https://en.wikipedia.org/wiki/Statistics"
    query = "What is 'Statistics'"
    result = asyncio.run(web_parse(pdf_url, query))
    print(result)

    config.web_parser.jina_api_key = _jina_key
    config.web_parser.use_jina = _use_jina


if __name__ == '__main__':
    config_path = "configs/config.yaml"
    config.load(config_path)
    load_dotenv(override=True)

    test_web_search()
    test_web_parser_pdf()
    test_web_parser_jina()
    test_web_parser_html()
