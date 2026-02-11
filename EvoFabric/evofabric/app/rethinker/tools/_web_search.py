# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, Dict, List, Union

import httpx

from evofabric.logger import get_logger
from ._utils import RetryHandler
from .._config import config

logger = get_logger()

# Constants
_SERPER_BASE_URL = "https://google.serper.dev/search"


async def _query_serper(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    ssl: bool
) -> Dict[str, Any]:
    """Sends a POST request to the Serper API and processes the response.

    Args:
        url (str): The target URL for the Serper API.
        payload (Dict[str, Any]): The JSON payload containing search parameters.
        headers (Dict[str, str]): HTTP headers including authentication.
        timeout (int): The request timeout in seconds.
        ssl (bool): Whether to verify SSL certificates.

    Returns:
        Dict[str, Any]: The processed JSON response data from the API.

    Raises:
        Exception: If the HTTP status is not 200 or if the response data is empty
            after filtering.
    """
    async with httpx.AsyncClient(verify=ssl, timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")

        data = response.json()

        data.pop('searchParameters', None)
        data.pop('credits', None)

        if not data:
            raise Exception(
                "The google search API is temporarily unavailable, "
                "please try again later."
            )

        return data


async def web_search(
    query: str,
    top_k: int = 10
) -> Union[Dict[str, Any], List[Any]]:
    """Uses the Google Search engine to find information on the web.

    This function constructs the necessary payload and headers to query the
    Serper API, utilizing a retry mechanism for robustness.

    Args:
        query (str): The search query to submit.
        top_k (int, optional): The maximum number of results to return.
            Defaults to 10.

    Returns:
        Union[Dict[str, Any], List[Any]]: A dictionary containing search results
        (specifically under the 'organic' key) if successful. Returns an empty
        list if the retry handler returns None.

        Structure of success result:
            - organic (list):
                - title (str): The title of the web page.
                - link (str): The URL of the web page.
                - snippet (str): A summary of the content.
                - sitelinks (list): Additional links from the same domain.
    """
    cfg = config.web_search

    payload = {
        "q": query,
        "num": top_k,
        "gl": "us",
        "hl": "en",
        "location": "United States"
    }

    headers = {
        'X-API-KEY': cfg.serper_api_key,
        'Content-Type': 'application/json'
    }

    # Execute request with retry logic
    data, error = await RetryHandler.execute(
        _query_serper,
        url=_SERPER_BASE_URL,
        payload=payload,
        headers=headers,
        timeout=cfg.timeout,
        ssl=cfg.ssl_verify,
        max_retries=cfg.retries
    )

    return data or []
