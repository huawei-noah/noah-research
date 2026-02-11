# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import functools
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
import requests
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel
from transformers import AutoTokenizer

from evofabric.app.rethinker.adapter import get_client
from evofabric.core.typing import UserMessage
from evofabric.logger import get_logger
from ._pdf_reader import download_and_read_pdf
from ._utils import RetryHandler
from .._config import config

logger = get_logger()

CONSTANTS = {
    "JINA_API_BASE_URL": "https://r.jina.ai/",
    "FAILED_INFO": "Failed to fetch web page",
    "USER_AGENT": {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
    }
}

_TAGS_TO_REMOVE = [
    'script', 'style', 'meta', 'link', 'noscript',
    'iframe', 'svg', 'canvas', 'video', 'audio',
    'input', 'button', 'form'
]


def _clean_html(html_content: str) -> str:
    """
    Removes JavaScript, CSS, comments, and attributes from HTML content.

    Args:
        html_content: The raw HTML string to be cleaned.

    Returns:
        A cleaned string containing only the text structure of the HTML,
        or an empty string if the input is empty.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted tags (JS, CSS, Media, Forms)
    for tag in soup.find_all(_TAGS_TO_REMOVE):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove attributes from all remaining tags (e.g., class, id, style)
    for tag in soup.find_all(True):
        tag.attrs.clear()

    return str(soup)


class LLMService:
    @staticmethod
    def split_chunks(text: str) -> List[str]:
        cfg = config.web_parser
        chunks = []
        for i in range(0, len(text), cfg.llm_input_max_char):
            chunk = text[i: i + cfg.llm_input_max_char]
            chunks.append(chunk)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    @staticmethod
    async def call_model(query: str) -> str:
        """
        Calls the configured LLM model with a given query.

        Args:
            query (str): The prompt content to send to the model.

        Returns:
            str: The raw content string returned by the model.

        Raises:
            Exception: If the underlying model client call fails.
        """
        model_name = config.web_parser.model
        client = get_client(model_name)
        input_msg = [UserMessage(content=query)]

        try:
            chat_response = await client.create(input_msg)
            return chat_response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e


class BaseWebParser(ABC, BaseModel):
    """
    Abstract base class for web content parsers.

    Defines the interface for fetching and processing web content.
    """

    timeout: int = 60

    @abstractmethod
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetches raw text content from the specified URL.

        Args:
            url (str): The target URL to fetch.

        Returns:
            Optional[str]: The raw text content if successful, None otherwise.
        """
        pass

    @abstractmethod
    async def process_content(
        self,
        raw_text: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Processes raw text content to extract relevant information based on a query.

        Args:
            raw_text (str): The raw text content retrieved from the web.
            query (str): The search query used to filter or analyze the content.

        Returns:
            Dict[str, Any]: A dictionary containing the processed results.
        """
        pass

    async def parse(self, url: str, query: str) -> Dict[str, Any]:
        """
        Orchestrates the fetching and processing of web content.

        This method attempts to fetch content using a retry mechanism. If successful,
        it delegates processing to the concrete implementation of `process_content`.

        Args:
            url (str): The target URL to parse.
            query (str): The query context for parsing.

        Returns:
            Dict[str, Any]: A dictionary containing the result.
                On failure, returns a default dictionary with a specific error message,
                empty URLs list, and a score of -1.
        """
        raw_text, error = await RetryHandler.execute(self.fetch_content, url)
        if config.web_parser.show_url_content_max_char > 0:
            show_content = raw_text if not raw_text else raw_text[:config.web_parser.show_url_content_max_char]
            logger.info(f"Web content for {url}: \n{show_content}")

        if not raw_text or raw_text == CONSTANTS["FAILED_INFO"]:
            return {
                "content": "failed to fetch web content",
                "urls": [],
                "score": -1
            }

        try:
            result = await self.process_content(raw_text, query)
            return result
        except Exception as e:
            logger.error(f"Process content failed: {e}")
            traceback.print_exc()
            return {
                "content": "failed to parse web content",
                "urls": [],
                "score": -1
            }


class PDFParser(BaseWebParser):
    """Parses content from PDF files located at a specific URL.

    This parser handles the downloading and text extraction of PDFs asynchronously,
    then utilizes an LLM to analyze the content based on a user query.
    """

    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Downloads and extracts text from a PDF file asynchronously.

        Since PDF processing is a blocking I/O and CPU-bound operation,
        it is executed in a separate thread to avoid blocking the event loop.

        Args:
            url (str): The URL of the PDF file.

        Returns:
            Optional[str]: The extracted text content, or None if extraction
            fails or returns empty text.
        """
        loop = asyncio.get_running_loop()

        # Run the blocking function in the default executor
        text = await loop.run_in_executor(
            None,
            functools.partial(download_and_read_pdf, url)
        )

        return text if text else None

    async def process_content(
        self,
        raw_text: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Analyzes the extracted PDF text using an LLM.

        Note: To manage token limits, this method splits the text and currently
        uses the first chunk for the analysis.

        Args:
            raw_text (str): The full text extracted from the PDF.
            query (str): The user's query context.

        Returns:
            Dict[str, Any]: A dictionary containing the LLM's response,
            with a fixed score of 1.
        """
        chunks = LLMService.split_chunks(raw_text)
        first_chunk = chunks[0]
        prompt_template = str(config.prompts.web_parser_pdf)
        final_query = prompt_template.format(
            user_query=query,
            pdf_info=first_chunk
        )
        response_text = await LLMService.call_model(final_query)

        return {"content": response_text}


class SimpleWebParser(BaseWebParser):
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Wrapper method to fetch content from a URL.

        Args:
            url: The target URL.

        Returns:
            The content string if found, otherwise None.
        """
        headers = CONSTANTS["USER_AGENT"].copy()
        headers.update({
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1'
        })
        cfg = config.web_parser
        try:
            # Using a session context to handle potential redirect cookies gracefully
            with requests.Session() as session:
                response = session.get(
                    url,
                    headers=headers,
                    timeout=cfg.timeout,
                    verify=cfg.ssl_verify
                )
                response.raise_for_status()

                # Detect encoding to ensure correct text decoding (can be slow)
                response.encoding = response.apparent_encoding

                cleaned_html = _clean_html(response.text)
                logger.info(f"Requests success and cleaned: {url}")
                return cleaned_html

        except Exception as e:
            logger.warning(f"Requests failed for {url}: {e}")
            return None

    async def process_content(self, raw_text: str, query: str) -> Dict[str, Any]:
        """
        Extract content related to query from the raw text using LLM

        Args:
            raw_text: The raw text content to be processed.
            query: The user's query or instruction.

        Returns:
            A dictionary containing the parsed JSON response from the LLM.
        """
        chunks = LLMService.split_chunks(raw_text)

        content_snippet = chunks[0] if chunks else ""

        template = str(config.prompts.web_parser_html)
        final_prompt = template.format(user=query, info=content_snippet)

        answer = await LLMService.call_model(final_prompt)
        return {"content": answer}


class JinaWebParser(SimpleWebParser):
    """A parser that attempts to fetch content via Jina's Reader API before falling back to requests."""

    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetches URL content using Jina Reader API, falling back to SimpleWebParser on failure.

        This method checks for a Jina API key. If present, it proxies the request
        through Jina. If the key is missing or the request fails, it calls the
        parent class's `fetch_content` method.

        Args:
            url: The target URL to fetch.

        Returns:
            The text content of the URL (cleaned by Jina or the parent parser),
            or None if both methods fail.
        """
        api_key = config.web_parser.jina_api_key

        if api_key:
            # Construct Jina Reader URL: Base URL + target URL (stripped of leading slashes)
            jina_target_url = f"{CONSTANTS['JINA_API_BASE_URL']}{url.lstrip('/')}"
            headers = {"Authorization": f"Bearer {api_key}"}

            try:
                logger.info(f"Fetching via Jina: {jina_target_url}")

                # Use async client to fetch from Jina
                # Note: verify=False is maintained from original code, likely to handle specific cert issues.
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.get(
                        jina_target_url,
                        headers=headers,
                        timeout=self.timeout,
                        follow_redirects=True
                    )
                    response.raise_for_status()
                    return response.text

            except Exception as e:
                logger.warning(f"Jina fetch failed ({e}), downgrading to SimpleWebParser")
        else:
            logger.warning("Jina API Key not found, downgrading.")

        # Fallback to the logic defined in SimpleWebParser
        return await super().fetch_content(url)

    async def process_content(self, raw_text: str, query: str) -> Dict[str, Any]:
        """
        Processes text using an LLM, specifically handling reasoning model outputs.

        This method overrides the parent logic to:
        1. Log input/chunk sizes.
        2. Handle LLM responses that may contain 'thought chain' tags (e.g., <think>...</think>).
        3. Return a specific dictionary format `{"content": ...}` instead of parsing JSON.

        Args:
            raw_text: The raw text fetched from the URL.
            query: The user query.

        Returns:
            A dictionary containing the cleaned content string.
        """
        # Calculate statistics for logging
        raw_len = len(raw_text)
        chunks = LLMService.split_chunks(raw_text)

        # Guard against empty chunks to avoid IndexError, though split_chunks usually returns data
        first_chunk = chunks[0] if chunks else ""

        logger.info(f"Jina Parser Input Content {raw_len} -> {len(first_chunk)}")

        # Prepare Prompt
        template = str(config.prompts.web_parser_html)
        final_prompt = template.format(user=query, info=first_chunk)

        # Call LLM
        answer = await LLMService.call_model(final_prompt)

        # Post-process: Remove <think> blocks common in reasoning models (e.g., DeepSeek)
        think_close_tag = '</think>'
        think_end_index = answer.rfind(think_close_tag)

        if think_end_index != -1:
            clean_text = answer[think_end_index + len(think_close_tag):].strip()
        else:
            clean_text = answer

        return {"content": clean_text}


class WebParserFactory:
    """Factory class to create the appropriate web parser based on URL or content type."""

    @staticmethod
    def get_content_type(url: str) -> str:
        """
        Retrieves the Content-Type header of a URL via a HEAD request.

        Args:
            url: The URL to inspect.

        Returns:
            The lowercase Content-Type string (e.g., 'application/pdf') or
            'none' if the request fails.
        """
        try:
            # Perform a HEAD request to fetch headers without downloading the body.
            cfg = config.web_parser
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=cfg.timeout,
                verify=cfg.ssl_verify
            )
            return response.headers.get('Content-Type', '').lower()
        except requests.RequestException:
            return "none"

    @staticmethod
    def create_parser(url: str) -> BaseWebParser:
        """
        Instantiates a parser suitable for the given URL.

        Selection Logic:
        1. Returns PDFParser if the URL contains '.pdf' or arXiv identifiers.
        2. Returns PDFParser if the remote Content-Type header contains 'pdf'.
        3. Returns JinaWebParser if configured to use Jina.
        4. Defaults to SimpleWebParser.

        Args:
            url: The target URL.

        Returns:
            An instance of a subclass of BaseWebParser.
        """
        # 1. Heuristic Check: Determine if URL string suggests a PDF
        # Note: 'arxiv.org/abs' is technically HTML, but logic dictates handling it via PDFParser
        is_pdf_url = (
            ".pdf" in url or
            'arxiv.org/abs' in url or
            'arxiv.org/pdf' in url
        )

        if is_pdf_url:
            return PDFParser()

        content_type = WebParserFactory.get_content_type(url)
        if "pdf" in content_type:
            return PDFParser()

        parser_timeout = config.web_parser.timeout

        if config.web_parser.use_jina:
            return JinaWebParser(timeout=parser_timeout)

        return SimpleWebParser(timeout=parser_timeout)


async def web_parse(link: str, query: str) -> Dict[str, Any]:
    """
    Unified entry point for web content parsing.

    Dynamically selects the appropriate parser based on the provided link
    and executes the parsing logic.

    Args:
        link (str): The URL of the web page to parse.
        query (str): The search query context used for extracting relevant
            information from the page.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed results (content,
        urls, score). Returns a standardized error dictionary if an unhandled
        exception occurs.
    """
    try:
        parser = WebParserFactory.create_parser(link)
        logger.info(f"Selected parser for {link}: {parser.__class__.__name__} ")

        return await parser.parse(link, query)

    except Exception as e:
        logger.error(f"Unhandled error in web_parse for link '{link}': {e}")

        return {"content": "System error during parsing"}
