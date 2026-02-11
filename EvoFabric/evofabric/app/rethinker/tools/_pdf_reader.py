# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import traceback

import fitz
import requests
from requests.exceptions import RequestException

from evofabric.logger import get_logger
from .._config import config

logger = get_logger()


def download_and_read_pdf(url: str) -> str:
    """
    Synchronously downloads a PDF from a URL and extracts its text content.

    This function handles standard PDF URLs and special arXiv abstract URLs,
    transforming them into direct PDF links. It downloads the file content,

    then uses PyMuPDF (fitz) to parse the PDF and extract all text.

    Args:
        url: The URL of the PDF file or an arXiv abstract page.
             Example: "https://arxiv.org/abs/2106.07682"

    Returns:
        A string containing the extracted text from the PDF. On failure,
        it returns a descriptive error string like "Failed to read the PDF"
        or "Failed to get the PDF content".
    """
    try:
        if "arxiv.org/abs/" in url:
            paper_id = url.split('/')[-1]
            url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        headers = {
            "Accept": "application/pdf",
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=config.web_parser.timeout,
            verify=config.web_parser.ssl_verify
        )
        response.raise_for_status()

        pdf_content = response.content
        if not pdf_content:
            logger.error("Content is None or empty")
            return "Failed to get the PDF content"

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)

        return text.strip()

    except RequestException as e:
        logger.error(f"Request failed: {e},{traceback.format_exc()}")
        return "Failed to read the PDF"
    except Exception as e:
        logger.error(f"Read failed: {e},{traceback.format_exc()}")
        return "Failed to read the PDF"
