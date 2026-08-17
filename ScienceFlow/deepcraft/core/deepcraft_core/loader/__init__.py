# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""
Document loader module for deepcraft Agent Core.

This module provides abstract base classes and concrete implementations
for loading documents from various sources and file formats.
"""

from .base_loader import (
    BaseLoader,
    File,
    DocxFile,
    PdfFile,
    TxtFile,
    JsonFile,
    HtmlFile,
    create_file,
    create_file_from_raw_bytes,
    strip_consecutive_newlines,
)

__all__ = [
    "BaseLoader",
    "File",
    "DocxFile",
    "PdfFile", 
    "TxtFile",
    "JsonFile",
    "HtmlFile",
    "create_file",
    "create_file_from_raw_bytes",
    "strip_consecutive_newlines",
]
