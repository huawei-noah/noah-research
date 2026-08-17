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

"""Shared utilities for automatic dependency detection and installation."""

from __future__ import annotations

import re

MODULE_TO_PACKAGE: dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "Bio": "biopython",
    "gi": "PyGObject",
    "wx": "wxPython",
    "serial": "pyserial",
    "usb": "pyusb",
    "Crypto": "pycryptodome",
    "magic": "python-magic",
    "jose": "python-jose",
    "dateutil": "python-dateutil",
    "git": "gitpython",
    "nufft": "pynufft",
    "shapefile": "pyshp",
}

_RE_MISSING_MODULE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s+No module named ['\"]([^'\"]+)['\"]"
)


def parse_missing_package(stderr: str) -> str | None:
    """Extract the pip package name from a ``ModuleNotFoundError`` traceback.

    Returns the mapped pip package name, or ``None`` if no import error is found.
    """
    m = _RE_MISSING_MODULE.search(stderr)
    if not m:
        return None
    module = m.group(1).split(".")[0]
    return MODULE_TO_PACKAGE.get(module, module)
