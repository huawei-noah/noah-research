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

"""Small formatting helpers."""

from __future__ import annotations


def _fmt_num(x: float | int | None) -> str:
    """Format a number to at most 5 significant digits for display."""
    if x is None:
        return ""
    try:
        return f"{float(x):.5g}"
    except (TypeError, ValueError):
        return str(x)
