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

"""ESTRA magent helpers for LNR resource-time sidecar evidence."""

from scienceflow.solver.lnr.estra_magent.join_gate import (
    build_join_packet,
    load_join_packets,
    write_join_packet,
)
from scienceflow.solver.lnr.estra_magent.prompt_blocks import format_magent_recommendations

__all__ = [
    "build_join_packet",
    "format_magent_recommendations",
    "load_join_packets",
    "write_join_packet",
]
