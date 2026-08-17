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

"""REPL-native long-horizon stage-capture solver."""

__all__ = ["LnrSolver"]


def __getattr__(name: str):
    if name == "LnrSolver":
        from scienceflow.solver.lnr.solver import LnrSolver

        return LnrSolver
    raise AttributeError(name)
