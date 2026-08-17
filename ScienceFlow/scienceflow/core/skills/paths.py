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

from __future__ import annotations

from pathlib import Path

DEFAULT_SKILL_LIBRARY_REL = Path(".scienceflow") / "skills"


def default_skill_library_dir(repo_root: Path) -> Path:
    """Return the repo-level Markdown skill library directory."""
    return repo_root / DEFAULT_SKILL_LIBRARY_REL
