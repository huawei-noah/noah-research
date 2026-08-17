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

"""Shared ``python-dotenv`` bootstrap for CLI and standalone scripts.

Where LLM/API-related values ultimately come from (conceptually):

1. **Shell / parent process** — keys already present in ``os.environ`` before dotenv runs.
2. **Repository root** — ``<repo_root>/.env`` is loaded first (same directory as a typical ``.venv``).
   Set ``SCIENCEFLOW_DOTENV_OVERRIDE=1`` so this file **replaces** existing env keys (useful when stale
   ``export API_*`` would otherwise block updates from ``.env``).
3. **Working-directory chain** — parameterless ``load_dotenv()`` finds ``.env`` starting from cwd;
   uses ``override=False`` so values from steps 1–2 are not overwritten by a nested file.

Additionally, **parallel** runs may inject ``API_*`` / ``BASE_*`` into the child process from manifest
YAML before ``python -m scienceflow.cli`` starts; those behave like exports in (1).

Finally, **resolved YAML** (e.g. ``resolved_config.yaml``) may embed ``agent.*`` credentials;
:func:`~scienceflow.config.settings.load_cfg` + ``_apply_env`` only fill missing fields from env, so
embedded YAML can still win unless a caller applies a stronger override (e.g. ensemble replay).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("scienceflow.env_bootstrap")


def _dotenv_override_enabled() -> bool:
    return os.environ.get("SCIENCEFLOW_DOTENV_OVERRIDE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def bootstrap_dotenv(*, repo_root: Path) -> None:
    """Load ``<repo_root>/.env`` then cwd ``.env`` (standard discovery).

    Mirrors ensemble-replay intent: repo-root secrets first, then cwd. When
    ``SCIENCEFLOW_DOTENV_OVERRIDE`` is truthy, the repo-root file may replace existing
    ``os.environ`` entries; the cwd pass always uses ``override=False``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(repo_root).expanduser().resolve()
    repo_env = root / ".env"
    override = _dotenv_override_enabled()
    load_dotenv(dotenv_path=repo_env, override=override)
    load_dotenv(override=False)
    if repo_env.is_file():
        logger.debug("dotenv: repo_root=%s override_first=%s", root, override)
