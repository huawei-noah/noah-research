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

import logging
import os
import sys
from pathlib import Path

from scienceflow.utils.workspace_interaction_log import colorize_interaction_message


def _scienceflow_log_color_enabled(interaction_log_color: bool | None) -> bool:
    """Match :attr:`~scienceflow.config.settings.Config.scienceflow_interaction_log_color` resolution."""
    if interaction_log_color is not None:
        return bool(interaction_log_color)
    v = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_COLOR", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return True


class ScienceFlowPlainFormatter(logging.Formatter):
    """``asctime | level | name | message`` (same shape as historical scienceflow.log)."""

    def __init__(self, *, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=datefmt,
        )


class ScienceFlowColorFormatter(logging.Formatter):
    """Same as :class:`ScienceFlowPlainFormatter` with colored interaction messages."""

    def __init__(self, *, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<8}"
        raw = record.getMessage()
        msg = colorize_interaction_message(raw, enabled=True)
        return f"{asctime} | {levelname} | {record.name} | {msg}"


def setup_logging(
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    name: str = "scienceflow",
    *,
    console: bool = True,
    interaction_log_color: bool | None = None,
) -> logging.Logger:
    """Configure the ``scienceflow`` logger.

    Args:
        log_dir: If set, append ``scienceflow.log`` under this directory.
        level: Log level for new handlers.
        name: Logger name (default ``scienceflow``).
        console: If True, also emit logs to stderr. If False, file-only
            (used by REPL so the interactive UI is not mixed with log lines).
        interaction_log_color: When true, apply the same ANSI rules as workspace
            ``interaction.log`` to the message field (``scienceflow_interaction_log_color`` /
            ``SCIENCEFLOW_INTERACTION_LOG_COLOR``). ``None`` reads env then defaults to on.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    use_color = _scienceflow_log_color_enabled(interaction_log_color)
    fmt: logging.Formatter = (
        ScienceFlowColorFormatter() if use_color else ScienceFlowPlainFormatter()
    )

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "scienceflow.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
