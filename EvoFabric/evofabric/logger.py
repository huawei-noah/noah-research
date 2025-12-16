# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

_LOGGER = None


def _get_loguru_logger():
    try:
        from loguru import logger
        return logger
    except ImportError:
        raise ValueError(
            "EvoFabric logger requires loguru to be installed."
            "Please install loguru by running `pip install loguru`"
        )


def get_logger():
    return _LoggerProxy()


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


class _LoggerProxy:
    def __init__(self):
        object.__setattr__(self, "_dummy", True)

    @property
    def _real(self):
        return _LOGGER or _get_loguru_logger()

    def __getattribute__(self, name: str):
        return getattr(object.__getattribute__(self, "_real"), name)

    def debug(self, msg, *args, **kwargs):
        return self._real.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self._real.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._real.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._real.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        return self._real.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._real.critical(msg, *args, **kwargs)