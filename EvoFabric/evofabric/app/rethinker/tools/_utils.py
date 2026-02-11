# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import traceback

from evofabric.logger import get_logger

logger = get_logger()


class RetryHandler:
    @staticmethod
    async def execute(func, *args, max_retries=3, delay=1, **kwargs):
        last_exception = None
        last_traceback = None
        for i in range(max_retries):
            try:
                result = await func(*args, **kwargs)
                if result:
                    return result
            except Exception as e:
                last_exception = e
                last_traceback = traceback.format_exc()
                logger.warning(f"Attempt {i + 1}/{max_retries} failed for {func.__name__}: {e}")

            if i < max_retries - 1:
                await asyncio.sleep(delay)

        if last_exception:
            logger.error(f"All retries failed for {func.__name__}\nTraceback: {last_traceback}")
        return None, last_exception
