# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
import inspect
import os
import threading
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Optional

from grpc import FutureTimeoutError
from loguru import logger
from pydantic import BaseModel

from ._web_parse import web_parse
from ._web_search import web_search


class CodeResponse(BaseModel):
    output: str
    error: Optional[str]
    execution_time: float


class OutputCapture:
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()

    def write(self, data: str):
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()
        self.stderr.flush()

    def get_stdout(self) -> str:
        return self.stdout.getvalue()

    def get_stderr(self) -> str:
        return self.stderr.getvalue()

    def close(self):
        self.stdout.close()
        self.stderr.close()


class ThreadOutputManager:
    def get_capture(self) -> OutputCapture:
        return OutputCapture()


def create_sync_wrapper(async_func):
    if not inspect.iscoroutinefunction(async_func):
        return async_func

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper


output_manager = ThreadOutputManager()


def _execute_code_safely(code: str, timeout: int):
    capture = output_manager.get_capture()

    def run_code():
        module = types.ModuleType("dynamic_module")
        available_tools = {
            'web_search': web_search,
            'web_parse': web_parse,
        }
        wrapped_tools = {
            name: create_sync_wrapper(func) for name, func in available_tools.items()
        }

        module.__dict__.update({
            'print': lambda *args, **kwargs: capture.write(
                ' '.join(str(arg) for arg in args) + ('\n' if kwargs.get('end', '\n') else '')
            ),
            '__builtins__': __builtins__,
            'sys': type('sys', (), {
                'stdout': capture,
                'stderr': capture,
                'stdin': None,
            })(),
        })
        module.__dict__.update(wrapped_tools)
        exec(code, module.__dict__)
        return capture.get_stdout(), capture.get_stderr()

    error = None
    output_value = None
    error_value = None
    start_time = time.time()
    single_executor = ThreadPoolExecutor(max_workers=1)

    try:
        future = single_executor.submit(run_code)
        output_value, error_value = future.result(timeout=timeout)

    except FutureTimeoutError:
        error = f"Execution timed out after {timeout} seconds"
        logger.warning(f"Code execution timeout: {timeout}s")
        error_value = error

    except SystemExit as se:
        error = f"Code called sys.exit({se.code})"
        if not capture.stderr.closed:
            capture.stderr.write(error)
        logger.warning(f"Code triggered SystemExit: {error}\n\n-----\n{code}")
        error_value = error

    except Exception as e:
        error = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

        if not capture.stderr.closed:
            capture.stderr.write(error)
        logger.warning(f"Code execution error: {error}\n\n-----\n{code}")
        error_value = error

    finally:
        if not capture.stdout.closed or not capture.stderr.closed:
            if output_value is None:
                output_value = capture.get_stdout()
            if error_value is None or error_value != error:
                error_value = capture.get_stderr()
        capture.close()
        single_executor.shutdown(wait=False)

    execution_time = time.time() - start_time

    return execution_time, output_value, error_value if error_value else None


class ExecutionRecord(BaseModel):
    context: dict
    code: str
    output: Optional[str]
    error: Optional[str]


executor = ThreadPoolExecutor(max_workers=1000)


async def execute_python_code(code: str, timeout: int = 300) -> CodeResponse:
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    try:
        execution_time, output, error = await loop.run_in_executor(executor, _execute_code_safely, code, timeout)
    except Exception as e:
        error = f"Execution failed: {str(e)}"
        output = ""
        logger.error(f"Unexpected error: {error}", exc_info=True)
        execution_time = loop.time() - start_time

    return CodeResponse(
        output=output,
        error=error,
        execution_time=execution_time,
    )
