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

from typing import Mapping
import re

from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any
import traceback
import sys 
import gc

class PythonREPL:
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
        self,
        user_ns: Mapping[str, Any],
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.user_ns = user_ns
        self.timeout = timeout
        self.shell: InteractiveShell | None = None
        self._init_shell()

    def _init_shell(self) -> None:
        """Initialize a persistent InteractiveShell once."""
        if self.shell is None:
            self.shell = InteractiveShell(
                user_ns=self.user_ns,
                colors="NoColor",
            )
            # disable history and Out[...] cache to avoid memory bloat
            self.shell.history_manager.enabled = False
            self.shell.displayhook.cache_size = 0

    #def reset(self) -> None:
    #    """Explicit reset (fresh shell, but same namespace)."""
    #    self.close()
    #    self._init_shell()

    def __call__(self, query: str) -> str:
        """Use the tool and return observation"""
        # NOTE: The timeout error will be caught by the InteractiveShell
        if self.shell is None:
            raise AssertionError("Shell not initialized; call reset() first.")
        old_show = self.shell.showtraceback
        try:
            self.shell.showtraceback = lambda *a, **k: None
            # Capture all output
            with io.capture_output() as captured:
                res = self.shell.run_cell(query, store_history=False)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            self.shell.showtraceback = old_show

        exc = getattr(res, "error_in_exec", None) or getattr(res, "error_before_exec", None)

        error_text = ""
        if exc:
            error_text = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))

        cap_stdout = captured.stdout.strip()

        if not cap_stdout:
            cap_stdout = "[Executed Successfully with No Output, Did you forget to print?]"

        # replace potentially sensitive filepath
        # e.g., File /mint/mint/tools/python_tool.py:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
        # with File <filepath>:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
        # use re
        cap_stdout = re.sub(
            # r"File (/mint/)mint/tools/python_tool.py:(\d+)",
            r"File (.*)mint/tools/python_tool.py:(\d+)",
            r"File <hidden_filepath>:\1",
            cap_stdout,
        )
        if len(cap_stdout) > 2000:
            cap_stdout = cap_stdout[:2000] + "...\n[Output Truncated]"
        
        #self.user_ns.update(self.shell.user_ns)
        gc.collect()
            
        return cap_stdout, error_text

    #def close(self) -> None:
    #    # Don’t call IPython internals during interpreter shutdown.
    #    self.shell = None

    def __enter__(self):
        self._init_shell()      # Just ensure a shell exists, but don't reset/destroy it 
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def __del__(self):
        # Stay quiet during shutdown; avoid heavy work here.
        try:
            # sys.is_finalizing() is available in modern Python
            if getattr(sys, "is_finalizing", lambda: False)() or getattr(sys, "meta_path", None) is None:
                return
        except Exception as e:
            print(f'[WARNING] {repr(e)}')
            return
        self.shell = None