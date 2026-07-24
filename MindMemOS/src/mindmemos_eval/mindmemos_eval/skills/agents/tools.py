"""Filesystem and shell tools for benchmark agents."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from .react import Tool

_INSTALL_RE = re.compile(
    r"^(?:sudo\s+)?(?:\w+=\S+\s+)*(?:"
    r"(?:python[\d.]*\s+-m\s+)?pip[\d.]*\s+(?:install|uninstall)"
    r"|uv\s+(?:pip\s+(?:install|uninstall|sync)|add|remove)"
    r"|(?:conda|mamba)\s+(?:install|remove|uninstall)"
    r"|poetry\s+(?:add|remove)"
    r"|pipenv\s+(?:install|uninstall)"
    r"|easy_install)\b",
    re.IGNORECASE,
)


def _blocked_install(command: str) -> bool:
    for segment in re.split(r"&&|\|\||[;\n|]", command):
        if _INSTALL_RE.search(segment.strip()):
            return True
    return False


def _is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            chunk = file.read(1024)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _readable_suffix(path: Path) -> bool:
    if path.suffix in {".py", ".txt", ".json", ".csv", ".md"}:
        return True
    if path.suffix in {".doc", ".docx", ".pdf", ".xlsx", ".pptx", ".ppt", ".xls"}:
        return False
    return not _is_binary_file(path)


class ReactAgentTools:
    """Core tools mounted into each benchmark case workdir."""

    def __init__(self, cwd: Path | str, python_path: Path | str | None = None) -> None:
        self.cwd = Path(cwd).resolve()
        self.python_path = Path(python_path).expanduser().absolute() if python_path else None
        self._shim_dir = self._build_python_shim() if self.python_path else None

    def _build_python_shim(self) -> str:
        shim = Path(tempfile.mkdtemp(prefix="mindmemos_skill_pyshim_"))
        for name in ("python", "python3"):
            wrapper = shim / name
            wrapper.write_text(f'#!/bin/sh\nexec "{self.python_path}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        return str(shim)

    def _resolve_in_cwd(self, path: str) -> Path | None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.cwd / resolved
        resolved = resolved.resolve()
        if resolved != self.cwd and self.cwd not in resolved.parents:
            return None
        return resolved

    def read(self, path: str) -> str:
        resolved = self._resolve_in_cwd(path)
        if resolved is None:
            return f"Error: access denied, {path} is outside the working directory"
        if not resolved.exists():
            return f"Error: File {resolved} not found"
        if not _readable_suffix(resolved):
            return f"Error: {resolved.suffix} file cannot be read as plain text."
        try:
            return resolved.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error: File {resolved} read error: {exc}"

    def write(self, path: str, content: str) -> str:
        resolved = self._resolve_in_cwd(path)
        if resolved is None:
            return f"Error: access denied, {path} is outside the working directory"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error writing file: {exc}"
        return f"Successfully wrote {len(content)} characters to {resolved}"

    def edit(self, path: str, original_text: str, replacement_text: str) -> str:
        resolved = self._resolve_in_cwd(path)
        if resolved is None:
            return f"Error: access denied, {path} is outside the working directory"
        if not resolved.exists():
            return f"Error: File {resolved} not found"
        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error: File {resolved} read error: {exc}"

        count = content.count(original_text)
        if count == 0:
            return f"Error: original_text not found in {resolved}"
        if count > 1:
            return f"Error: original_text matched {count} times in {resolved}, must be unique"
        try:
            resolved.write_text(content.replace(original_text, replacement_text), encoding="utf-8")
        except OSError as exc:
            return f"Error writing file: {exc}"
        return f"Successfully edited {resolved}"

    def shell(self, commands: list[str], timeout_ms: int | None = None) -> str:
        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120
        env = {**os.environ}
        if self._shim_dir:
            env["PATH"] = self._shim_dir + os.pathsep + env.get("PATH", "")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["UV_OFFLINE"] = "1"

        outputs: list[str] = []
        for command in commands:
            block = [f"Command: {command}"]
            if _blocked_install(command):
                block.append(
                    "Error: installing packages is disabled in this environment. "
                    "Required packages are preinstalled; import and use them directly."
                )
                outputs.append("\n".join(block))
                continue
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.cwd),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                block.append(f"Error: timed out after {int(timeout_sec)}s")
                outputs.append("\n".join(block))
                break
            except OSError as exc:
                block.append(f"Error: {exc}")
                outputs.append("\n".join(block))
                break

            if proc.stdout:
                block.append(f"stdout:\n{proc.stdout}")
            if proc.stderr:
                block.append(f"stderr:\n{proc.stderr}")
            block.append(f"exit_code: {proc.returncode}")
            outputs.append("\n".join(block))

        return "\n\n".join(outputs)

    def as_tools(self) -> list[Tool]:
        return [
            Tool(
                name="read",
                description="Read a text file's full contents. Returns an error for missing or non-text files.",
                func=self.read,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the working directory."}
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="write",
                description="Write or overwrite a text file, creating parent directories as needed.",
                func=self.write,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the working directory."},
                        "content": {"type": "string", "description": "Full text content to write."},
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="edit",
                description="Replace an exact, unique snippet of text in a file.",
                func=self.edit,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the working directory."},
                        "original_text": {"type": "string", "description": "Exact text to replace once."},
                        "replacement_text": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["path", "original_text", "replacement_text"],
                },
            ),
            Tool(
                name="shell",
                description="Run shell commands sequentially in the working directory.",
                func=self.shell,
                parameters={
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Shell commands to run in order.",
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "description": "Per-command timeout in milliseconds.",
                        },
                    },
                    "required": ["commands"],
                },
            ),
        ]
