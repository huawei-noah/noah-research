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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepcraft_core.storage.kv_storage import BaseKeyValueStorage


class JsonKeyValueStorage(BaseKeyValueStorage):
    r"""A concrete implementation of the :obj:`BaseKeyValueStorage` using JSON
    files. Allows for persistent storage of records in a human-readable format.

    Args:
        path (Path, optional): Path to the desired JSON file. If `None`, a
            default path `./chat_history.json` will be used.
            (default: :obj:`None`)
        mode (str, optional): 'w' (write), 'a' (append)
    """

    def __init__(
        self,
        path: Optional[str] = "./chat_history.json",
        mode: Optional[str] = "a",
        ) -> None:
        self.json_path = Path(path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.touch()
        if mode == 'w':
            self.clear()

    def save(self, records: List[Dict[str, Any]]) -> None:
        r"""Saves a batch of records to the key-value storage system.

        Args:
            records (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary represents a unique record to be stored.
        """
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("a", encoding='utf-8') as f:
            f.writelines(
                [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
            )

    def load(self) -> List[Dict[str, Any]]:
        r"""Loads all stored records from the key-value storage system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                represents a stored record.
        """
        if not self.json_path.exists():
            return []
        with self.json_path.open("r", encoding='utf-8') as f:
            return [
                json.loads(r)
                for r in f.readlines()
            ]

    def clear(self) -> None:
        r"""Removes all records from the key-value storage system."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w"):
            pass
