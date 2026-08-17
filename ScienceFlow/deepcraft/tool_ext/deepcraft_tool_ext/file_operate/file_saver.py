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

import os
import aiofiles

from deepcraft_core.tool import BaseTool

FILE_SAVER_DESCRIPTION = """
Save content to a local file at a specified path.
Use this tool when you need to save text, code, or generated content to a file on the local filesystem.
The tool accepts content and a file path, and saves the content to that location.
"""

class FileSaver(BaseTool):
    """A tool for saving content to a file with specified path and mode.
        Attributes:
            name (str): The name of the tool, set to "file_saver".
            description (str): A description of the tool, taken from FILE_SAVER_DESCRIPTION.
            parameters (dict): A dictionary defining the parameters required for saving content to a file.
    """
    name: str = "file_saver"
    description: str = FILE_SAVER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "(required) The content to save to the file.",
            },
            "file_path": {
                "type": "string",
                "description": "(required) The path where the file should be saved, including filename and extension.",
            },
            "mode": {
                "type": "string",
                "description": "(optional) The file opening mode. Default is 'w' for write. Use 'a' for append.",
                "enum": ["w", "a"],
                "default": "w",
            },
        },
        "required": ["content", "file_path"],
    }

    async def execute(self, content: str, file_path: str, mode: str = "w") -> str:
        """
        Save content to a file at the specified path.

        Args:
            content (str): The content to save to the file.
            file_path (str): The path where the file should be saved.
            mode (str, optional): The file opening mode. Default is 'w' for write. Use 'a' for append.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            # Ensue the directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Write directly to the file
            async with aiofiles.open(file_path, mode, encoding="utf-8") as file:
                await file.write(content)

            return f"Content successfully saved to {file_path}"

        except Exception as e:
            return f"Error saving file: {str(e)}"


if __name__ == "__main__":
    import asyncio
    Model = FileSaver()
    data  = """
x = 10
y = 20
print(f'Sum: {x + y}')
    """
    file_path = './tmp.md'
    rst   = asyncio.run(Model.execute(data, file_path))
    print(rst)