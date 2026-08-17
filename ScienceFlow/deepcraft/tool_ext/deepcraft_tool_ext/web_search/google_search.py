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

import asyncio
from typing import List

from googlesearch import search

from deepcraft_core.tool import BaseTool

GOOGLE_SEARCH_DESCRIPTION = """
Perform a Google search and return a list of relevant links.
Use this tool when you need to find information on the web, get up-to-date data, or research specific topics.
The tool returns a list of URLs that match the search query.
"""

class GoogleSearch(BaseTool):
    """A tool for performing Google searches.
        Attributes:
            name (str): The name of the tool, set to "google_search".
            description (str): A description of the tool, taken from GOOGLE_SEARCH_DESCRIPTION.
            parameters (dict): A dictionary defining the parameters required for a Google search.
    """
    name: str = "google_search"
    description: str = GOOGLE_SEARCH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to Google.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        Execute a Google search and return a list of URLs.

        Args:
            query (str): The search query to submit to Google.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            List[str]: A list of URLs matching the search query.
        """
        # Run the search in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        links = await loop.run_in_executor(
            None, lambda: list(search(query, num_results=num_results))
        )

        return links

if __name__ == "__main__":
    Model = GoogleSearch()
    rst   = asyncio.run(Model.execute("China news"))
    print(rst)
