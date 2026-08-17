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
from typing import Optional

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext
from browser_use.dom.service import DomService
from pydantic import Field, field_validator

from deepcraft_core.tool import BaseTool, ToolResult

BROWSER_USE_DESCRIPTION = """
Interact with a web browser to perform various actions such as navigation, element interaction,
content extraction, and tab management. Supported actions include:
- 'navigate': Go to a specific URL
- 'click': Click an element by index
- 'input_text': Input text into an element
- 'screenshot': Capture a screenshot
- 'get_html': Get page HTML content
- 'get_text': Get text content of the page
- 'read_links': Get all links on the page
- 'execute_js': Execute JavaScript code
- 'scroll': Scroll the page
- 'switch_tab': Switch to a specific tab
- 'new_tab': Open a new tab
- 'close_tab': Close the current tab
- 'refresh': Refresh the current page
"""

BROWSER_USE_TOOL_ACTION_ENUM = [
"navigate", "click", "input_text", "screenshot", "get_html",
"get_text", "execute_js", "scroll", "switch_tab", "new_tab",
"close_tab", "refresh",
]

class BrowserUseTool(BaseTool):
    """A tool for browser use.
    Attributes:
        name (str): The name of the tool, set to "browser_use".
        description (str): A description of the tool, taken from BROWSER_USE_DESCRIPTION.
        parameters (dict): A dictionary defining the parameters required for different browser actions.
        lock (asyncio.Lock): A lock for synchronizing access to browser operations.
        browser (Optional[BrowserUseBrowser]): The browser instance associated with this tool.
        context (Optional[BrowserContext]): The browser context associated with this tool.
        dom_service (Optional[DomService]): The DOM service associated with this tool.

        """

    name: str = "browser_use"
    description: str = BROWSER_USE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": BROWSER_USE_TOOL_ACTION_ENUM,
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'navigate' or 'new_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click' or 'input_text' actions",
            },
            "text": {"type": "string", "description": "Text for 'input_text' action"},
            "script": {
                "type": "string",
                "description": "JavaScript code for 'execute_js' action",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll' action",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "navigate": ["url"],
            "click": ["index"],
            "input_text": ["index", "text"],
            "execute_js": ["script"],
            "switch_tab": ["tab_id"],
            "new_tab": ["url"],
            "scroll": ["scroll_amount"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")

        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            # Use Chrome command-line arguments to set the window size and position
            browser_config = BrowserConfig(
                headless=False,
                disable_security=True,
            )
            self.browser = BrowserUseBrowser(browser_config)

        if self.context is None:
            self.context = await self.browser.new_context()
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        script: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action
            script: JavaScript code for execution
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                if action == "navigate":
                    if not url:
                        return ToolResult(error="URL is required for 'navigate' action")

                    await context.navigate_to(url)
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "click":
                    if index is None:
                        return ToolResult(error="Index is required for 'click' action")
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")

                    await context._input_text_element_node(element, text)

                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "screenshot":
                    screenshot = await context.take_screenshot(full_page=True)
                    return ToolResult(
                        output=f"Screenshot captured (base64 length: {len(screenshot)})",
                        system=screenshot,
                    )

                elif action == "get_html":
                    html = await context.get_page_html()
                    truncated = html[:2000] + "..." if len(html) > 2000 else html
                    return ToolResult(output=truncated)

                elif action == "get_text":
                    text = await context.execute_javascript("document.body.innerText")
                    return ToolResult(output=text)

                elif action == "read_links":
                    links = await context.execute_javascript(
                        "document.querySelectorAll('a[href]').forEach((elem) => {if (elem.innerText) "
                        "{console.log(elem.innerText, elem.href)}})"
                    )
                    return ToolResult(output=links)

                elif action == "execute_js":
                    if not script:
                        return ToolResult(
                            error="Script is required for 'execute_js' action"
                        )
                    result = await context.execute_javascript(script)
                    return ToolResult(output=str(result))

                elif action == "scroll":
                    if scroll_amount is None:
                        return ToolResult(
                            error="Scroll amount is required for 'scroll' action"
                        )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {scroll_amount});"
                    )
                    direction = "down" if scroll_amount > 0 else "up"
                    return ToolResult(
                        output=f"Scrolled {direction} by {abs(scroll_amount)} pixels"
                    )

                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "new_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'new_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with URL {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            # Close the browser context if it exists
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None

            # Close the browser instance if it exists
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())

            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

if __name__ == "__main__":
    Model = BrowserUseTool()
    rst   = asyncio.run(Model.execute(
            action="navigate",
            url="",
        )
    )