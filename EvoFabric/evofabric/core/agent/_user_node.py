# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
from typing import Dict

from pydantic import Field

from ..graph import AsyncNode
from ..typing import State, StateDelta, UserMessage
from ...logger import get_logger


logger = get_logger()


class UserNode(AsyncNode):
    """
    A node that gets user input from terminal and passes it to the graph.
    """

    prompt_message: str = Field(
        default="Please enter your input: ",
        description="Message to display when asking for user input"
    )

    input_key: str = Field(
        default="user_input",
        description="Key in the state where the user input will be stored"
    )

    @classmethod
    def _get_field_descriptions(cls) -> Dict[str, str]:
        """Get descriptions for pydantic fields"""
        return {
            "prompt_message": "Message to display when asking for user input",
            "input_key": "Key in the state where the user input will be stored"
        }

    async def __call__(self, state: State) -> StateDelta:
        """
        Get user input from terminal and return it as state delta.

        Args:
            state: Current state (not used in this node)

        Returns:
            StateDelta containing user input
        """
        try:
            # Display prompt message
            logger.info(self.prompt_message)

            # Get user input from terminal in async way
            # Use asyncio.to_thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input)

            if user_input:
                user_input = user_input.strip()
                # Return user input as a new message in state delta
                return {
                    "messages": [UserMessage(content=user_input)]
                }
            else:
                # Return empty message if input is empty
                return {"messages": []}

        except (EOFError, KeyboardInterrupt):
            # Handle EOF or keyboard interrupt gracefully
            logger.info("\nUser input interrupted.")
            return {"messages": []}
        except Exception as e:
            logger.info(f"Error getting user input: {e}")
            return {"messages": []}
