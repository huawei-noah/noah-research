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

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepcraft_core.message import ROLE_TYPE, Role

from ..base import BaseAgent, AgentState
from .react_prompts import STUCK_PROMPT, THINKING_COMPLETE_PROMPT

logger = logging.getLogger(__name__)

class ReActAgent(BaseAgent):
    """An agent implementing the ReAct (Reasoning and Acting) framework.
        Attributes:
            nextStepPrompt (Optional[str]): Prompt used to determine the next action.
            maxSteps (Optional[int]): Maximum number of steps before termination.
            currentStep (Optional[int]): Current step in the execution process.
            duplicateThreshold (Optional[int]): Threshold for detecting duplicate actions.
    """
    nextStepPrompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Execution control
    maxSteps: Optional[int] = Field(default=3, description="Maximum steps before termination")
    currentStep: Optional[int] = Field(default=0, description="Current step in execution")
    duplicateThreshold: Optional[int] = Field(default=2, description="Threshold for duplicate detection")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields for flexibility in subclasses
    )

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        shouldAct = await self.think()

        if not shouldAct:
            return THINKING_COMPLETE_PROMPT

        return await self.act()

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.updateMemory(Role.USER, request)

        results: List[str] = []
        async with self.stateContext(AgentState.RUNNING):
            while (
                self.currentStep < self.maxSteps and self.state != AgentState.FINISHED
            ):
                self.currentStep += 1
                logger.info(f"Executing step {self.currentStep}/{self.maxSteps}")
                stepResult = await self.step()

                # Check for stuck state
                if self.isStuck():
                    self.handleStuckState()

                results.append(f"Step {self.currentStep}: {stepResult}")

            if self.currentStep >= self.maxSteps:
                results.append(f"Terminated: Reached max steps ({self.maxSteps})")

        return "\n".join(results) if results else "No steps executed"

    def isStuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        lastMessage = None
        for Msg in self.memory.messages:
            if Msg.role == Role.ASSISTANT:
                lastMessage = Msg.content

        if not lastMessage:
            return False

        # Count identical content occurrences
        duplicateCount = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
                if msg.role == Role.ASSISTANT and msg.content == lastMessage
        )

        return duplicateCount >= self.duplicateThreshold

    def handleStuckState(self):
        """Handle stuck state by adding a prompt to change strategy"""
        self.nextStepPrompt = f"{STUCK_PROMPT}\n{self.nextStepPrompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {STUCK_PROMPT}")

    @asynccontextmanager
    async def stateContext(self, newState: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(newState, AgentState):
            raise ValueError(f"Invalid state: {newState}")

        previousState = self.state
        self.state = newState

        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previousState  # Revert to previous state

