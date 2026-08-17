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

from typing import List, Optional, Dict
from pydantic import Field, model_validator


class TokenUsageTracker:
    """Tracks token usage"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    usage_history: list = []

    @model_validator(mode="after")
    def initialize(self) -> "TokenUsageTracker":
        """Initialize the TokenUsageTracker instance.
        Returns:
            TokenUsageTracker: The initialized TokenUsageTracker instance.
        """
        self.usage_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        return self
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.usage_history.append(usage_record)

        return usage_record

    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }
        
    def reset(self):
        """Clear the usage history"""
        self.usage_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0