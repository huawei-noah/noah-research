# src/llm/client.py
"""
LLM Client for TridentMem
Now supports environment-based configuration
"""

import openai
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from ..core.config import TridentMemConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response data model"""
    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str
    response_time: float


class LLMClient:
    """Language model client for TridentMem"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None, 
        base_url: Optional[str] = None,
        config: Optional[TridentMemConfig] = None
    ):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key (optional, defaults to config)
            model: Model name (optional, defaults to config)
            base_url: API base URL (optional, defaults to config)
            config: TridentMemConfig instance (optional, creates default if None)
        """
        # Create default config if not provided
        if config is None:
            config = TridentMemConfig()
        
        # Load from config or explicit parameters
        self.api_key = api_key or config.llm_api_key
        self.model = model or config.llm_model
        self.base_url = base_url or config.llm_base_url
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Configuration parameters
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0
        
        logger.info(
            f"Initialized LLMClient: model={self.model}, "
            f"base_url={self.base_url or 'default'}"
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion
        
        Args:
            messages: Message list [{"role": "user", "content": "..."}]
            temperature: Temperature parameter (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            LLMResponse object
            
        Raises:
            Exception: If all retries fail
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response_time = time.time() - start_time
                response_content = response.choices[0].message.content
                
                return LLMResponse(
                    content=response_content,
                    usage=response.usage.model_dump() if response.usage else {},
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason,
                    response_time=response_time
                )
                
            except Exception as e:
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise e
    
    def generate_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        default_response: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            default_response: Fallback response if parsing fails
            max_retries: Maximum retry attempts for JSON parsing
            
        Returns:
            Parsed JSON object
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Retry loop for JSON parsing
        for attempt in range(max_retries):
            try:
                response = self.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Try to parse JSON
                result = self._extract_json_from_response(response.content)
                
                if result is not None:
                    return result
                
                # If parsing failed and this is not the last attempt
                if attempt < max_retries - 1:
                    logger.info(f"JSON parsing failed, retrying ({attempt + 1}/{max_retries})")
                    
                    # Add clarification message
                    if attempt == 0:
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": "Please provide valid JSON format only, without markdown or explanations."
                        })
                    
                    # Slightly increase temperature for variation
                    temperature = min(temperature + 0.1, 0.5)
                    
            except Exception as e:
                logger.error(f"Error generating JSON response (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    if default_response is not None:
                        logger.warning("All retries failed, returning default response")
                        return default_response
                    else:
                        raise e
        
        # All retries exhausted
        if default_response is not None:
            return default_response
        else:
            return {
                "error": "JSON parsing failed after all retries",
                "attempts": max_retries
            }
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from response with multiple parsing strategies
        
        Args:
            content: Response content
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        import re
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code blocks
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Regex to find JSON objects
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # Strategy 4: Find content between first { and last }
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"JSON parsing failed for content: {content[:200]}...")
        return None
    
    def generate_text_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate plain text response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.content
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "max_retries": self.max_retries,
            "timeout": self.timeout
        }