# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import asyncio
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, PrivateAttr

from evofabric.app.rethinker.prompts._default_prompts import (
    CRITIC_TWICE_USER_PROMPT,
    CRITIC_USER_PROMPT,
    GUIDED_SUMMARY_PROMPT,
    SELECTOR_ITERATION_USER_PROMPT,
    SELECTOR_USER_PROMPT,
    SOLVER_TWICE_USER_PROMPT, SOLVER_USER_PROMPT,
    WEB_PARSER_PROMPT_HTML,
    WEB_PARSER_PROMPT_PDF
)
from evofabric.core.factory import BaseComponent, ComponentFactory

_DEFAULT_CHAT_TEMPLATE = """{%- for message in tool_logs -%}
{%- if message['role'] == 'system' -%}
{{ message['content'] }}
{%- endif -%}
{%- if message['role'] == 'user' -%}
<｜User｜> {{ message['content'] }}
{%- endif -%}
{%- if message['role'] == 'assistant' -%}
<｜Assistant｜>
{%- if message.get('reasoning_content') -%}
<think>{{ message['reasoning_content'] }}</think>
{%- endif -%}
{{ message['content'] }}
{%- endif -%}
{%- if message['role'] == 'tool_call' -%}
<code>{{ message['content'] }}</code>
  {%- elif message['role'] == 'tool_call_result' -%}
<execution_results>{{ message['content'] }}</execution_results>
  {%- endif -%}
{%- endfor -%}
"""


class LLMConfig(BaseModel):
    model_name: str = "your-model-name"
    """Name of the LLM model to use."""

    api_key: str = "your-api-key"
    """API key used for OpenAI-compatible backends. Typically required for OpenAI-compatible backends."""

    base_url: str = "your-base-url"
    """Base URL of the LLM API endpoint."""

    csb_token: Optional[str] = None
    """CSB authentication token used for CSB-token–based authorization."""

    max_retries: int = 3
    """Maximum number of retry attempts for transient request failures."""

    timeout: Optional[int] = None
    """Request timeout in seconds before aborting the LLM call."""

    top_p: Optional[float] = None
    """Nucleus sampling parameter controlling cumulative probability mass."""

    temperature: Optional[float] = None
    """Sampling temperature controlling the randomness of generated tokens."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate; backend default is used if None."""

    extra_body: dict = Field(default_factory=dict)
    """Additional backend-specific request body parameters."""

    fast_think: bool = False
    """Enable fast-thinking mode by appending '/no_think' to the last message."""

    output_logp: bool = False
    """Whether to return token-level log probabilities in the response."""

    stop_condition: Optional[str] = None
    """Streaming stop condition string that terminates generation when matched."""

    http_client_kwargs: Optional[dict] = Field(default_factory=dict)
    """Keyword arguments passed to the underlying OpenAI HTTP client."""

    stream: bool = True
    """Whether to enable streaming mode; must be True if stop_condition is set."""

    def create_client_kwargs(self) -> dict:
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
        if self.csb_token:
            client_kwargs['default_headers'] = {"csb-token": self.csb_token}  # type: ignore
        return client_kwargs

    def create_inference_kwargs(self) -> dict:
        inference_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.extra_body:
            inference_kwargs["extra_body"] = self.extra_body  # type: ignore

        if self.output_logp:
            inference_kwargs["top_logprobs"] = 5
            inference_kwargs["logprobs"] = True

        if self.max_tokens is not None:
            inference_kwargs["max_tokens"] = self.max_tokens
        return inference_kwargs


class WebParserConfig(BaseModel):
    llm_input_max_char: int = 1024 * 120
    """Maximum allowed character budget per document after web parsing."""

    model: str = "pangu_auto"
    """Backend model name used for web content parsing or downstream inference."""

    use_jina: bool = True
    """Whether to enable Jina-based web content parsing."""

    jina_api_key: Optional[str] = None
    """API key for the Jina service, required when use_jina is enabled."""

    ssl_verify: bool = False
    """Whether to verify SSL certificates during HTTP requests."""

    timeout: int = 60
    """Network request timeout in seconds for web content fetching."""

    show_url_content_max_char: int = 200
    """Maximum number of characters retained per URL; values below zero disable URL content inclusion."""


class WebSearchConfig(BaseModel):
    serper_api_key: Optional[str] = None
    """API key for the Serper service, required to enable web search requests."""

    retries: int = 3
    """Maximum number of retry attempts for failed web search requests."""

    ssl_verify: bool = False
    """Whether to verify SSL certificates during web search HTTP requests."""

    timeout: int = 10
    """Request timeout in seconds for a single web search call."""


class SolutionConfig(BaseModel):
    solver_model: str = "pangu_auto"
    """LLM backend model used during the solution generation phase."""

    critic_model: str = "pangu_auto"
    """LLM backend model used during the critic evaluation phase."""

    summary_model: str = "pangu_auto"
    """LLM backend model used for generating guided summaries."""

    selector_model: str = "pangu_auto"
    """LLM backend model used during the selection phase."""

    selector_iteration: int = 1
    """Number of iterations for the selection phase to refine choices."""

    stop_condition: str = '<code[^>]*>((?:(?!<code).)*?)</code>'
    """Regex pattern to extract code blocks from LLM output."""

    selection_condition: str = '<select[^>]*>\\s*Response\\s*([1-5])\\s*</select>'
    """Regex pattern to extract selected response indices from LLM output."""

    answer_condition: str = '<answer[^>]*>((?:(?!<answer).)*?)</answer>'
    """Regex pattern to extract the final answer content from LLM output."""

    max_agent_step: int = 30
    """Maximum number of reasoning or action steps allowed per agent execution."""

    tool_timeout: int = 300
    """Timeout in seconds for invoking a single tool or external service."""

    max_empty_response: int = 5
    """Maximum consecutive empty responses allowed before aborting agent execution."""

    chat_template: Optional[str] = _DEFAULT_CHAT_TEMPLATE
    """Template string for serializing multi-round conversations; not used as an LLM prompt."""


class ExpConfig(BaseModel):
    input_file_path: str = "data/HLE_all.json"
    """Path to the input dataset file used for the experiment."""

    exp_name: str = "HLE"
    """Experiment name, used as a subdirectory under the output root."""

    output_root: Optional[str] = "output"
    """Root directory for saving experiment outputs; if None, no cache will be written."""

    max_question_thread_limit: int = 20
    """Maximum number of questions allowed to run concurrently."""

    max_node_thread_limit: int = 20
    """Maximum number of nodes allowed to run concurrently during execution."""


class StructureConfig(BaseModel):
    num_parallel: int = 5
    """Number of solution candidates generated in parallel for each task."""

    num_solution_iteration: int = 2
    """Number of iterations for solution generation per candidate."""

    num_critic_iteration: int = 4
    """Number of iterations for the critic/reflection phase to evaluate solutions."""


class PromptConfig(BaseModel):
    web_parser_pdf: str = WEB_PARSER_PROMPT_PDF
    """Prompt template used for parsing PDF web content."""

    web_parser_html: str = WEB_PARSER_PROMPT_HTML
    """Prompt template used for parsing HTML web content."""

    solver_user_prompt: str = SOLVER_USER_PROMPT
    """User prompt template for the initial solution generation phase."""

    solver_twice_user_prompt: str = SOLVER_TWICE_USER_PROMPT
    """User prompt template for second-pass solution generation."""

    critic_user_prompt: str = CRITIC_USER_PROMPT
    """User prompt template for the critique/evaluation phase."""

    critic_twice_user_prompt: str = CRITIC_TWICE_USER_PROMPT
    """User prompt template for second-pass critique/evaluation."""

    guided_summary_prompt: str = GUIDED_SUMMARY_PROMPT
    """Prompt template for generating guided solution summaries."""

    selector_user_prompt: str = SELECTOR_USER_PROMPT
    """User prompt template for solution selection phase."""

    selector_iteration_user_prompt: str = SELECTOR_ITERATION_USER_PROMPT
    """User prompt template for iterative selection refinement."""


class GraphConfig(BaseModel):
    llm_resources: dict[str, LLMConfig] = Field(default_factory=dict)
    """Dictionary of available LLM backends; keys are model names, values are corresponding LLMConfig objects."""

    web_parser: WebParserConfig = Field(default_factory=WebParserConfig)
    """Configuration for web content parsing."""

    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    """Configuration for web search operations."""

    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    """Settings for solution generation, critique, summary, and selection phases."""

    structure: StructureConfig = Field(default_factory=StructureConfig)
    """Configuration of the iterative reasoning graph, including parallelism and selector behavior."""

    exp: ExpConfig = Field(default_factory=ExpConfig)
    """Experiment-level configuration including I/O paths, output directories, and threading limits."""

    prompts: PromptConfig = Field(default_factory=PromptConfig)
    """Prompt templates used throughout the experiment for parsing, solving, critiquing, and selecting."""


class Config(BaseModel):
    """Manager for experiment configurations.

    This class handles loading and accessing the full experiment
    configuration, either from a YAML file or an existing GraphConfig
    instance. It provides convenient access to sub-configurations
    such as LLM resources, web parsing/search, solution settings,
    structure, experiment I/O, and prompts.

    It also manages a global semaphore for controlling node-level
    concurrency during execution.

    Usage:
        config = Config()
        config.load("path/to/config.yaml")
        graph_conf = config.graph
        llm_models = config.llm_resources
        semaphore = config.get_semaphore()
    """
    _config: Optional[GraphConfig] = PrivateAttr(default=None)

    _loaded: bool = PrivateAttr(default=False)

    _config_path: Optional[str] = PrivateAttr(default=None)

    _max_concurrency: Optional[int] = PrivateAttr(default=None)

    _async_sem: Optional[asyncio.Semaphore] = PrivateAttr(default=False)

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "Config is not loaded yet.\n"
                "Use config.load('your-config-path') to load the yaml config file.\n"
                "Or use config.loads(config: GraphConfig) to load a config instance.\n"
            )

    @property
    def graph(self) -> GraphConfig:
        self._check_loaded()
        return self._config

    @property
    def llm_resources(self) -> dict[str, LLMConfig]:
        self._check_loaded()
        return self._config.llm_resources

    @property
    def web_parser(self) -> WebParserConfig:
        self._check_loaded()
        return self._config.web_parser

    @property
    def web_search(self) -> WebSearchConfig:
        self._check_loaded()
        return self._config.web_search

    @property
    def solution(self) -> SolutionConfig:
        self._check_loaded()
        return self._config.solution

    @property
    def structure(self) -> StructureConfig:
        self._check_loaded()
        return self._config.structure

    @property
    def exp(self) -> ExpConfig:
        self._check_loaded()
        return self._config.exp

    @property
    def prompts(self) -> PromptConfig:
        self._check_loaded()
        return self._config.prompts

    def load(self, config_path: Union[str, Path]) -> None:
        """Load experiment configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)

        raw = yaml.safe_load(config_path.read_text())
        self._config = GraphConfig.model_validate(raw)
        self._config_path = str(config_path)
        self._loaded = True

    def loads(self, config: GraphConfig) -> None:
        """Load experiment configuration from an existing GraphConfig object.

        Args:
            config: Pre-constructed GraphConfig instance.
        """
        self._config = config
        self._loaded = True

    def get_semaphore(self) -> asyncio.Semaphore:
        """Get a global semaphore for node-level concurrency control.

        Returns:
            An asyncio.Semaphore object limiting concurrent node execution.
        """
        self._check_loaded()
        if self._max_concurrency is None:
            self._max_concurrency = self.exp.max_node_thread_limit
            self._async_sem = asyncio.Semaphore(self._max_concurrency)
        return self._async_sem


config = Config()  # Global singleton instance for managing experiment configuration
