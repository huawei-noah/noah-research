# src/core/config.py
"""
Configuration Management for TridentMem
Loads API credentials from .env and manages system configuration
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载 .env 文件
load_dotenv()


@dataclass
class TemporalIndexConfig:
    """时间索引配置"""
    enable_multi_range: bool = True
    max_ranges_per_query: int = 10
    index_granularity: str = "hour"  # hour, day, week, month


@dataclass
class SemanticIndexConfig:
    """语义索引配置"""
    enable_dag: bool = True
    max_tag_depth: int = 5
    tag_similarity_threshold: float = 0.8
    enable_rl_alignment: bool = False


@dataclass
class ConsolidationConfig:
    """Embedding整合配置"""
    enable_consolidation: bool = True
    consolidation_interval_hours: int = 24
    min_memories_for_consolidation: int = 100
    batch_size: int = 1000


@dataclass
class TridentMemConfig:
    """TridentMem完整配置"""
    
    # ============ 从环境变量读取的 API 配置 ============
    # LLM 配置
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    
    # Embedding 配置
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_base_url: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536")))
    
    # ============ 功能配置（代码中设置）============
    # 存储路径
    storage_path: str = "./data"
    
    # 子配置
    temporal_config: TemporalIndexConfig = field(default_factory=TemporalIndexConfig)
    semantic_config: SemanticIndexConfig = field(default_factory=SemanticIndexConfig)
    consolidation_config: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    
    # 性能配置
    max_concurrent_queries: int = 10
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    def __post_init__(self):
        """验证配置"""
        # 检查必需的 API key
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY not set in environment variables")
        if not self.embedding_api_key:
            logger.warning("EMBEDDING_API_KEY not set in environment variables")
        
        # 确保存储路径存在
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """获取LLM客户端初始化参数"""
        kwargs = {
            "api_key": self.llm_api_key,
            "model": self.llm_model,
        }
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        return kwargs
    
    def get_embedding_kwargs(self) -> Dict[str, Any]:
        """获取Embedding客户端初始化参数"""
        kwargs = {
            "api_key": self.embedding_api_key,
            "model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }
        if self.embedding_base_url:
            kwargs["base_url"] = self.embedding_base_url
        return kwargs
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        if not self.llm_api_key:
            logger.error("LLM_API_KEY is required")
            return False
        if not self.embedding_api_key:
            logger.error("EMBEDDING_API_KEY is required")
            return False
        if self.embedding_dim <= 0:
            logger.error("EMBEDDING_DIM must be positive")
            return False
        return True
    
    def display(self) -> str:
        """显示配置信息（隐藏敏感信息）"""
        return f"""
TridentMem Configuration
========================
API Configuration:
  LLM Model: {self.llm_model}
  LLM Base URL: {self.llm_base_url or 'default'}
  LLM API Key: {'*' * 8 if self.llm_api_key else 'NOT SET'}
  
  Embedding Model: {self.embedding_model}
  Embedding Base URL: {self.embedding_base_url or 'default'}
  Embedding API Key: {'*' * 8 if self.embedding_api_key else 'NOT SET'}
  Embedding Dimension: {self.embedding_dim}

Storage Configuration:
  Storage Path: {self.storage_path}

Temporal Index:
  Multi-range: {self.temporal_config.enable_multi_range}
  Max Ranges: {self.temporal_config.max_ranges_per_query}
  Granularity: {self.temporal_config.index_granularity}

Semantic Index:
  DAG Enabled: {self.semantic_config.enable_dag}
  Max Tag Depth: {self.semantic_config.max_tag_depth}
  Tag Similarity: {self.semantic_config.tag_similarity_threshold}
  RL Alignment: {self.semantic_config.enable_rl_alignment}

Consolidation:
  Enabled: {self.consolidation_config.enable_consolidation}
  Interval: {self.consolidation_config.consolidation_interval_hours}h
  Min Memories: {self.consolidation_config.min_memories_for_consolidation}
  Batch Size: {self.consolidation_config.batch_size}

Performance:
  Max Concurrent Queries: {self.max_concurrent_queries}
  Cache Size: {self.cache_size_mb}MB
  Profiling: {self.enable_profiling}
"""


# 创建默认配置实例（单例模式）
_default_config: Optional[TridentMemConfig] = None

def get_config() -> TridentMemConfig:
    """获取默认配置实例（单例）"""
    global _default_config
    if _default_config is None:
        _default_config = TridentMemConfig()
    return _default_config


def set_config(config: TridentMemConfig) -> None:
    """设置全局配置实例"""
    global _default_config
    _default_config = config


# 向后兼容的别名
AgenticMemoryConfig = TridentMemConfig
Config = TridentMemConfig


__all__ = [
    'TemporalIndexConfig',
    'SemanticIndexConfig', 
    'ConsolidationConfig',
    'TridentMemConfig',
    'AgenticMemoryConfig',
    'Config',
    'get_config',
    'set_config',
]