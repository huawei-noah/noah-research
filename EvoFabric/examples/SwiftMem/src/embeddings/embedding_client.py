# src/embeddings/embedding_client.py
"""
Embedding Client for TridentMem - OpenAI API based
Now supports environment-based configuration
"""

import openai
import numpy as np
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..core.config import TridentMemConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """Embedding API response"""
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    response_time: float


class EmbeddingClient:
    """OpenAI Embedding API client"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        config: Optional[TridentMemConfig] = None
    ):
        """
        Initialize embedding client
        
        Args:
            api_key: OpenAI API key (optional, defaults to config)
            model: Model name (optional, defaults to config)
            base_url: Custom API base URL (optional, defaults to config)
            embedding_dim: Embedding dimension (optional, defaults to config)
            config: TridentMemConfig instance (optional, creates default if None)
        """
        # Create default config if not provided
        if config is None:
            config = TridentMemConfig()
        
        # Load from config or explicit parameters
        self.api_key = api_key or config.embedding_api_key
        self.model = model or config.embedding_model
        self.base_url = base_url or config.embedding_base_url
        self.embedding_dim = embedding_dim or config.embedding_dim
        
        # Initialize OpenAI client
        if self.base_url:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.timeout = 30.0  # seconds
        self.batch_size = 100  # max texts per batch
        
        logger.info(
            f"Initialized EmbeddingClient: model={self.model}, "
            f"dimension={self.embedding_dim}, base_url={self.base_url or 'default'}"
        )
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            EmbeddingResponse with embeddings and metadata
            
        Raises:
            Exception: If API call fails after max retries
        """
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                usage={"prompt_tokens": 0, "total_tokens": 0},
                model=self.model,
                response_time=0.0
            )
        
        start_time = time.time()
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        timeout=self.timeout
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Accumulate usage
                    if response.usage:
                        usage_dict = response.usage.model_dump()
                        total_usage["prompt_tokens"] += usage_dict.get("prompt_tokens", 0)
                        total_usage["total_tokens"] += usage_dict.get("total_tokens", 0)
                    
                    logger.debug(f"Batch {i//self.batch_size + 1}: {len(batch)} texts embedded")
                    break
                    
                except Exception as e:
                    logger.warning(
                        f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        sleep_time = self.retry_delay * (2 ** attempt)
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                        raise e
        
        response_time = time.time() - start_time
        
        logger.info(
            f"Generated {len(all_embeddings)} embeddings in {response_time:.2f}s, "
            f"tokens used: {total_usage['total_tokens']}"
        )
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=total_usage,
            model=self.model,
            response_time=response_time
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector (list of floats)
        """
        response = self.embed_texts([text])
        return response.embeddings[0] if response.embeddings else []
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score [-1, 1]
            
        Raises:
            ValueError: If vector dimensions don't match
        """
        if len(vector1) != len(vector2):
            raise ValueError(
                f"Vector dimensions don't match: {len(vector1)} vs {len(vector2)}"
            )
        
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return float(dot_product / (norm_v1 * norm_v2))
    
    def batch_cosine_similarity(
        self,
        query_vector: List[float],
        vectors: List[List[float]]
    ) -> List[float]:
        """
        Calculate cosine similarity between query and multiple vectors
        
        Args:
            query_vector: Query vector
            vectors: List of vectors
            
        Returns:
            List of similarity scores
        """
        if not vectors:
            return []
        
        query = np.array(query_vector)
        matrix = np.array(vectors)
        
        # Batch compute cosine similarities
        dot_products = np.dot(matrix, query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        
        # Avoid division by zero
        similarities = np.divide(
            dot_products,
            norms,
            out=np.zeros_like(dot_products),
            where=norms != 0
        )
        
        return similarities.tolist()
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        v = np.array(vector)
        norm = np.linalg.norm(v)
        
        if norm == 0:
            return vector
        
        return (v / norm).tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information"""
        return {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "batch_size": self.batch_size,
        }