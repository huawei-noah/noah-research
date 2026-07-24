# src/embeddings/embedding_manager.py
"""
Embedding Manager for TridentMem - Caching and batch processing
Now supports environment-based configuration
"""

from typing import List, Optional, Dict
import numpy as np
import hashlib
import json
from pathlib import Path
import logging

from .embedding_client import EmbeddingClient
from ..core.config import TridentMemConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Embedding manager with caching and batch processing
    Manages embedding generation for episodes and semantic memories
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        cache_enabled: bool = True,
        cache_path: Optional[Path] = None,
        config: Optional[TridentMemConfig] = None 
    ):
        """
        Initialize embedding manager
        
        Args:
            api_key: OpenAI API key (optional, from env if not provided)
            model: Embedding model name (optional, from env if not provided)
            base_url: Custom API base URL (optional, from env if not provided)
            embedding_dim: Embedding dimension (optional, from env if not provided)
            cache_enabled: Whether to enable caching
            cache_path: Path to cache file (optional)
            config: TridentMemConfig instance (optional, creates default if None)
        """
        # Initialize client (will use env config if parameters not provided)
        self.client = EmbeddingClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            embedding_dim=embedding_dim,
            config=config
        )
        
        # Cache configuration
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_path = cache_path
        
        # Get embedding dimension
        self.dimension = self.client.embedding_dim
        
        # Load cache from disk
        if self.cache_enabled and self.cache_path:
            self._load_cache()
        
        logger.info(
            f"Initialized EmbeddingManager: model={self.client.model}, "
            f"dimension={self.dimension}, cache_enabled={cache_enabled}"
        )
    
    def get_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Get embedding for a single text
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        # Handle empty text
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for: {text[:50]}...")
                return self.cache[cache_key]
        
        # Generate embedding
        try:
            embedding_list = self.client.embed_text(text)
            embedding = np.array(embedding_list, dtype=np.float32)
            
            # Save to cache
            if self.cache_enabled:
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                
                # Periodically save cache to disk
                if len(self.cache) % 100 == 0 and self.cache_path:
                    self._save_cache()
            
            logger.debug(f"Generated embedding for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (with caching)
        
        Args:
            texts: List of texts
            use_cache: Whether to use cache
            
        Returns:
            Embeddings matrix (shape: [n_texts, dimension])
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Separate cached and uncached texts
        cached_embeddings: Dict[int, np.ndarray] = {}
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        
        if use_cache and self.cache_enabled:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    # Empty text gets zero vector
                    cached_embeddings[i] = np.zeros(self.dimension, dtype=np.float32)
                    continue
                
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    cached_embeddings[i] = self.cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = [t for t in texts if t and t.strip()]
            uncached_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            # Handle empty texts
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    cached_embeddings[i] = np.zeros(self.dimension, dtype=np.float32)
        
        # Initialize result array
        all_embeddings = [None] * len(texts)
        
        # Generate uncached embeddings
        if uncached_texts:
            try:
                response = self.client.embed_texts(uncached_texts)
                
                for idx, embedding_list in zip(uncached_indices, response.embeddings):
                    embedding = np.array(embedding_list, dtype=np.float32)
                    all_embeddings[idx] = embedding
                    
                    # Cache the result
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(texts[idx])
                        self.cache[cache_key] = embedding
                
                logger.info(
                    f"Generated {len(uncached_texts)} new embeddings, "
                    f"{len(cached_embeddings)} from cache"
                )
                
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                # Fallback: zero vectors for failed texts
                for idx in uncached_indices:
                    all_embeddings[idx] = np.zeros(self.dimension, dtype=np.float32)
        
        # Fill in cached embeddings
        for idx, embedding in cached_embeddings.items():
            all_embeddings[idx] = embedding
        
        # Save cache if we generated new embeddings
        if self.cache_enabled and self.cache_path and uncached_texts:
            self._save_cache()
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score [-1, 1]
        """
        vec1 = embedding1.tolist() if isinstance(embedding1, np.ndarray) else embedding1
        vec2 = embedding2.tolist() if isinstance(embedding2, np.ndarray) else embedding2
        
        return self.client.cosine_similarity(vec1, vec2)
    
    def compute_similarities_batch(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple embeddings
        
        Args:
            query_embedding: Query embedding (shape: [dimension])
            embeddings: Embeddings matrix (shape: [n, dimension])
            
        Returns:
            Similarity scores (shape: [n])
        """
        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        vecs = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        similarities = self.client.batch_cosine_similarity(query_vec, vecs)
        return np.array(similarities, dtype=np.float32)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text (MD5 hash)"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        if not self.cache_path or not self.cache_path.exists():
            return
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in cache_data.items():
                self.cache[key] = np.array(value, dtype=np.float32)
            
            logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk"""
        if not self.cache_path:
            return
        
        try:
            # Ensure directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {
                key: value.tolist()
                for key, value in self.cache.items()
            }
            
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.cache.clear()
        if self.cache_path and self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        return len(self.cache)
    
    def get_stats(self) -> Dict:
        """Get manager statistics"""
        return {
            "model": self.client.model,
            "dimension": self.dimension,
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled,
            "cache_path": str(self.cache_path) if self.cache_path else None,
        }