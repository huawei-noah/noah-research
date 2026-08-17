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

from __future__ import annotations

# memory_retrieval_mixin.py
import os
import pickle
import logging
from typing import Optional, Dict, List, Any
from abc import ABC
from pydantic import Field
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryRetrievalMixin(ABC):
    """Mixin class for memory retrieval functionality"""

    # Vector retrieval configuration
    vector_db_path: str = Field(default="./vector_db", description="Vector database path")
    model_name: str = Field(default="./intfloat/multilingual-e5-large-instruct", description="Encoding model path")
    use_memory_retrieval: bool = Field(default=False, description="Whether to enable memory retrieval")
    retrieval_top_k: int = Field(default=3, description="Number of retrieval results")
    min_common_length: int = Field(default=100, description="Minimum length of common substring")
    save_to_vector_db: bool = Field(default=False, description="Whether to save to vector database")
    fusion_retrieval: bool = Field(default=False, description="Whether to use fusion retrieval")
    collection_name: str = Field(default="conversations", description="Vector database collection name")

    # Vector retrieval components (lazy initialization)
    _encoder: Optional[Any] = None
    _vector_db: Optional[Any] = None
    _clusters_info: Optional[Dict[str, Any]] = None
    _retrieval_initialized: bool = False

    def _ensure_directory_exists(self, path: str) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            directory = Path(path)
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    def _initialize_retrieval_components(self):
        """Initialize vector retrieval components"""
        if self.use_memory_retrieval and not self._retrieval_initialized:
            try:
                from deepcraft_core import ChromaVectorStorage, SentenceTransformerEncoder

                # Ensure vector database directory exists
                if not self._ensure_directory_exists(self.vector_db_path):
                    raise RuntimeError(f"Cannot create vector DB directory: {self.vector_db_path}")

                # Initialize encoder
                logger.info(f"Initializing encoder with model: {self.model_name}")
                self._encoder = SentenceTransformerEncoder(model=self.model_name)

                # Initialize vector database
                logger.info(f"Initializing vector DB at: {self.vector_db_path}")
                self._vector_db = ChromaVectorStorage(
                    collection_name=self.collection_name,
                    persist_directory=self.vector_db_path,
                    distance_metric="cosine"
                )

                # Check if it's a newly created database
                try:
                    # Try to get the number of documents in the collection
                    collection = self._vector_db.client.get_collection(self.collection_name)
                    doc_count = collection.count()
                    logger.info(f"Vector DB initialized with {doc_count} existing documents")
                except:
                    logger.info("Created new vector DB collection")

                # Load or initialize clustering information
                clusters_path = os.path.join(self.vector_db_path, "clusters_info.pkl")
                if os.path.exists(clusters_path):
                    try:
                        with open(clusters_path, 'rb') as f:
                            self._clusters_info = pickle.load(f)
                        logger.info(f"Loaded clustering info with {len(self._clusters_info.get('clusters', {}))} clusters")
                    except Exception as e:
                        logger.warning(f"Failed to load clusters info: {e}, initializing empty clusters")
                        self._clusters_info = {'clusters': {}, 'version': '1.0'}
                        self._save_clusters_info()
                else:
                    logger.info("No clustering info found, initializing empty clusters")
                    self._clusters_info = {'clusters': {}, 'version': '1.0'}
                    self._save_clusters_info()

                self._retrieval_initialized = True
                logger.info("Memory retrieval components initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize retrieval components: {e}")
                # Set to not use memory retrieval to avoid subsequent errors
                self.use_memory_retrieval = False
                raise

    def _save_clusters_info(self):
        """Save clustering information to file"""
        if self._clusters_info is not None:
            try:
                clusters_path = os.path.join(self.vector_db_path, "clusters_info.pkl")
                with open(clusters_path, 'wb') as f:
                    pickle.dump(self._clusters_info, f)
                logger.debug("Saved clusters info to file")
            except Exception as e:
                logger.error(f"Failed to save clusters info: {e}")

    def _find_longest_common_substring(self, text1: str, text2: str) -> str:
        """Find the longest common substring between two texts"""
        if len(text1) < self.min_common_length or len(text2) < self.min_common_length:
            return ""

        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        ending_pos = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        ending_pos = i

        if max_length >= self.min_common_length:
            return text1[ending_pos - max_length:ending_pos]
        return ""

    def _extract_unique_part(self, text: str, common_part: str) -> str:
        """Remove common part from text, keep unique part"""
        if not common_part or common_part not in text:
            return text

        idx = text.find(common_part)
        before = text[:idx].strip()
        after = text[idx + len(common_part):].strip()

        unique_parts = []
        if before:
            unique_parts.append(before)
        if after:
            unique_parts.append(after)

        return " ".join(unique_parts) if unique_parts else text

    async def _retrieve_similar_memories(self,
                                       query: str,
                                       retrieval_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """General memory retrieval method"""
        if not self.use_memory_retrieval:
            return []

        try:
            # Ensure retrieval components are initialized
            if not self._retrieval_initialized:
                self._initialize_retrieval_components()

            # Check if there is data to retrieve
            try:
                collection = self._vector_db.client.get_collection(self.collection_name)
                if collection.count() == 0:
                    logger.info("Vector DB is empty, no memories to retrieve")
                    return []
            except:
                logger.warning("Cannot check collection count, proceeding with retrieval")

            # Check if query contains common substring of any cluster
            query_optimized = query
            matched_cluster = None

            if self._clusters_info and 'clusters' in self._clusters_info:
                for cluster_label, cluster_data in self._clusters_info['clusters'].items():
                    common_substring = cluster_data.get('common_substring', '')
                    if common_substring and common_substring in query:
                        matched_cluster = (cluster_label, cluster_data)
                        query_optimized = self._extract_unique_part(query, common_substring)
                        logger.info(f"Matched cluster {cluster_label}, optimized query")
                        break

            # Create embedding for query
            query_embedding = await self._encoder.ask_embedding(input=query_optimized)
            if self.fusion_retrieval:
                # Perform retrieval
                results = self._vector_db.fusion_query(
                    query_embeddings=[query_embedding], query_text = query_optimized,
                    n_results=self.retrieval_top_k * 2 if matched_cluster else self.retrieval_top_k,
                    where=retrieval_filter  # Support for filter conditions
                )
            else:
                results = self._vector_db.query(
                    query_embeddings=[query_embedding],
                    n_results=self.retrieval_top_k * 2 if matched_cluster else self.retrieval_top_k,
                    where=retrieval_filter  # Support for filter conditions
                )

            # Check if there are results
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                logger.info("No matching memories found")
                return []

            # Process retrieval results
            retrieved_memories = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]

                # If there is a matching cluster, prioritize results from the same cluster
                if matched_cluster:
                    cluster_label = str(meta.get('cluster_label', -1))
                    if cluster_label != matched_cluster[0] and cluster_label != '-1':
                        continue

                memory = {
                    'metadata': meta,
                    'similarity': 1 - results['distances'][0][i],
                    'document': results['documents'][0][i] if 'documents' in results else None
                }
                retrieved_memories.append(memory)

                if len(retrieved_memories) >= self.retrieval_top_k:
                    break

            logger.info(f"Retrieved {len(retrieved_memories)} similar memories")
            return retrieved_memories

        except Exception as e:
            logger.error(f"Error during memory retrieval: {e}")
            return []

    async def _save_to_vector_db(self,
                            text: str,
                            metadata: Dict[str, Any],
                            doc_id: Optional[str] = None):
        """General method for saving to vector database"""
        if not self.save_to_vector_db or not self.use_memory_retrieval:
            return

        try:
            # Ensure retrieval components are initialized
            if not self._retrieval_initialized:
                self._initialize_retrieval_components()

            # Check if text has common substring with existing clusters
            text_optimized = text
            matched_cluster = None
            cluster_label = metadata.get('cluster_label', '-1')

            if self._clusters_info and 'clusters' in self._clusters_info:
                # If already has cluster label, check if cluster info needs updating
                if cluster_label != '-1' and cluster_label in self._clusters_info['clusters']:
                    cluster_data = self._clusters_info['clusters'][cluster_label]
                    common_substring = cluster_data.get('common_substring', '')
                    if common_substring and common_substring in text:
                        matched_cluster = (cluster_label, cluster_data)
                        text_optimized = self._extract_unique_part(text, common_substring)
                        logger.info(f"Matched existing cluster {cluster_label}, optimized text for embedding")

                # If no cluster label, check if it matches existing clusters
                else:
                    for cluster_label, cluster_data in self._clusters_info['clusters'].items():
                        common_substring = cluster_data.get('common_substring', '')
                        if common_substring and common_substring in text:
                            matched_cluster = (cluster_label, cluster_data)
                            text_optimized = self._extract_unique_part(text, common_substring)
                            # Update cluster label in metadata
                            metadata['cluster_label'] = cluster_label
                            metadata['unique_query'] = text_optimized
                            metadata['common_substring'] = common_substring
                            logger.info(f"Found matching cluster {cluster_label}, optimized text for embedding")
                            break

            # Create embedding (using optimized text)
            embedding = await self._encoder.ask_embedding(input=text_optimized)

            # Generate ID
            if not doc_id:
                import hashlib
                import time
                doc_id = f"{getattr(self, 'name', 'agent')}_{hashlib.md5(text.encode()).hexdigest()[:8]}_{int(time.time())}"

            # Add to vector database
            self._vector_db.add(
                embeddings=[embedding],
                documents=[text],  # Save original text
                metadata=[metadata],
                ids=[doc_id]
            )

            logger.info(f"Saved to vector DB with ID: {doc_id}")

            # Periodically save cluster info (if updated)
            if hasattr(self, '_clusters_updated') and self._clusters_updated:
                self._save_clusters_info()
                self._clusters_updated = False

        except Exception as e:
            logger.error(f"Error saving to vector DB: {e}")

    def get_vector_db_stats(self) -> Dict[str, Any]:
        """Get statistics of the vector database"""
        if not self.use_memory_retrieval:
            return {"enabled": False}

        try:
            if not self._retrieval_initialized:
                self._initialize_retrieval_components()

            collection = self._vector_db._client.get_collection(self.collection_name)
            stats = {
                "enabled": True,
                "initialized": True,
                "collection_name": self.collection_name,
                "document_count": collection.count(),
                "cluster_count": len(self._clusters_info.get('clusters', {})),
                "db_path": self.vector_db_path
            }
            return stats

        except Exception as e:
            return {
                "enabled": True,
                "initialized": False,
                "error": str(e)
            }
