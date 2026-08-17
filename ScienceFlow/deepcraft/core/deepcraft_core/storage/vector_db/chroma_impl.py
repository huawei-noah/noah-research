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

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import logging

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from deepcraft_core.storage.vector_db import BaseVectorStorage

# Setup logging
logger = logging.getLogger(__name__)

class ChromaVectorStorage(BaseVectorStorage):
    r"""A concrete implementation of the :obj:`BaseVectorStorage` using
    ChromaDB. Provides persistent vector storage with efficient similarity search.
    Also supports BM25 fusion retrieval for combining vector and keyword-based search.

    Args:
        collection_name (str, optional): Name of the ChromaDB collection.
            (default: "default_collection")
        persist_directory (str, optional): Directory to persist the database.
            If None, data will be stored in memory only.
            (default: "./chroma_db")
        distance_metric (str, optional): Distance metric for similarity search.
            Options: "cosine", "l2", "ip" (inner product).
            (default: "cosine")
        enable_bm25 (bool, optional): Whether to enable BM25 fusion retrieval.
            (default: True)
        chinese_stopwords (Optional[List[str]]): Custom Chinese stopwords list.
            If None, will use default Chinese stopwords.
        enable_advanced_tokenization (bool, optional): Whether to use advanced
            tokenization with jieba and NLTK. (default: True)
    """

    def __init__(
        self,
        collection_name: str = "default_collection",
        persist_directory: Optional[str] = "./chroma_db",
        distance_metric: str = "cosine",
        enable_bm25: bool = True,
        chinese_stopwords: Optional[List[str]] = None,
        enable_advanced_tokenization: bool = True,
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Please install it with: pip install chromadb"
            )

        if enable_bm25 and not BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 is not installed. Please install it with: pip install rank_bm25"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self.enable_bm25 = enable_bm25
        self.enable_advanced_tokenization = enable_advanced_tokenization
        
        # BM25 related attributes
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_ids = []

        # Tokenization setup
        self._setup_tokenization(chinese_stopwords)

        # Initialize ChromaDB client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection with specific error handling
        collection_exists = self._collection_exists(collection_name)
        
        if collection_exists:
            logger.info(f"Loading existing ChromaDB collection: {collection_name}")
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Successfully loaded collection with {self.collection.count()} documents")
        else:
            logger.info(f"Creating new ChromaDB collection: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            logger.info(f"Successfully created new collection: {collection_name}")

        # Initialize BM25 index if enabled
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection with the given name already exists."""
        try:
            existing_collections = self.client.list_collections()
            return any(col.name == collection_name for col in existing_collections)
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False

    def _setup_tokenization(self, chinese_stopwords: Optional[List[str]] = None) -> None:
        """Setup tokenization components and stopwords."""
        # Default Chinese stopwords (basic set)
        default_chinese_stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要',
            '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '这些', '那些',
            '什么', '怎么', '为什么', '哪里', '什么时候', '怎样', '多少', '哪个', '谁', '如何', '能够', '应该', '可以', '需要',
            '但是', '然后', '因为', '所以', '如果', '虽然', '虽说', '尽管', '不过', '而且', '或者', '还是', '无论', '不管',
            '啊', '呀', '吧', '呢', '吗', '哦', '嗯', '嗳', '嘿', '哈', '呵', '唉', '嘻', '哟', '呦', '咦',
        ]
        
        # Try to load Chinese stopwords from file if not provided via parameter
        if chinese_stopwords is None:
            try:
                # Get the directory of the current file
                current_dir = Path(__file__).parent
                stopwords_file = current_dir / "stopwords.txt"
                
                if stopwords_file.exists():
                    with open(stopwords_file, 'r', encoding='utf-8') as f:
                        file_stopwords = [line.strip() for line in f.readlines() if line.strip()]
                    if file_stopwords:  # Only use file stopwords if file is not empty
                        self.chinese_stopwords = file_stopwords
                    else:
                        self.chinese_stopwords = default_chinese_stopwords
                else:
                    self.chinese_stopwords = default_chinese_stopwords
            except (IOError, UnicodeDecodeError) as e:
                # Fall back to default if file reading fails
                print(f"Warning: Could not read stopwords.txt file: {e}. Using default Chinese stopwords.")
                self.chinese_stopwords = default_chinese_stopwords
        else:
            # Use provided stopwords
            self.chinese_stopwords = chinese_stopwords
        
        # English stopwords
        self.english_stopwords = set()
        if NLTK_AVAILABLE:
            try:
                self.english_stopwords = set(stopwords.words('english'))
            except LookupError:
                # Download stopwords if not available
                try:
                    nltk.download('stopwords', quiet=True)
                    nltk.download('punkt', quiet=True)
                    self.english_stopwords = set(stopwords.words('english'))
                except:
                    # Fallback to basic English stopwords
                    self.english_stopwords = {
                        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                        'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
                        'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out',
                        'many', 'then', 'them', 'can', 'would', 'my', 'no', 'him', 'his', 'has', 'her'
                    }

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))

    def _chinese_tokenizer(self, text: str) -> List[str]:
        """
        Tokenizes Chinese text using the jieba library.
        
        Args:
            text (str): The input Chinese text to be tokenized.
            
        Returns:
            list: A list of tokens after tokenization, excluding stopwords.
        """
        if not JIEBA_AVAILABLE:
            # Fallback to simple tokenization
            return [token.strip() for token in re.findall(r'\S+', text) if token.strip()]
            
        tokens = jieba.lcut_for_search(text)  # search-engine mode; suited for tokenization when building inverted indexes for search engines, with finer granularity
        tokens = [token.strip() for token in tokens 
                 if token.strip() != "" and token.strip() not in self.chinese_stopwords]
        return tokens

    def _english_tokenizer(self, text: str) -> List[str]:
        """
        Tokenizes English text using NLTK.
        
        Args:
            text (str): The input English text to be tokenized.
            
        Returns:
            list: A list of tokens after tokenization, excluding stopwords.
        """
        if not NLTK_AVAILABLE:
            # Fallback to simple regex tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            return [token for token in tokens if token not in self.english_stopwords]
        
        try:
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens 
                     if token.isalnum() and token not in self.english_stopwords]
            return tokens
        except LookupError:
            # Fallback if NLTK data is not available
            tokens = re.findall(r'\b\w+\b', text.lower())
            return [token for token in tokens if token not in self.english_stopwords]

    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization that handles multiple languages."""
        if not self.enable_advanced_tokenization:
            # Simple fallback tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            return tokens
        
        # Check if text contains Chinese characters
        if self._is_chinese(text):
            return self._chinese_tokenizer(text)
        else:
            return self._english_tokenizer(text)

    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from all documents in the collection."""
        if not self.enable_bm25:
            return

        # Get all documents from the collection
        all_docs = self.collection.get(include=["documents"])
        
        if all_docs["documents"]:
            self.bm25_documents = all_docs["documents"]
            self.bm25_ids = all_docs["ids"]
            
            # Tokenize documents for BM25
            tokenized_docs = [self._tokenize(doc) for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
        else:
            self.bm25_documents = []
            self.bm25_ids = []
            self.bm25_index = None

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        r"""Adds embeddings with associated documents and metadata to the vector storage.

        Args:
            embeddings (List[List[float]]): A list of embedding vectors.
            documents (List[str]): A list of documents corresponding to the embeddings.
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each document.
            ids (Optional[List[str]]): Optional IDs for each document. If not provided,
                IDs will be auto-generated.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        elif len(metadata) != len(documents):
            raise ValueError("Number of metadata entries must match number of documents")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Convert metadata to ChromaDB format (ensure all values are strings, numbers, or booleans)
        processed_metadata = []
        for meta in metadata:
            processed_meta = {}
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    processed_meta[key] = value
                else:
                    processed_meta[key] = str(value)
            processed_metadata.append(processed_meta)

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=processed_metadata,
            ids=ids
        )

        # Rebuild BM25 index after adding documents
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[Any]]:
        r"""Queries the vector storage for similar embeddings.

        Args:
            query_embeddings (List[List[float]]): Query embedding vectors.
            n_results (int): Number of results to return. (default: 10)
            where (Optional[Dict[str, Any]]): Metadata filter conditions.
            where_document (Optional[Dict[str, str]]): Document content filter conditions.

        Returns:
            Dict[str, List[Any]]: A dictionary containing 'ids', 'distances', 
                'documents', and 'metadatas' for the most similar results.
        """
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "ids": results["ids"],
            "distances": results["distances"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }

    def fusion_query(
        self,
        query_embeddings: List[List[float]],
        query_text: str,
        n_results: int = 10,
        vector_weight: float = 0.7,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Performs fusion retrieval combining vector similarity and BM25 keyword search using relative score fusion.
        
        This implementation follows the relative score fusion approach as described in:
        [1] 
        [2] S. Bruch, S. Gai, and A. Ingber, “An Analysis of Fusion Functions for Hybrid Retrieval,” ACM Trans. Inf. Syst., vol. 42, no. 1, Aug. 2023, doi: 10.1145/3596512.

        Args:
            query_embeddings (List[List[float]]): Query embedding vectors.
            query_text (str): Query text for BM25 search.
            n_results (int): Number of results to return. (default: 10)
            vector_weight (float): Weight for vector similarity scores (alpha parameter). (default: 0.7)
            where (Optional[Dict[str, Any]]): Metadata filter conditions.
            where_document (Optional[Dict[str, str]]): Document content filter conditions.

        Returns:
            Dict[str, List[Any]]: Fusion search results with combined scores.
        """
        if not self.enable_bm25:
            return self.query(query_embeddings, n_results, where, where_document)

        if self.bm25_index is None:
            return self.query(query_embeddings, n_results, where, where_document)

        # Get all documents to ensure proper normalization
        total_docs = self.count()
        if total_docs == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        
        # Vector search - get all documents for proper normalization
        vector_results = self.query(
            query_embeddings=query_embeddings,
            n_results=total_docs,
            where=where,
            where_document=where_document
        )

        # BM25 search - get scores for all documents
        query_tokens = self._tokenize(query_text)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Create mapping of document ID to vector similarity
        vector_score_map = {}
        for i in range(len(vector_results["ids"][0])):
            doc_id = vector_results["ids"][0][i]
            vector_distance = vector_results["distances"][0][i]
            # Convert distance to similarity
            vector_similarity = 1 - vector_distance
            vector_score_map[doc_id] = vector_similarity
        
        # Create mapping of document ID to BM25 score
        bm25_score_map = {}
        for i, doc_id in enumerate(self.bm25_ids):
            if doc_id in vector_score_map:  # Only include docs that passed vector filters
                bm25_score_map[doc_id] = bm25_scores[i]
        
        # Extract scores for normalization
        vector_similarities = list(vector_score_map.values())
        bm25_score_values = list(bm25_score_map.values())
        
        # Min-max normalization function
        def min_max_normalization(scores):
            if not scores or len(scores) == 0:
                return scores
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)  # All scores are equal
            return [(score - min_score) / (max_score - min_score) for score in scores]
        
        # Normalize scores
        normalized_vector_scores = min_max_normalization(vector_similarities)
        normalized_bm25_scores = min_max_normalization(bm25_score_values)
        
        # Create normalized score mappings
        doc_ids = list(vector_score_map.keys())
        normalized_vector_map = {doc_ids[i]: normalized_vector_scores[i] for i in range(len(doc_ids))}
        normalized_bm25_map = {doc_id: normalized_bm25_scores[i] 
                              for i, doc_id in enumerate(bm25_score_map.keys())}
        
        # Fusion by relative score: (1-alpha)*bm25_score + alpha*vector_score
        alpha = vector_weight
        fusion_results = []
        
        for i in range(len(vector_results["ids"][0])):
            doc_id = vector_results["ids"][0][i]
            
            # Get normalized scores
            norm_vector_score = normalized_vector_map.get(doc_id, 0.0)
            norm_bm25_score = normalized_bm25_map.get(doc_id, 0.0)
            
            # Relative score fusion
            fusion_score = (1 - alpha) * norm_bm25_score + alpha * norm_vector_score
            
            fusion_results.append({
                "id": doc_id,
                "document": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i],
                "fusion_score": fusion_score,
                "vector_similarity": vector_score_map[doc_id],
                "bm25_score": bm25_score_map.get(doc_id, 0.0),
                "normalized_vector_score": norm_vector_score,
                "normalized_bm25_score": norm_bm25_score
            })

        # Sort by fusion score and take top n_results
        fusion_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        fusion_results = fusion_results[:n_results]

        # Format results to match standard query format
        return {
            "ids": [[result["id"] for result in fusion_results]],
            "distances": [[1 - result["fusion_score"] for result in fusion_results]],
            "documents": [[result["document"] for result in fusion_results]],
            "metadatas": [[result["metadata"] for result in fusion_results]]
        }

    def bm25_query(
        self,
        query_text: str,
        n_results: int = 10,
    ) -> Dict[str, List[Any]]:
        """
        Performs BM25 keyword search only.

        Args:
            query_text (str): Query text for BM25 search.
            n_results (int): Number of results to return. (default: 10)

        Returns:
            Dict[str, List[Any]]: BM25 search results.
        """
        if not self.enable_bm25 or self.bm25_index is None:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        query_tokens = self._tokenize(query_text)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get document indices sorted by BM25 score
        doc_scores = [(i, score) for i, score in enumerate(bm25_scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n_results
        top_results = doc_scores[:n_results]
        
        # Get metadata for top results
        top_ids = [self.bm25_ids[i] for i, _ in top_results]
        doc_data = self.collection.get(ids=top_ids, include=["documents", "metadatas"])
        
        # Create mapping for preserving order
        id_to_data = {}
        for i, doc_id in enumerate(doc_data["ids"]):
            id_to_data[doc_id] = {
                "document": doc_data["documents"][i],
                "metadata": doc_data["metadatas"][i]
            }
        
        # Build results in score order
        results = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]]
        }
        
        for doc_idx, score in top_results:
            doc_id = self.bm25_ids[doc_idx]
            results["ids"][0].append(doc_id)
            results["distances"][0].append(1 - score)  # Convert score to distance
            results["documents"][0].append(id_to_data[doc_id]["document"])
            results["metadatas"][0].append(id_to_data[doc_id]["metadata"])
        
        return results

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        r"""Retrieves documents from the vector storage.

        Args:
            ids (Optional[List[str]]): Specific document IDs to retrieve.
            where (Optional[Dict[str, Any]]): Metadata filter conditions.
            limit (Optional[int]): Maximum number of results to return.
            offset (Optional[int]): Number of results to skip.

        Returns:
            Dict[str, List[Any]]: A dictionary containing 'ids', 'documents',
                'metadatas', and 'embeddings' for the matching documents.
        """
        results = self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )

        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "embeddings": results["embeddings"]
        }

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        r"""Updates existing documents in the vector storage.

        Args:
            ids (List[str]): IDs of documents to update.
            embeddings (Optional[List[List[float]]]): New embeddings for the documents.
            documents (Optional[List[str]]): New document content.
            metadata (Optional[List[Dict[str, Any]]]): New metadata for the documents.
        """
        # Process metadata to ensure ChromaDB compatibility
        processed_metadata = None
        if metadata is not None:
            processed_metadata = []
            for meta in metadata:
                processed_meta = {}
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        processed_meta[key] = value
                    else:
                        processed_meta[key] = str(value)
                processed_metadata.append(processed_meta)

        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=processed_metadata
        )

        # Rebuild BM25 index after updating documents
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Deletes documents from the vector storage.

        Args:
            ids (Optional[List[str]]): Specific document IDs to delete.
            where (Optional[Dict[str, Any]]): Metadata filter conditions for deletion.
        """
        self.collection.delete(ids=ids, where=where)

        # Rebuild BM25 index after deleting documents
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def count(self) -> int:
        r"""Returns the number of documents in the vector storage.

        Returns:
            int: The total number of documents stored.
        """
        return self.collection.count()

    def clear(self) -> None:
        r"""Removes all documents from the vector storage."""
        # ChromaDB doesn't have a direct clear method, so we delete the collection
        # and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )

        # Clear BM25 index
        if self.enable_bm25:
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_ids = []

    def reset(self) -> None:
        r"""Resets the entire database by deleting and recreating the collection.
        This is useful for testing or when you want to start fresh.
        """
        self.clear()

    def set_chinese_stopwords(self, stopwords: List[str]) -> None:
        """
        Update Chinese stopwords list and rebuild BM25 index if needed.
        
        Args:
            stopwords (List[str]): New Chinese stopwords list.
        """
        self.chinese_stopwords = stopwords
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def add_chinese_stopwords(self, stopwords: List[str]) -> None:
        """
        Add additional Chinese stopwords to existing list.
        
        Args:
            stopwords (List[str]): Additional Chinese stopwords to add.
        """
        self.chinese_stopwords.extend(stopwords)
        if self.enable_bm25:
            self._rebuild_bm25_index()

    def get_tokenization_info(self) -> Dict[str, Any]:
        """
        Get information about tokenization setup.
        
        Returns:
            Dict[str, Any]: Tokenization configuration and capabilities.
        """
        return {
            "advanced_tokenization_enabled": self.enable_advanced_tokenization,
            "jieba_available": JIEBA_AVAILABLE,
            "nltk_available": NLTK_AVAILABLE,
            "chinese_stopwords_count": len(self.chinese_stopwords),
            "english_stopwords_count": len(self.english_stopwords),
            "supported_languages": ["chinese", "english"] if JIEBA_AVAILABLE and NLTK_AVAILABLE else ["basic"]
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the current collection including BM25 and tokenization status."""
        info = {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
            "bm25_enabled": self.enable_bm25,
        }
        
        if self.enable_bm25:
            info["bm25_documents_count"] = len(self.bm25_documents)
            info["bm25_index_ready"] = self.bm25_index is not None
            info["tokenization_info"] = self.get_tokenization_info()
            
        return info
