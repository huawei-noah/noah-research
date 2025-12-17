# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import chromadb

    # The Collection type is chromadb.Collection at runtime
    Collection = chromadb.Collection
except ImportError as e:
    raise ImportError(
        "chromadb is not installed. Please install it with: `pip install chromadb`") from e

from ._base_db import VectorDB, SearchResult
from ..typing import DBItem
from ...logger import get_logger

logger = get_logger()


class ChromaDB(VectorDB):
    """Vector database implementation based on native chromadb"""
    _client: chromadb.Client = None
    _collection: Optional[chromadb.Collection] = None

    def model_post_init(self, context: Any, /) -> None:
        """Initialize ChromaDB after pydantic validation"""
        # Initialize client and collection
        self._init_client()
        self._get_or_create_collection()

    def _init_client(self):
        """Initialize the ChromaDB client"""
        if self._client is None:
            # Use default ChromaDB client (new API)
            if self.persist_directory and self.persist_directory.strip():
                # Non-empty persist_directory: use persistent client
                Path(self.persist_directory).mkdir(exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                # Empty persist_directory: use in-memory client
                self._client = chromadb.Client()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create a collection"""
        if self._collection is not None:
            return self._collection

        # Check if the collection already exists
        existing_collections = self._client.list_collections()
        collection_exists = any(coll.name == self.collection_name for coll in existing_collections)

        if collection_exists:
            # Get existing collection
            self._collection = self._client.get_collection(self.collection_name)
        else:
            # Create the collection
            self._create_collection()

        return self._collection

    def _create_collection(self):
        """Create a new collection"""
        # Create collection using the embedding wrapper
        embedding_adapter = self._create_embedding_wrapper()
        if embedding_adapter is not None:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_adapter
            )
        else:
            # Fallback to default collection creation
            self._collection = self._client.create_collection(name=self.collection_name)

    def _create_embedding_wrapper(self):
        """Create a simple adapter to adapt custom embedding to ChromaDB's expected interface"""
        if self.embedding is None:
            return None

        class ChromaDBEmbeddingAdapter:
            """Simple adapter for ChromaDB embedding interface"""

            def __init__(self, embed_client):
                self.embed_client = embed_client

            def __call__(self, input):
                """ChromaDB expected interface: __call__(input) -> List[List[float]]"""
                try:
                    if isinstance(input, str):
                        # Single text input
                        embedding_vector = self.embed_client.embed_query(input)
                        return [embedding_vector]
                    elif isinstance(input, list):
                        # List input handling
                        if len(input) == 1:
                            # Single element in list: extract and process
                            text_input = input[0]
                            if isinstance(text_input, str):
                                embedding_vector = self.embed_client.embed_query(text_input)
                                return [embedding_vector]
                            else:
                                embedding_vector = self.embed_client.embed_query(str(text_input))
                                return [embedding_vector]
                        else:
                            # Multiple elements: batch process
                            texts = []
                            for item in input:
                                if isinstance(item, str):
                                    texts.append(item)
                                else:
                                    texts.append(str(item))
                            return self.embed_client.embed_documents(texts)
                    else:
                        # Other types: convert to string
                        return [self.embed_client.embed_query(str(input))]
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

            def embed_query(self, input):
                """Compatibility method for ChromaDB internal calls"""
                if isinstance(input, list) and len(input) == 1:
                    input = input[0]  # Extract single text from list
                if not isinstance(input, str):
                    input = str(input)
                return [self.embed_client.embed_query(input)]

            def embed_documents(self, input):
                """Compatibility method for ChromaDB internal calls"""
                if not isinstance(input, list):
                    input = [str(input)]
                else:
                    # Process list to ensure all items are strings
                    processed_texts = []
                    for item in input:
                        if isinstance(item, str):
                            processed_texts.append(item)
                        else:
                            processed_texts.append(str(item))
                    input = processed_texts

                return self.embed_client.embed_documents(input)

        return ChromaDBEmbeddingAdapter(self.embedding)

    # Async methods from DBBase interface
    async def persist(self):
        """Persist data. In ChromaDB, data is automatically persisted."""
        pass

    def _clear_collection(self) -> int:
        """Private helper to clear all documents in the collection."""
        if not self._collection:
            return 0

        # Only fetch IDs to reduce memory usage
        all_data = self._collection.get(include=[])
        doc_ids = all_data.get('ids', [])
        if not doc_ids:
            return 0

        self._collection.delete(ids=doc_ids)
        return len(doc_ids)

    async def clear_db(self) -> int:
        """Clear all documents from the vector store."""
        try:
            return self._clear_collection()
        except Exception as e:
            raise RuntimeError(f"Failed to clear db: {str(e)}") from e

    async def similarity_search(
            self,
            query: str,
            k: int = None,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """The retrieval entrance"""
        top_k = k if k is not None else self.top_k

        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            query_kwargs["where"] = filter

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"[ChromaDB] similarity_search failed: {e}")
            return []

        search_results: List[SearchResult] = []
        if not results.get("documents"):
            return search_results

        for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0] or [],
                results["distances"][0] or [],
        ):
            if meta is None:
                meta = {}
            meta["_distance"] = dist
            item = DBItem(
                text=doc,
                ids=meta.get("ids"),
                metadata=meta,
            )
            search_results.append(SearchResult.from_db_item(
                item,
                score=1 - dist
            ))

        return search_results

    async def add_texts(
            self,
            items: Union[Sequence[DBItem], Sequence[str]],
            *,
            metadatas: Optional[Sequence[dict]] = None,
            ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Add new db item to vectorstore"""
        # Ensure collection is properly initialized

        if items and isinstance(items[0], str):
            texts: List[str] = list(items)
            target_len = len(texts)

            def _broadcast(src: Optional[Sequence], factory):
                if not src:
                    return [factory() for _ in range(target_len)]
                src = list(src)
                return src + [src[-1]] * (target_len - len(src))

            metadatas = _broadcast(metadatas, lambda: {"default": "metadata"})
            ids = _broadcast(ids, lambda: str(uuid.uuid4()))

            items = [
                DBItem(text=t, metadata=m, ids=i)
                for t, m, i in zip(texts, metadatas, ids)
            ]
        else:
            items: List[DBItem] = list(items)

        try:
            self._get_or_create_collection()
        except Exception as e:
            return []

        # Extract data from DBItem objects
        texts = []
        metadatas = []
        ids = []

        # Reset items for iteration
        items_list = list(items)
        for item in items_list:
            texts.append(item.text)
            metadatas.append(item.metadata if item.metadata and item.metadata != {} else {"default": "metadata"})
            if item.ids:
                ids.append(str(item.ids))
            else:
                ids.append(str(uuid.uuid4()))

        if not ids:
            return []

        for metadata, id in zip(metadatas, ids):
            metadata["ids"] = id

        try:
            self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return ids
        except Exception as e:
            # Try to refresh collection reference and retry
            try:
                self._collection = None
                self._get_or_create_collection()
                self._collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                return ids
            except Exception:
                return []

    def get_vector_count(self) -> int:
        """Get the number of vectors stored"""
        try:
            return self._collection.count()
        except Exception as e:
            print(f"Error getting vector count: {e}")
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            info = {
                'name': self.collection_name,
                'count': self.get_vector_count(),
                'persist_directory': self.persist_directory,
                'has_custom_embedding': self.embedding is not None
            }
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

    async def delete_by_ids(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        if not ids:
            return True

        try:
            self._collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting documents by IDs: {e}")
            return False
