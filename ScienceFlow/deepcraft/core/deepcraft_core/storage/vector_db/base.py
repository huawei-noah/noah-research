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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseVectorStorage(ABC):
    r"""An abstract base class for vector storage systems. Provides a
    consistent interface for storing, querying, and managing vector embeddings
    with metadata.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def count(self) -> int:
        r"""Returns the number of documents in the vector storage.

        Returns:
            int: The total number of documents stored.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        r"""Removes all documents from the vector storage."""
        pass
