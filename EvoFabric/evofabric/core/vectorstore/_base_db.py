# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import Field

from ..clients import EmbedClientBase
from ..factory import BaseComponent, FactoryTypeAdapter
from ..typing import DBItem, SearchResult


class DBBase(BaseComponent, ABC):
    """A basic interface define of vector db"""
    collection_name: str = Field(description="database name")
    persist_directory: str = Field(description="database save path")
    embedding: Annotated[EmbedClientBase, FactoryTypeAdapter, Field(description="the callable embedding function")]
    top_k: int = Field(description="returned search item number")

    @abstractmethod
    async def persist(self):
        """save the vectorstore"""

    @abstractmethod
    async def clear_db(self):
        """clear the vector store"""

    @abstractmethod
    async def similarity_search(
            self,
            query: str,
            k: int = None,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """the retrieval entrance"""

    @abstractmethod
    async def add_texts(
            self,
            items: Union[Sequence[DBItem], Sequence[str]],
            *,
            metadatas: Optional[Sequence[dict]] = None,
            ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """add new db item to vectorstore"""

    @abstractmethod
    async def delete_by_ids(self, ids: list[str]) -> None:
        """
        delete item by it original ids
        :param ids:
        :return:
        """


class VectorDB(DBBase, ABC):
    """Vector database abstract class, simplified design"""
    persist_directory: str = Field(default_factory=str, description="database save path")
    top_k: int = Field(default=5, description="returned search item number")

    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get vector count

        Returns:
            Number of vectors
        """

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information

        Returns:
            Collection information dictionary
        """
