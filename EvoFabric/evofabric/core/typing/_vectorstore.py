# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DBItem(BaseModel):
    """RAG database basic item"""
    text: str  # Text need to be vectorized

    ids: Optional[str] = None  # id in database

    metadata: Optional[dict] = None


class SearchResult(BaseModel):
    """Search result data structure"""
    text: str

    metadata: Optional[Dict[str, Any]] = None

    score: Optional[float] = None

    id: str = Field(default_factory=str)

    @classmethod
    def from_db_item(cls, item: DBItem, *, score: Optional[float] = None) -> 'SearchResult':
        return cls(
            text=item.text,
            metadata=item.metadata,
            score=score,
            id=item.ids,
        )

    def to_db_item(self) -> DBItem:
        return DBItem(
            text=self.text,
            metadata=self.metadata,
            ids=self.id,
        )