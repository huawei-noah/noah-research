# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, List, Optional, Tuple, Union

from openai import OpenAI
from pydantic import Field, PrivateAttr

from ._base import EmbedClientBase, RerankClientBase


class OpenAIEmbedClient(EmbedClientBase):
    """OpenAI client based embedding client"""
    base_url: Optional[str] = Field(default_factory=str, description="the request post url")
    api_key: Optional[str] = Field(default_factory=str, description="the request api key")
    model: str = Field(description="embedding model name")
    dimensions: Optional[int] = Field(default=None, description="optional dimension reduction")
    max_retries: Optional[int] = Field(default=2, description="post request service time")
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None,
        description="post waiting time")
    _client: Any = PrivateAttr(init=False)

    def model_post_init(self, context: Any, /) -> None:
        """Init OpenAI Client"""
        self._client = OpenAI(
            base_url=self.base_url or None,
            api_key=self.api_key or "ollama",
            max_retries=self.max_retries,
            timeout=self.request_timeout,
        )

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """batch embed texts, return matrix of embeddings"""
        kwargs.setdefault("model", self.model)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(input=texts, **kwargs)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """embed single texts"""
        return self.embed_documents([text], **kwargs)[0]


class SentenceTransformerEmbed(EmbedClientBase):
    """Local rerank model based on sentence transformer"""
    model: str = Field(description="Rerank model name")
    device: str = Field(default="cpu", description="local embedding model device")
    _embedding_model: Any = PrivateAttr(init=False)

    def model_post_init(self, context: Any, /) -> None:
        """create local embedding_model function"""
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
        except ImportError:
            from langchain.embeddings import SentenceTransformerEmbeddings
        self._embedding_model = SentenceTransformerEmbeddings(
            model_name=self.model,
            model_kwargs={"device": self.device}
        )

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """batch embed texts based on local model"""
        return self._embedding_model.embed_documents(texts)

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """embed single text based on local model"""
        return self.embed_documents(texts=[text], **kwargs)[0]


class FlagRerankModel(RerankClientBase):
    """FlagRerank based rerank model local implementation"""
    model: str = Field(description="rerank model name")
    top_n: int = Field(default=1, description="returned rerank num")
    device: str = Field(default="cpu", description="rerank deployment device")
    _rerank_model: Any = PrivateAttr(init=False)

    def model_post_init(self, context: Any, /) -> None:
        """create local rerank model"""
        from FlagEmbedding import FlagReranker
        self._rerank_model = FlagReranker(self.model,
            top_n=self.top_n,
            use_fp16=False,
            devices=[self.device])

    async def rank(self, query: str, texts: List[str], **kwargs) -> List[int]:
        """FlagReranker rerank texts"""
        pairs = [(query, text) for text in texts]
        scores = self._rerank_model.compute_score(pairs)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        indexes = [idx for idx, _ in indexed_scores]
        return indexes[:self.top_n]
