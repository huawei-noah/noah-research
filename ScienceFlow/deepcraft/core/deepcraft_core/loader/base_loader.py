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

import json
import re
from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Optional


def encrypt_data_id(value):
    sha512_hash = hashlib.sha512()
    sha512_hash.update(value)
    return sha512_hash.hexdigest()

class BaseLoader(ABC):
    r"""Abstract base class for document loaders.
    
    This class defines the interface that all document loaders must implement
    to provide consistent document loading functionality across different formats.
    """

    @abstractmethod
    def load(self, source: Any) -> List[Dict[str, Any]]:
        r"""Load documents from a source.

        Args:
            source (Any): The source to load documents from. This could be
                a file path, URL, database connection, etc.

        Returns:
            List[Dict[str, Any]]: A list of loaded documents with their content
                and metadata.
        """
        pass

    @abstractmethod
    def load_batch(self, sources: List[Any]) -> List[Dict[str, Any]]:
        r"""Load documents from multiple sources.

        Args:
            sources (List[Any]): A list of sources to load documents from.

        Returns:
            List[Dict[str, Any]]: A list of loaded documents from all sources.
        """
        pass


def strip_consecutive_newlines(text: str) -> str:
    r"""Strips consecutive newlines from a string.

    Args:
        text (str): The string to strip.

    Returns:
        str: The string with consecutive newlines stripped.
    """
    return re.sub(r"\s*\n\s*", "\n", text)


def create_file(file: BytesIO, filename: str) -> "File":
    r"""Reads an uploaded file and returns a File object.

    Args:
        file (BytesIO): A BytesIO object representing the contents of the file.
        filename (str): The name of the file.

    Returns:
        File: A File object.

    Raises:
        NotImplementedError: If the file type is not supported.
    """
    ext_to_cls = {
        "docx": DocxFile,
        "pdf": PdfFile,
        "txt": TxtFile,
        "json": JsonFile,
        "html": HtmlFile,
    }

    ext = filename.split(".")[-1].lower()
    if ext not in ext_to_cls:
        raise NotImplementedError(f"File type {ext} not supported")

    out_file = ext_to_cls[ext].from_bytes(file, filename)
    return out_file


def create_file_from_raw_bytes(raw_bytes: bytes, filename: str) -> "File":
    r"""Reads raw bytes and returns a File object.

    Args:
        raw_bytes (bytes): The raw bytes content of the file.
        filename (str): The name of the file.

    Returns:
        File: A File object.
    """
    file = BytesIO(raw_bytes)
    return create_file(file, filename)


class File(ABC):
    r"""Abstract base class representing an uploaded file comprised of Documents.

    Args:
        name (str): The name of the file.
        file_id (str): The unique identifier of the file.
        metadata (Optional[Dict[str, Any]]): Additional metadata
            associated with the file. (default: None)
        docs (Optional[List[Dict[str, Any]]]): A list of documents
            contained within the file. (default: None)
        raw_bytes (bytes): The raw bytes content of the file.
            (default: b"")
    """

    def __init__(
        self,
        name: str,
        file_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        docs: Optional[List[Dict[str, Any]]] = None,
        raw_bytes: bytes = b"",
    ) -> None:
        self.name = name
        self.file_id = file_id
        self.metadata = metadata or {}
        self.docs = docs or []
        self.raw_bytes = raw_bytes

    @classmethod
    @abstractmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "File":
        r"""Creates a File object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the file.
            filename (str): The name of the file.

        Returns:
            File: A File object.
        """
        pass

    @classmethod
    def from_raw_bytes(cls, raw_bytes: bytes, filename: str) -> "File":
        r"""Creates a File object from raw bytes.

        Args:
            raw_bytes (bytes): The raw bytes content of the file.
            filename (str): The name of the file.

        Returns:
            File: A File object.
        """
        file = BytesIO(raw_bytes)
        return cls.from_bytes(file, filename)

    def __repr__(self) -> str:
        return (
            f"File(name={self.name}, id={self.file_id}, "
            f"metadata={self.metadata}, docs={self.docs})"
        )

    def __str__(self) -> str:
        return (
            f"File(name={self.name}, id={self.file_id}, metadata="
            f"{self.metadata})"
        )

    def copy(self) -> "File":
        r"""Create a deep copy of this File.

        Returns:
            File: A deep copy of the current File instance.
        """
        return self.__class__(
            name=self.name,
            file_id=self.file_id,
            metadata=deepcopy(self.metadata),
            docs=deepcopy(self.docs),
            raw_bytes=self.raw_bytes,
        )


class DocxFile(File):
    r"""Concrete implementation of File for DOCX documents."""

    @classmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "DocxFile":
        r"""Creates a DocxFile object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the
                docx file.
            filename (str): The name of the file.

        Returns:
            DocxFile: A DocxFile object.

        Raises:
            ImportError: If docx2txt is not installed.
        """
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is not installed. Please install it with: pip install docx2txt"
            )

        text = docx2txt.process(file)
        text = strip_consecutive_newlines(text)
        doc = {"page_content": text.strip()}
        file_id = encrypt_data_id(file.getvalue())
        file.seek(0)
        return cls(
            name=filename,
            file_id=file_id,
            docs=[doc],
            raw_bytes=file.getvalue(),
        )


class PdfFile(File):
    r"""Concrete implementation of File for PDF documents."""

    @classmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "PdfFile":
        r"""Creates a PdfFile object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the
                pdf file.
            filename (str): The name of the file.

        Returns:
            PdfFile: A PdfFile object.

        Raises:
            ImportError: If PyMuPDF is not installed.
        """
        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "PyMuPDF is not installed. Please install it with: pip install PyMuPDF"
            )

        pdf = pymupdf.open(stream=file.read(), filetype="pdf")
        docs = []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            text = strip_consecutive_newlines(text)
            doc = {"page_content": text.strip(), "page": i + 1}
            docs.append(doc)
        pdf.close()
        file_id = md5(file.getvalue()).hexdigest()
        file.seek(0)
        return cls(
            name=filename,
            file_id=file_id,
            docs=docs,
            raw_bytes=file.getvalue(),
        )


class TxtFile(File):
    r"""Concrete implementation of File for text documents."""

    @classmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "TxtFile":
        r"""Creates a TxtFile object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the
                txt file.
            filename (str): The name of the file.

        Returns:
            TxtFile: A TxtFile object.
        """
        text = file.read().decode("utf-8")
        text = strip_consecutive_newlines(text)
        doc = {"page_content": text.strip()}
        file_id = encrypt_data_id(file.getvalue())
        file.seek(0)
        return cls(
            name=filename,
            file_id=file_id,
            docs=[doc],
            raw_bytes=file.getvalue(),
        )


class JsonFile(File):
    r"""Concrete implementation of File for JSON documents."""

    @classmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "JsonFile":
        r"""Creates a JsonFile object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the
                json file.
            filename (str): The name of the file.

        Returns:
            JsonFile: A JsonFile object.
        """
        data = json.load(file)
        doc = {"page_content": json.dumps(data, ensure_ascii=False)}
        file_id = encrypt_data_id(file.getvalue())
        file.seek(0)
        return cls(
            name=filename,
            file_id=file_id,
            docs=[doc],
            raw_bytes=file.getvalue(),
        )


class HtmlFile(File):
    r"""Concrete implementation of File for HTML documents."""

    @classmethod
    def from_bytes(cls, file: BytesIO, filename: str) -> "HtmlFile":
        r"""Creates a HtmlFile object from a BytesIO object.

        Args:
            file (BytesIO): A BytesIO object representing the contents of the
                html file.
            filename (str): The name of the file.

        Returns:
            HtmlFile: A HtmlFile object.

        Raises:
            ImportError: If beautifulsoup4 is not installed.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is not installed. Please install it with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
        text = strip_consecutive_newlines(text)
        doc = {"page_content": text.strip()}
        file_id = encrypt_data_id(file.getvalue())
        file.seek(0)
        return cls(
            name=filename,
            file_id=file_id,
            docs=[doc],
            raw_bytes=file.getvalue(),
        )
