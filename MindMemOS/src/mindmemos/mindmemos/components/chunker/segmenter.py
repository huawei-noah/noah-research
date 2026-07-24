"""Message segmentation for memory add and later ingestion pipelines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...typing import AddPipelineInput, DialogueMessage, FileMessage, SourceRef, TextMessage, UrlMessage


class SourceAwareSegment(BaseModel):
    """A text segment with enough source metadata to build graph edges."""

    segment_id: str
    text: str
    source_ref: SourceRef
    message_index: int
    role: str | None = None
    timestamp: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageSegmenter:
    """Split add input messages into source-aware text segments."""

    def segment(self, inp: AddPipelineInput) -> tuple[list[SourceAwareSegment], list[SourceRef]]:
        segments: list[SourceAwareSegment] = []
        source_refs: list[SourceRef] = []

        for index, message in enumerate(inp.messages):
            source_ref = self._source_ref_for_message(message, index)
            text = self._message_text(message)
            if isinstance(message, (FileMessage, UrlMessage)):
                source_refs.append(source_ref)
                continue
            if not text.strip():
                continue
            source_refs.append(source_ref)
            segments.append(
                SourceAwareSegment(
                    segment_id=f"message-{index}",
                    text=text,
                    source_ref=source_ref,
                    message_index=index,
                    role=getattr(message, "role", None),
                    timestamp=getattr(message, "timestamp", None),
                    end_offset=len(text),
                    metadata={"message_type": type(message).__name__},
                )
            )

        return segments, source_refs

    def _source_ref_for_message(
        self,
        message: DialogueMessage | UrlMessage | FileMessage | TextMessage,
        index: int,
    ) -> SourceRef:
        if isinstance(message, FileMessage):
            return SourceRef(
                source_type="file",
                file_path=message.file_path,
                file_name=message.file_name,
                mime_type=message.file_type or None,
                is_parsed=False,
                metadata={"message_index": index},
            )
        if isinstance(message, UrlMessage):
            return SourceRef(
                source_type="url",
                uri=message.url,
                title=message.url,
                is_parsed=False,
                metadata={"message_index": index},
            )
        return SourceRef(
            source_type="message",
            message_id=f"message-{index}",
            is_parsed=True,
            metadata={"message_index": index},
        )

    def _message_text(self, message: DialogueMessage | UrlMessage | FileMessage | TextMessage) -> str:
        if isinstance(message, TextMessage):
            return message.text
        if isinstance(message, DialogueMessage):
            return message.content
        return ""
