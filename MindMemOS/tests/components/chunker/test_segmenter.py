from mindmemos.components.chunker import MessageSegmenter
from mindmemos.typing.service import AddPipelineInput


def test_message_segmenter_builds_source_aware_segments_for_text_and_dialogue() -> None:
    segments, sources = MessageSegmenter().segment(
        AddPipelineInput(
            messages=[
                {"text": "Remember Qdrant."},
                {"role": "user", "content": "I prefer FastAPI.", "timestamp": 1770000000000},
            ]
        )
    )

    assert [segment.text for segment in segments] == ["Remember Qdrant.", "I prefer FastAPI."]
    assert segments[0].source_ref.source_type == "message"
    assert segments[1].role == "user"
    assert segments[1].timestamp == 1770000000000
    assert [source.metadata["message_index"] for source in sources] == [0, 1]


def test_message_segmenter_keeps_file_and_url_as_pending_sources() -> None:
    segments, sources = MessageSegmenter().segment(
        AddPipelineInput(
            messages=[
                {"file_name": "notes.pdf", "file_path": "oss://bucket/notes.pdf"},
                {"url": "https://example.com/design"},
            ]
        )
    )

    assert segments == []
    assert [source.source_type for source in sources] == ["file", "url"]
    assert sources[0].file_path == "oss://bucket/notes.pdf"
    assert sources[1].uri == "https://example.com/design"
    assert all(source.is_parsed is False for source in sources)
