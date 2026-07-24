from datetime import UTC, datetime
from types import SimpleNamespace

import mindmemos.components.extractor._records as records


def _record(payload):
    return SimpleNamespace(add_record_id="add-1", payload=payload)


def test_records_datetime_prefers_payload_event_time_over_message_timestamp() -> None:
    record = _record(
        {
            "event_timestamp_ms": 1700000000000,
            "timestamp": 1700000000000,
            "messages": [{"role": "user", "content": "hello", "timestamp": 1770000000000}],
        }
    )

    assert records.records_datetime([record]) == datetime.fromtimestamp(1700000000000 / 1000, tz=UTC)
    assert records.to_conversation_text([record]).startswith("0. 2023-11-14 22:13:20 user: hello")


def test_records_datetime_uses_message_timestamp_when_payload_event_time_missing() -> None:
    record = _record({"messages": [{"role": "user", "content": "hello", "timestamp": 1770000000000}]})

    assert records.records_datetime([record]) == datetime.fromtimestamp(1770000000000 / 1000, tz=UTC)
    assert records.to_conversation_text([record]).startswith("0. 2026-02-02 02:40:00 user: hello")


def test_records_datetime_uses_legacy_timestamp_when_event_field_missing() -> None:
    record = _record({"timestamp": 1700000000000, "messages": [{"text": "hello"}]})

    assert records.records_datetime([record]) == datetime.fromtimestamp(1700000000000 / 1000, tz=UTC)


def test_records_added_datetime_uses_server_add_time() -> None:
    added_at = datetime(2026, 6, 1, 1, 2, 3, tzinfo=UTC)
    record = _record(
        {
            "event_timestamp_ms": 1700000000000,
            "added_at": added_at,
            "buffered_at": datetime(2026, 6, 1, 1, 2, 4, tzinfo=UTC),
            "messages": [{"text": "hello"}],
        }
    )

    assert records.records_datetime([record]) == datetime.fromtimestamp(1700000000000 / 1000, tz=UTC)
    assert records.records_added_datetime([record]) == added_at


def test_records_datetime_falls_back_to_added_time_when_event_time_is_missing() -> None:
    added_at = datetime(2026, 6, 1, 1, 2, 3, tzinfo=UTC)
    record = _record({"added_at": added_at, "messages": [{"text": "hello"}]})

    assert records.records_datetime([record]) == added_at
