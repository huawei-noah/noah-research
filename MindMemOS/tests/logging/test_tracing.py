from inspect import signature

from mindmemos.logging.tracing import _attr_value, traced


def test_traced_defaults_to_metadata_only() -> None:
    params = signature(traced).parameters

    assert params["record_args"].default is False
    assert params["record_result"].default is False


def test_attr_value_truncates_large_strings() -> None:
    value = _attr_value("x" * 9000)

    assert isinstance(value, str)
    assert len(value) < 9000
    assert "<truncated chars=" in value
