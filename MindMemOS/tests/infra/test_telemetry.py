from types import SimpleNamespace

from mindmemos.infra import telemetry


def test_setup_tracer_provider_skips_unavailable_batch_endpoint(monkeypatch) -> None:
    telemetry._provider = None
    monkeypatch.setattr(telemetry, "_otlp_endpoint_available", lambda endpoint: False)

    config = SimpleNamespace(
        service_name="mindmemos-test",
        span_type="batch",
        telemetry_endpoint="http://localhost:4318",
        telemetry_timeout=5,
        max_queue_size=10,
        max_export_batch_size=5,
        logs_enabled=True,
    )

    assert telemetry.setup_tracer_provider(config) is None
    assert telemetry._provider is None
