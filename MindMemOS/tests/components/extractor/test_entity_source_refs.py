"""Tests for entity provenance source-ref resolution in add_builder."""

from mindmemos.components.extractor.vanilla.add_builder import _entity_evidence_source_refs
from mindmemos.typing.memory import Entity, SourceRef


def _src(source_id: str) -> SourceRef:
    return SourceRef(source_id=source_id, source_type="message", is_parsed=True)


def _entity(source_refs=None) -> Entity:
    metadata = {"source_refs": source_refs} if source_refs is not None else {}
    return Entity(name="OpenAI", canonical_name="OpenAI", entity_type="organization", metadata=metadata)


def test_resolves_each_evidence_ref_to_source() -> None:
    sources = {0: _src("s0"), 1: _src("s1")}
    out = _entity_evidence_source_refs(_entity(["s0", "s1"]), sources, _src("fallback"))
    assert [s.source_id for s in out] == ["s0", "s1"]


def test_skips_unknown_refs_and_falls_back() -> None:
    out = _entity_evidence_source_refs(_entity(["s9"]), {0: _src("s0")}, _src("fallback"))
    assert [s.source_id for s in out] == ["fallback"]


def test_falls_back_when_entity_has_no_source_refs() -> None:
    out = _entity_evidence_source_refs(_entity(None), {0: _src("s0")}, _src("fallback"))
    assert [s.source_id for s in out] == ["fallback"]


def test_dedupes_repeated_refs() -> None:
    out = _entity_evidence_source_refs(_entity(["s0", "s0"]), {0: _src("s0")}, _src("fallback"))
    assert [s.source_id for s in out] == ["s0"]
