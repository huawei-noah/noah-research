from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "locomo"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
LOCOMO_DIR = Path(__file__).resolve().parents[2] / "src" / "application" / "locomo"
if str(LOCOMO_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_DIR))

import locomo_search_only_eval as runner  # noqa: E402
from locomo_search_only_eval import build_parser  # noqa: E402


def test_search_only_eval_defaults_to_top50() -> None:
    parser = build_parser()

    args = parser.parse_args(["--collection-prefix", "locomo_test", "--output-dir", "/tmp/locomo-test"])

    assert args.top_k == 50
    assert args.search_recall_size == 50
    assert args.search_concurrency == 8
    assert args.score_concurrency == 2


def test_search_only_eval_uses_unified_search_pipeline_with_vanilla_strategy() -> None:
    source = (SCRIPT_DIR / "locomo_search_only_eval.py").read_text(encoding="utf-8")
    legacy_flat_alias = "non" + "_schema"

    assert legacy_flat_alias not in source
    assert "mindmemos.config.non_schema_search" not in source
    assert "mindmemos.pipelines.search.non_schema" not in source
    assert "SearchPipelineImpl" in source
    assert 'search_pipeline="vanilla"' in source


class _RecordingSearchPipeline:
    def __init__(self) -> None:
        self.inputs = []

    async def search(self, inp, context):
        self.inputs.append((inp, context))
        return SimpleNamespace(memories=[])


def test_search_only_question_filters_to_current_conversation_user() -> None:
    pipeline = _RecordingSearchPipeline()

    async def run() -> None:
        await runner._search_question(
            pipeline,
            {"sample_id": "sample-1"},
            conversation_index=4,
            question_index=0,
            qa={"question": "q", "answer": "a", "category": 1},
            collection_prefix="locomo_test",
            top_k=5,
            rerank=False,
        )

    asyncio.run(run())

    assert pipeline.inputs[0][0].filters == {"user_id": "conv_4"}
    assert pipeline.inputs[0][0].search_pipeline == "vanilla"
    assert pipeline.inputs[0][0].rerank is False
