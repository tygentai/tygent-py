from __future__ import annotations

import pytest
from typing import Dict

from tyapi.service.transform import PlanTransformer, extract_links_from_text, merge_links


@pytest.fixture
def sample_spec() -> Dict[str, Dict[str, object]]:
    return {
        "gather": {
            "kind": "llm",
            "prompt": "Collect data from https://example.com",
            "deps": [],
            "links": [],
            "metadata": {"redundancy_hint": True},
        },
        "analyze": {
            "kind": "llm",
            "prompt": "Analyze findings",
            "deps": ["gather"],
            "links": ["https://analysis"]
        },
        "validate": {
            "kind": "llm",
            "prompt": "Validate insights",
            "deps": ["analyze"],
        },
        "executive_summary": {
            "kind": "llm",
            "prompt": "Summarize",
            "deps": ["validate"],
        },
    }


def test_extract_links(sample_spec: Dict[str, Dict[str, object]]) -> None:
    links = extract_links_from_text(sample_spec["gather"]["prompt"])
    assert links == ["https://example.com"]


def test_merge_links_dedup() -> None:
    merged = merge_links(["https://a"], ["https://a", "http://b"])
    assert merged == ["https://a", "http://b"]


def test_transformer_builds_metadata(sample_spec: Dict[str, Dict[str, object]]) -> None:
    transformer = PlanTransformer()
    plan = transformer.transform(sample_spec)
    assert plan["name"] == "tygent_ingested_plan"
    assert len(plan["steps"]) >= len(sample_spec)
    step_names = {step["name"] for step in plan["steps"]}
    assert "gather" in step_names
    assert plan["meta"]["critical_path"][-1] == "executive_summary"
    assert plan["prefetch"]["links"]
    assert plan["prefetch"]["by_step"]["analyze"] == ["https://analysis"]


def test_transformer_redundancy_steps(sample_spec: Dict[str, Dict[str, object]]) -> None:
    transformer = PlanTransformer(redundancy_mode="steps")
    plan = transformer.transform(sample_spec)
    redundancy_steps = [s for s in plan["steps"] if s["name"].endswith("_redundancy")]
    assert redundancy_steps, "Expected redundancy steps when redundancy_mode='steps'"
