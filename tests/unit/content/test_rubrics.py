"""Rubrics as data: the real mock and topic-answer files load, and bad rubrics are refused."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from dogma.content.rubrics import Rubric, check_scores, load_rubric, load_rubric_version
from dogma.core.errors import ContentError, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[3]
RUBRICS = REPO_ROOT / "curriculum" / "rubrics"

PART_MAXIMA = {
    "coding": 24,
    "reframe": 16,
    "ask": 16,
    "structure": 28,
    "synthesize": 16,
    "communication": 20,
}


def _anchors() -> dict[int, str]:
    return {1: "Not shown.", 2: "Attempted, weak.", 3: "Solid.", 4: "Exceptional."}


def _topic_answer() -> dict[str, Any]:
    return {
        "id": "topic-answer",
        "version": 1,
        "kind": "topic-answer",
        "title": "Topic answer",
        "scale": {"low": 1, "high": 4, "anchors": _anchors()},
        "ready_score": 3,
        "dimensions": [
            {
                "id": "accuracy",
                "title": "Accuracy",
                "question": "Is it right?",
                "anchors": _anchors(),
            },
            {
                "id": "structure",
                "title": "Structure",
                "question": "Is it shaped?",
                "anchors": _anchors(),
            },
        ],
    }


def _mock() -> dict[str, Any]:
    return {
        "id": "mock",
        "version": 1,
        "kind": "mock",
        "title": "Mock",
        "scale": {"low": 1, "high": 4, "anchors": _anchors()},
        "parts": [
            {
                "id": "coding",
                "title": "Coding",
                "minutes": "10-15",
                "maximum": 8,
                "items": [
                    {"id": "coding.narration", "text": "Talks while coding."},
                    {"id": "coding.correctness", "text": "Output is right."},
                ],
            }
        ],
        "bands": [{"label": "All", "low": 0, "high": 8, "text": "Everything."}],
    }


def _write(directory: Path, name: str, data: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _load_bad(tmp_path: Path, mutate: Callable[[dict[str, Any]], None], kind: str = "mock") -> None:
    data = _mock() if kind == "mock" else _topic_answer()
    mutate(data)
    path = _write(tmp_path, f"{data['id']}-v{data['version']}.yaml", data)
    with pytest.raises(ContentError) as caught:
        load_rubric(path)
    assert caught.value.code is ErrorCode.CONTENT_SCHEMA_INVALID


# --- the real files ---------------------------------------------------------------------------


def test_the_imported_mock_rubric_loads_with_thirty_items_in_six_parts() -> None:
    rubric = load_rubric(RUBRICS / "mock-v1.yaml")

    assert rubric.kind == "mock"
    assert rubric.version == 1
    assert [part.id for part in rubric.parts] == [
        "coding",
        "reframe",
        "ask",
        "structure",
        "synthesize",
        "communication",
    ]
    assert len(rubric.criteria_ids) == 30
    assert len(set(rubric.criteria_ids)) == 30
    assert rubric.criteria_ids[0] == "coding.narration"
    assert {part.id: part.maximum for part in rubric.parts} == PART_MAXIMA
    assert rubric.maximum == 120


def test_the_mock_bands_cover_zero_to_the_maximum_without_gaps() -> None:
    rubric = load_rubric(RUBRICS / "mock-v1.yaml")

    assert [(band.low, band.high) for band in rubric.bands] == [
        (0, 49),
        (50, 69),
        (70, 89),
        (90, 120),
    ]


def test_the_topic_answer_rubric_has_four_dimensions_and_a_ready_score_of_three() -> None:
    rubric = load_rubric(RUBRICS / "topic-answer-v1.yaml")

    assert rubric.kind == "topic-answer"
    assert rubric.criteria_ids == ("accuracy", "structure", "trade-offs", "communication")
    assert rubric.ready_score == 3
    assert rubric.maximum == 16
    for dimension in rubric.dimensions:
        assert set(dimension.anchors) == {1, 2, 3, 4}


def test_load_rubric_version_finds_the_file_by_id_and_version() -> None:
    rubric = load_rubric_version(REPO_ROOT / "curriculum", "topic-answer", 1)

    assert isinstance(rubric, Rubric)
    assert rubric.id == "topic-answer"


def test_a_missing_rubric_version_is_a_content_error() -> None:
    with pytest.raises(ContentError) as caught:
        load_rubric_version(REPO_ROOT / "curriculum", "topic-answer", 9)

    assert caught.value.code is ErrorCode.CONTENT_NOT_FOUND


# --- refusals ---------------------------------------------------------------------------------


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    _load_bad(tmp_path, lambda data: data.update(colour="blue"))


def test_a_duplicate_item_id_is_refused(tmp_path: Path) -> None:
    def duplicate(data: dict[str, Any]) -> None:
        items = data["parts"][0]["items"]
        items[1]["id"] = items[0]["id"]

    _load_bad(tmp_path, duplicate)


def test_a_part_maximum_that_is_not_four_times_its_items_is_refused(tmp_path: Path) -> None:
    _load_bad(tmp_path, lambda data: data["parts"][0].update(maximum=7))


def test_bands_with_a_gap_are_refused(tmp_path: Path) -> None:
    def gap(data: dict[str, Any]) -> None:
        data["bands"] = [
            {"label": "Low", "low": 0, "high": 3, "text": "x"},
            {"label": "High", "low": 5, "high": 8, "text": "y"},
        ]

    _load_bad(tmp_path, gap)


def test_a_scale_without_all_four_anchors_is_refused(tmp_path: Path) -> None:
    def three_anchors(data: dict[str, Any]) -> None:
        del data["scale"]["anchors"][4]

    _load_bad(tmp_path, three_anchors)


def test_a_dimension_with_an_empty_anchor_is_refused(tmp_path: Path) -> None:
    def empty(data: dict[str, Any]) -> None:
        data["dimensions"][0]["anchors"][2] = ""

    _load_bad(tmp_path, empty, kind="topic-answer")


def test_a_topic_answer_without_a_ready_score_is_refused(tmp_path: Path) -> None:
    _load_bad(tmp_path, lambda data: data.pop("ready_score"), kind="topic-answer")


def test_a_file_name_that_does_not_match_its_id_and_version_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "wrong-name.yaml", _mock())

    with pytest.raises(ContentError) as caught:
        load_rubric(path)

    assert caught.value.code is ErrorCode.CONTENT_SCHEMA_INVALID


# --- scores against a rubric ------------------------------------------------------------------


def test_complete_scores_in_range_are_accepted() -> None:
    rubric = load_rubric(RUBRICS / "topic-answer-v1.yaml")

    problems = check_scores(
        rubric, {"accuracy": 3, "structure": 4, "trade-offs": 2, "communication": 3}
    )

    assert problems == []


def test_check_scores_names_missing_unknown_and_out_of_range_keys() -> None:
    rubric = load_rubric(RUBRICS / "topic-answer-v1.yaml")

    problems = check_scores(rubric, {"accuracy": 5, "structure": True, "made-up": 2})

    text = " ".join(problems)
    assert "accuracy" in text and "between 1 and 4" in text
    assert "structure" in text and "whole number" in text
    assert "made-up" in text
    assert "trade-offs" in text and "communication" in text
