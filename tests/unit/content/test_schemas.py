from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dogma.content.schemas import (
    CardsFile,
    ClozeCard,
    CorePath,
    ProblemMeta,
    Reveal,
    RevealsFile,
    Syllabus,
    TopicMeta,
)


def _syllabus_data(topics: list[dict[str, Any]] | None = None, count: int = 1) -> dict[str, Any]:
    if topics is None:
        topics = [{"id": "tst/one", "title": "One", "tier": "A"}]
    return {
        "version": 1,
        "source": "test",
        "tracks": [
            {
                "code": "tst",
                "order": 1,
                "title": "T",
                "origin": "deepml",
                "topic_count": count,
                "topics": topics,
            }
        ],
    }


def _topic_meta_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "tst/alpha",
        "title": "Alpha",
        "tier": "A",
        "requires": [],
        "aliases": [],
        "objectives": ["one", "two"],
        "minutes": {"lesson": 1, "practice": 1, "min_viable": 1},
        "interview_angle": "x",
        "sources": ["book: A, B"],
        "status": "draft",
    }
    data.update(overrides)
    return data


def _problem_meta_data(language: str, entry: str) -> dict[str, Any]:
    return {
        "id": "tst/alpha/p01-x",
        "version": 1,
        "title": "X",
        "language": language,
        "runtime": "py-numpy",
        "entry": entry,
        "difficulty": "easy",
        "minutes": 10,
        "core": True,
        "tolerance": {"atol": 1e-9, "rtol": 1e-9},
        "banned_imports": [],
        "timeout_s": 30,
        "status": "draft",
    }


def test_real_syllabus_loads_545_topics_in_15_tracks(real_syllabus: Syllabus) -> None:
    assert len(real_syllabus.topics_by_id()) == 545
    assert len(real_syllabus.tracks) == 15


def test_sys_track_is_local_and_tier_b(real_syllabus: Syllabus) -> None:
    sys_track = next(track for track in real_syllabus.tracks if track.code == "sys")
    assert (sys_track.origin, sys_track.order, sys_track.topic_count) == ("local", 15, 7)
    assert {topic.tier for topic in sys_track.topics} == {"B"}


def test_syllabus_accepts_valid_tracks() -> None:
    syllabus = Syllabus.model_validate(_syllabus_data())
    assert set(syllabus.topics_by_id()) == {"tst/one"}


def test_syllabus_rejects_topic_count_that_does_not_match_topics() -> None:
    with pytest.raises(ValidationError, match="topic_count 2 != 1"):
        Syllabus.model_validate(_syllabus_data(count=2))


def test_syllabus_rejects_topic_id_without_its_track_prefix() -> None:
    topics = [{"id": "other/one", "title": "One", "tier": "A"}]
    with pytest.raises(ValidationError, match="must start with 'tst/'"):
        Syllabus.model_validate(_syllabus_data(topics))


def test_syllabus_rejects_duplicate_topic_ids() -> None:
    topics = [
        {"id": "tst/one", "title": "One", "tier": "A"},
        {"id": "tst/one", "title": "Again", "tier": "B"},
    ]
    with pytest.raises(ValidationError, match="duplicate topic ids"):
        Syllabus.model_validate(_syllabus_data(topics, count=2))


def test_syllabus_rejects_unknown_tier() -> None:
    topics = [{"id": "tst/one", "title": "One", "tier": "D"}]
    with pytest.raises(ValidationError, match="tier"):
        Syllabus.model_validate(_syllabus_data(topics))


def test_syllabus_topic_kind_defaults_to_topic_and_accepts_case() -> None:
    topics = [
        {"id": "tst/one", "title": "One", "tier": "A"},
        {"id": "tst/two", "title": "Two", "tier": "B", "kind": "case"},
    ]
    syllabus = Syllabus.model_validate(_syllabus_data(topics, count=2))
    kinds = {topic.id: topic.kind for topic in syllabus.topics_by_id().values()}
    assert kinds == {"tst/one": "topic", "tst/two": "case"}


def test_syllabus_topic_rejects_unknown_kind() -> None:
    topics = [{"id": "tst/one", "title": "One", "tier": "A", "kind": "lab"}]
    with pytest.raises(ValidationError, match="kind"):
        Syllabus.model_validate(_syllabus_data(topics))


def test_real_syllabus_marks_only_the_sys_cases_as_case_kind(real_syllabus: Syllabus) -> None:
    kinds = {
        topic.id: topic.kind
        for track in real_syllabus.tracks
        for topic in track.topics
        if topic.kind == "case"
    }
    assert set(kinds) == {
        "sys/rass-case-framework",
        "sys/churn-prediction-case",
        "sys/fraud-detection-case",
        "sys/predictive-maintenance-case",
        "sys/recommendation-system-case",
        "sys/logistics-optimization-case",
        "sys/rag-system-design-case",
    }


def test_topic_meta_kind_defaults_to_topic_and_rejects_unknown_kinds() -> None:
    assert TopicMeta.model_validate(_topic_meta_data()).kind == "topic"
    with pytest.raises(ValidationError, match="kind"):
        TopicMeta.model_validate(_topic_meta_data(kind="lab"))


def test_reveals_file_is_a_list_of_reveals() -> None:
    reveals = RevealsFile.model_validate(
        [
            {"when_asked": "How many?", "answer": "Thirty."},
            {"when_asked": "Which plan?", "answer": "Basic.", "exhibit": "exhibits/plans.csv"},
        ]
    )
    assert [reveal.exhibit for reveal in reveals.root] == [None, "exhibits/plans.csv"]
    assert isinstance(reveals.root[0], Reveal)


@pytest.mark.parametrize("exhibit", ["plans.csv", "exhibits/plans.txt", "exhibits/../x.png"])
def test_reveal_exhibit_must_be_an_exhibits_png_or_csv(exhibit: str) -> None:
    with pytest.raises(ValidationError):
        Reveal.model_validate({"when_asked": "Q", "answer": "A", "exhibit": exhibit})


def test_topic_meta_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="colour"):
        TopicMeta.model_validate(_topic_meta_data(colour="blue"))


def test_topic_objectives_need_two_to_four_entries() -> None:
    with pytest.raises(ValidationError, match="objectives"):
        TopicMeta.model_validate(_topic_meta_data(objectives=["only one"]))


def test_topic_sources_need_a_known_format() -> None:
    with pytest.raises(ValidationError, match="sources must start with"):
        TopicMeta.model_validate(_topic_meta_data(sources=["wikipedia: Something"]))


def test_topic_requires_must_be_topic_ids() -> None:
    with pytest.raises(ValidationError, match="requires entries must be topic ids"):
        TopicMeta.model_validate(_topic_meta_data(requires=["not an id"]))


def test_cloze_card_must_contain_the_c1_marker() -> None:
    with pytest.raises(ValidationError, match=r"\{\{c1::"):
        ClozeCard.model_validate({"id": "tst/a#1", "type": "cloze", "text": "no marker"})


def test_cards_file_discriminates_on_type() -> None:
    parsed = CardsFile.model_validate(
        {
            "cards": [
                {"id": "tst/a#1", "type": "qa", "front": "q", "back": "a"},
                {"id": "tst/a#2", "type": "cloze", "text": "x {{c1::y}}"},
            ]
        }
    )
    assert [card.type for card in parsed.cards] == ["qa", "cloze"]


def test_card_with_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CardsFile.model_validate({"cards": [{"id": "tst/a#1", "type": "essay", "front": "q"}]})


@pytest.mark.parametrize("entry", ["double all", "class", "1st"])
def test_python_entry_must_be_an_identifier(entry: str) -> None:
    with pytest.raises(ValidationError, match="python entry"):
        ProblemMeta.model_validate(_problem_meta_data("python", entry))


def test_sql_entry_must_be_query() -> None:
    with pytest.raises(ValidationError, match="sql entry must be 'query'"):
        ProblemMeta.model_validate(_problem_meta_data("sql", "select_totals"))


def test_sql_entry_query_is_accepted() -> None:
    meta = ProblemMeta.model_validate(_problem_meta_data("sql", "query"))
    assert meta.language == "sql"


def test_core_path_rejects_end_before_start() -> None:
    data = {
        "id": "core",
        "title": "Core",
        "start": "2026-11-01",
        "end": "2026-10-26",
        "sessions_per_day": 3,
        "session_minutes": 45,
        "days_per_week": 6,
        "weeks": [],
    }
    with pytest.raises(ValidationError, match="before it starts"):
        CorePath.model_validate(data)


def test_models_forbid_extra_keys_and_are_frozen() -> None:
    with pytest.raises(ValidationError):
        Syllabus.model_validate({**_syllabus_data(), "surprise": 1})
    syllabus = Syllabus.model_validate(_syllabus_data())
    with pytest.raises(ValidationError):
        syllabus.version = 2
