"""``dogma content coverage``: numbers come from the syllabus and the packages on disk."""

from __future__ import annotations

from dogma.content.loader import load_syllabus, load_topic


def test_by_tier_counts_the_syllabus_and_the_one_authored_topic(invoke, settings):
    outcome = invoke("content", "coverage")

    syllabus = load_syllabus(settings.syllabus_path)
    status = load_topic(settings.content_dir / "cml/vector-operations").meta.status
    tier_a = sum(topic.tier == "A" for track in syllabus.tracks for topic in track.topics)
    assert outcome.exit_code == 0, outcome.output
    assert "By tier" in outcome.output
    assert f"1/{tier_a}" in outcome.output
    assert f"{status} 1" in " ".join(outcome.output.split())


def test_tier_filter_adds_per_topic_rows_for_that_tier_only(invoke, settings):
    outcome = invoke("content", "coverage", "--tier", "a")

    syllabus = load_syllabus(settings.syllabus_path)
    tier_b = next(
        topic.id for track in syllabus.tracks for topic in track.topics if topic.tier == "B"
    )
    assert outcome.exit_code == 0, outcome.output
    assert "cml/vector-operations" in outcome.output
    assert "sql/sql-select-filter-and-sort" in outcome.output, "tier A topics are listed"
    assert tier_b not in outcome.output, "tier B topics are filtered out"
    assert "By tier" not in outcome.output, "a tier filter drops the tier summary"


def test_track_filter_limits_the_rows_to_that_track(invoke):
    outcome = invoke("content", "coverage", "--track", "cml")

    assert outcome.exit_code == 0, outcome.output
    assert "cml/vector-operations" in outcome.output
    assert "sql/sql-select-filter-and-sort" not in outcome.output


def test_an_invalid_tier_is_a_usage_error(invoke):
    outcome = invoke("content", "coverage", "--tier", "Z")

    assert outcome.exit_code == 2
    assert "--tier must be A, B or C" in outcome.output


def test_an_unknown_track_is_a_usage_error(invoke):
    outcome = invoke("content", "coverage", "--track", "nope")

    assert outcome.exit_code == 2
    assert "no track 'nope'" in outcome.output


def test_a_broken_package_is_named_with_the_command_that_explains_it(invoke, settings):
    broken = settings.content_dir / "cml" / "broken"
    broken.mkdir(parents=True)
    (broken / "topic.yaml").write_text("id: [unclosed\n", encoding="utf-8")

    outcome = invoke("content", "coverage")

    assert outcome.exit_code == 0, outcome.output
    assert "package not counted: cml/broken" in outcome.output
    assert "dogma content verify cml/broken" in outcome.output
