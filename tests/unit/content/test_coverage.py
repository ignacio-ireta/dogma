from __future__ import annotations

import shutil
from pathlib import Path

from dogma.content.coverage import coverage
from dogma.content.loader import TopicPackage, discover_topic_dirs, load_syllabus, load_topic
from dogma.content.schemas import Syllabus


def _packages(fixtures: Path) -> list[TopicPackage]:
    return [load_topic(path) for path in discover_topic_dirs(fixtures / "packages")]


def test_rows_follow_syllabus_order_and_mark_missing_topics(
    fixtures: Path, syllabus: Syllabus
) -> None:
    report = coverage(syllabus, _packages(fixtures))
    assert [row.id for row in report.rows] == ["tst/alpha", "tst/beta", "tst/gamma"]
    alpha, beta, gamma = report.rows
    assert alpha.has_package and alpha.has_lesson and alpha.has_cards and alpha.has_interview
    assert alpha.n_problems == 1 and alpha.status == "draft"
    assert beta.has_package and beta.n_problems == 1
    assert not gamma.has_package and gamma.status == "missing"
    assert gamma.n_problems == 0 and not gamma.has_lesson


def test_summaries_by_track_and_tier(fixtures: Path, syllabus: Syllabus) -> None:
    report = coverage(syllabus, _packages(fixtures))
    (track,) = report.by_track
    assert (track.key, track.topics, track.with_package) == ("tst", 3, 2)
    assert track.statuses == {"draft": 2, "missing": 1}
    tiers = {summary.key: summary for summary in report.by_tier}
    assert (tiers["A"].topics, tiers["A"].with_package, tiers["A"].with_problems) == (2, 1, 1)
    assert (tiers["B"].topics, tiers["B"].with_package) == (1, 1)
    assert tiers["C"].topics == 0


def test_packages_outside_the_syllabus_are_ignored(
    fixtures: Path, syllabus: Syllabus, alpha: Path
) -> None:
    stray = load_topic(alpha)
    report = coverage(syllabus, [stray, stray])
    assert len(report.rows) == 3


def test_real_syllabus_with_no_packages_is_all_missing(real_syllabus: Syllabus) -> None:
    report = coverage(real_syllabus, [])
    assert len(report.rows) == 545
    assert all(row.status == "missing" for row in report.rows)
    assert {summary.key: summary.topics for summary in report.by_tier} == {
        "A": 164,
        "B": 228,
        "C": 153,
    }


def test_case_topics_are_counted_as_authored_content(tmp_path: Path, fixtures: Path) -> None:
    case_dir = tmp_path / "cases" / "tst" / "casey"
    shutil.copytree(fixtures / "cases" / "tst" / "casey", case_dir)
    syllabus = load_syllabus(fixtures / "case-syllabus.yaml")
    report = coverage(syllabus, [load_topic(case_dir)])

    (row,) = report.rows
    assert (row.kind, row.has_package, row.has_lesson, row.has_cards) == ("case", True, True, True)
    assert row.has_interview is False and row.n_problems == 0
    (track,) = report.by_track
    assert (track.cases, track.with_cases) == (1, 1)


def test_topics_without_a_package_are_not_authored_cases(
    fixtures: Path, syllabus: Syllabus
) -> None:
    report = coverage(syllabus, _packages(fixtures))
    assert all(row.kind == "topic" for row in report.rows)
    (track,) = report.by_track
    assert (track.cases, track.with_cases) == (0, 0)
