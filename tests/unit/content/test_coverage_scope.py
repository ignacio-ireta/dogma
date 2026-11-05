"""The coverage scope and package reader that the API and the coverage report share."""

from __future__ import annotations

import shutil
from pathlib import Path

from dogma.content.coverage_scope import read_packages, scope_syllabus
from dogma.content.schemas import Syllabus

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "content" / "cml" / "vector-operations"


def test_scope_by_tier_keeps_only_that_tier_and_drops_tracks_left_empty(
    real_syllabus: Syllabus,
) -> None:
    scoped = scope_syllabus(real_syllabus, tier="A")

    assert scoped.tracks
    for track in scoped.tracks:
        assert track.topics
        assert all(topic.tier == "A" for topic in track.topics)
        assert track.topic_count == len(track.topics)
    assert {track.code for track in scoped.tracks} <= {t.code for t in real_syllabus.tracks}


def test_scope_by_track_keeps_that_track_with_all_its_tiers(real_syllabus: Syllabus) -> None:
    first = real_syllabus.tracks[0]

    scoped = scope_syllabus(real_syllabus, track=first.code)

    assert [track.code for track in scoped.tracks] == [first.code]
    assert scoped.tracks[0].topics == first.topics


def test_no_filter_keeps_the_whole_syllabus(real_syllabus: Syllabus) -> None:
    assert scope_syllabus(real_syllabus) == real_syllabus


def test_read_packages_returns_loaded_packages_and_names_the_broken_topics(tmp_path: Path) -> None:
    content = tmp_path / "content"
    shutil.copytree(EXAMPLE, content / "cml" / "vector-operations")
    broken = content / "cml" / "broken"
    broken.mkdir(parents=True)
    (broken / "topic.yaml").write_text("id: [unclosed\n", encoding="utf-8")

    packages, unreadable = read_packages(content)

    assert [package.meta.id for package in packages] == ["cml/vector-operations"]
    assert unreadable == ["cml/broken"]
