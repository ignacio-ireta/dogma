"""The seeded sys packages stay valid and verified (I-18b). Rules: see test_case_packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.content.loader import load_topic
from dogma.content.schemas import Syllabus
from dogma.content.validate import validate_topic

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS = (
    "sys/rass-case-framework",
    "sys/churn-prediction-case",
    "sys/fraud-detection-case",
)


@pytest.mark.parametrize("topic_id", SEEDS)
def test_seed_package_has_no_validation_issues(topic_id: str, real_syllabus: Syllabus) -> None:
    package = load_topic(REPO_ROOT / "content" / topic_id)
    assert validate_topic(package, real_syllabus) == []
    assert package.meta.status == "verified"
    assert package.problems == ()


def test_rass_seed_is_a_case_with_a_worked_mini_case_and_interview(
    real_syllabus: Syllabus,
) -> None:
    package = load_topic(REPO_ROOT / "content" / "sys" / "rass-case-framework")
    assert package.meta.kind == "case"
    assert package.interview_present
    assert package.case is not None and package.case.debrief is not None
    assert package.case.debrief.headings == ("Reframe", "Ask", "Structure", "Synthesize")


def test_churn_seed_exhibit_is_loaded(real_syllabus: Syllabus) -> None:
    package = load_topic(REPO_ROOT / "content" / "sys" / "churn-prediction-case")
    assert package.case is not None
    assert package.case.exhibits == ("exhibits/segment-shares.csv",)
