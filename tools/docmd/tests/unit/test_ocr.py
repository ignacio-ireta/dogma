"""Unit tests for the OCR helpers (pure logic; no Vision engine needed)."""

from __future__ import annotations

import pytest

from docmd.ocr import (
    OcrLine,
    VisionOcrBackend,
    get_ocr_backend,
    group_into_paragraphs,
    is_ocr_available,
    reading_order,
)


def line(text: str, *, x: float = 0.1, y: float = 0.5, w: float = 0.5, h: float = 0.02) -> OcrLine:
    return OcrLine(text=text, confidence=1.0, bbox=(x, y, w, h))


class TestReadingOrder:
    def test_bottom_left_origin_puts_largest_y_first(self) -> None:
        bottom = line("bottom", y=0.1)
        top = line("top", y=0.9)
        middle = line("middle", y=0.5)
        assert [ln.text for ln in reading_order([bottom, top, middle])] == [
            "top",
            "middle",
            "bottom",
        ]

    def test_same_row_sorts_left_to_right(self) -> None:
        right = line("right", x=0.6, y=0.5)
        left = line("left", x=0.1, y=0.5)
        assert [ln.text for ln in reading_order([right, left])] == ["left", "right"]


class TestGroupIntoParagraphs:
    def test_empty_input(self) -> None:
        assert group_into_paragraphs([]) == []

    def test_blank_lines_are_dropped(self) -> None:
        assert group_into_paragraphs([line("   "), line("\t")]) == []

    def test_tight_lines_form_one_paragraph(self) -> None:
        # Line height 0.02; consecutive tops 0.025 apart -> gap 0.005 < 0.6*h.
        lines = [line("first", y=0.900), line("second", y=0.875), line("third", y=0.850)]
        assert group_into_paragraphs(lines) == ["first\nsecond\nthird"]

    def test_large_gap_starts_new_paragraph(self) -> None:
        lines = [
            line("para one a", y=0.900),
            line("para one b", y=0.875),
            line("para two", y=0.700),
        ]
        assert group_into_paragraphs(lines) == ["para one a\npara one b", "para two"]

    def test_zero_height_boxes_never_split(self) -> None:
        lines = [line("a", y=0.9, h=0.0), line("b", y=0.1, h=0.0)]
        assert group_into_paragraphs(lines) == ["a\nb"]

    def test_input_order_does_not_matter(self) -> None:
        lines = [line("second", y=0.875), line("first", y=0.900)]
        assert group_into_paragraphs(lines) == ["first\nsecond"]


class TestAvailability:
    def test_unavailable_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        assert is_ocr_available() is False

    def test_unavailable_without_ocrmac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert is_ocr_available() is False

    def test_available_on_macos_with_ocrmac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
        assert is_ocr_available() is True


def test_get_ocr_backend_builds_vision_backend() -> None:
    backend = get_ocr_backend(languages=["en-US", "de-DE"])
    assert isinstance(backend, VisionOcrBackend)
    assert backend.name == "apple-vision"
    assert backend.languages == ["en-US", "de-DE"]
    assert get_ocr_backend().languages is None
