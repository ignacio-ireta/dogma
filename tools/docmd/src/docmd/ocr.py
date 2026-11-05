"""OCR backends for scanned PDF pages.

v0.2 ships a single backend: Apple's Vision framework via ``ocrmac``, which is
macOS-only and installed through the ``ocr`` extra. The backend protocol keeps
a cross-platform engine (e.g. Tesseract) cheap to add later.

Vision returns line-level observations with normalized bounding boxes in a
bottom-left-origin coordinate system (y grows upward). The pure helpers here
turn those observations into reading order and paragraph groups; they run and
are tested on every platform, while the Vision call itself only runs on macOS.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class OcrLine:
    """One recognized line; bbox is normalized (x, y, w, h), bottom-left origin."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class OcrBackend(Protocol):
    name: str

    def recognize(self, png: bytes) -> list[OcrLine]: ...


def is_ocr_available() -> bool:
    """True when an OCR backend can run here (macOS with the ocr extra installed)."""
    return sys.platform == "darwin" and importlib.util.find_spec("ocrmac") is not None


def get_ocr_backend(*, languages: list[str] | None = None) -> OcrBackend:
    return VisionOcrBackend(languages=languages)


def reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    """Sort observations top-to-bottom, then left-to-right.

    Vision's y axis grows upward, so the top of the page has the largest y.
    """
    return sorted(lines, key=lambda line: (-line.bbox[1], line.bbox[0]))


def group_into_paragraphs(lines: list[OcrLine], *, gap_factor: float = 0.6) -> list[str]:
    """Group recognized lines into paragraph texts by vertical gap.

    A gap larger than ``gap_factor`` x the median line height starts a new
    paragraph. Lines within a paragraph are joined with newlines, which the
    normalizer later folds to spaces.
    """
    ordered = reading_order([line for line in lines if line.text.strip()])
    if not ordered:
        return []
    median_height = statistics.median(height for _, _, _, height in (ln.bbox for ln in ordered))
    threshold = median_height * gap_factor

    paragraphs: list[list[str]] = [[ordered[0].text]]
    previous = ordered[0]
    for line in ordered[1:]:
        # Bottom of the previous line is its y; top of this line is y + h.
        gap = previous.bbox[1] - (line.bbox[1] + line.bbox[3])
        if median_height > 0 and gap > threshold:
            paragraphs.append([line.text])
        else:
            paragraphs[-1].append(line.text)
        previous = line
    return ["\n".join(parts) for parts in paragraphs]


class VisionOcrBackend:
    """Apple Vision via ocrmac; requires macOS and the ``ocr`` extra.

    ``languages`` are BCP-47 codes for Vision's language preference (e.g.
    ``["en-US"]``); ``None`` lets Vision auto-detect. Engine exceptions are
    left to propagate — the PDF extractor wraps them with file context.
    """

    name = "apple-vision"

    def __init__(self, *, languages: list[str] | None = None) -> None:
        self.languages = languages

    def recognize(self, png: bytes) -> list[OcrLine]:  # pragma: no cover - macOS-only
        # glue over pyobjc; exercised for real by the macOS CI job.
        import io

        from ocrmac import ocrmac
        from PIL import Image

        image = Image.open(io.BytesIO(png))
        raw = ocrmac.OCR(
            image,
            recognition_level="accurate",
            language_preference=self.languages,
        ).recognize()
        return [
            OcrLine(
                text=str(text),
                confidence=float(confidence),
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
            for text, confidence, bbox in raw
        ]
